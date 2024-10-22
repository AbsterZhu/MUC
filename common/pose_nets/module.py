import torch
import torch.nn as nn
from torch.nn import functional as F
from common.pose_nets.layer import make_conv_layers, make_linear_layers, make_deconv_layers
from common.utils.transforms import sample_joint_features, soft_argmax_2d, soft_argmax_3d
from common.utils.human_models import smpl_x
from common.pose_nets.config import cfg
from mmcv.ops.roi_align import roi_align
import torchvision.models as models

class PositionNet(nn.Module):
    def __init__(self, part, feat_dim=768):
        super(PositionNet, self).__init__()
        if part == 'body':
            self.joint_num = len(smpl_x.pos_joint_part['body'])
            self.hm_shape = cfg.output_hm_shape
        elif part == 'hand':
            self.joint_num = len(smpl_x.pos_joint_part['rhand'])
            self.hm_shape = cfg.output_hand_hm_shape
        self.conv = make_conv_layers([feat_dim, self.joint_num * self.hm_shape[0]], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def forward(self, img_feat):
        joint_hm = self.conv(img_feat).view(-1, self.joint_num, self.hm_shape[0], self.hm_shape[1], self.hm_shape[2])
        joint_coord = soft_argmax_3d(joint_hm)
        joint_hm = F.softmax(joint_hm.view(-1, self.joint_num, self.hm_shape[0] * self.hm_shape[1] * self.hm_shape[2]), 2)
        joint_hm = joint_hm.view(-1, self.joint_num, self.hm_shape[0], self.hm_shape[1], self.hm_shape[2])
        return joint_hm, joint_coord

class HandRotationNet(nn.Module):
    def __init__(self, part, feat_dim = 768):
        super(HandRotationNet, self).__init__()
        self.part = part
        self.joint_num = len(smpl_x.pos_joint_part['rhand'])
        self.hand_conv = make_conv_layers([feat_dim, 512], kernel=1, stride=1, padding=0)
        self.hand_pose_out = make_linear_layers([self.joint_num * 515, len(smpl_x.orig_joint_part['rhand']) * 6], relu_final=False)
        self.feat_dim = feat_dim

    def forward(self, img_feat, joint_coord_img):
        batch_size = img_feat.shape[0]
        img_feat = self.hand_conv(img_feat)
        img_feat_joints = sample_joint_features(img_feat, joint_coord_img[:, :, :2])
        feat = torch.cat((img_feat_joints, joint_coord_img), 2)  # batch_size, joint_num, 512+3
        hand_pose = self.hand_pose_out(feat.view(batch_size, -1))
        return hand_pose

class BodyRotationNet(nn.Module):
    def __init__(self, feat_dim = 768):
        super(BodyRotationNet, self).__init__()
        self.joint_num = len(smpl_x.pos_joint_part['body'])
        self.body_conv = make_linear_layers([feat_dim, 512], relu_final=False)
        self.root_pose_out = make_linear_layers([self.joint_num * (512+3), 6], relu_final=False)
        self.body_pose_out = make_linear_layers(
            [self.joint_num * (512+3), (len(smpl_x.orig_joint_part['body']) - 1) * 6], relu_final=False)  # without root
        self.shape_out = make_linear_layers([feat_dim, smpl_x.shape_param_dim], relu_final=False)
        self.cam_out = make_linear_layers([feat_dim, 3], relu_final=False)
        self.feat_dim = feat_dim

    def forward(self, body_pose_token, shape_token, cam_token, body_joint_img):
        batch_size = body_pose_token.shape[0]

        # shape parameter
        shape_param = self.shape_out(shape_token)

        # camera parameter
        cam_param = self.cam_out(cam_token)

        # body pose parameter
        body_pose_token = self.body_conv(body_pose_token)
        body_pose_token = torch.cat((body_pose_token, body_joint_img), 2)
        root_pose = self.root_pose_out(body_pose_token.view(batch_size, -1))
        body_pose = self.body_pose_out(body_pose_token.view(batch_size, -1))

        return root_pose, body_pose, shape_param, cam_param

class FaceRegressor(nn.Module):
    def __init__(self, feat_dim=768):
        super(FaceRegressor, self).__init__()
        self.expr_out = make_linear_layers([feat_dim, smpl_x.expr_code_dim], relu_final=False)
        self.jaw_pose_out = make_linear_layers([feat_dim, 6], relu_final=False)

    def forward(self, expr_token, jaw_pose_token):
        expr_param = self.expr_out(expr_token)  # expression parameter
        jaw_pose = self.jaw_pose_out(jaw_pose_token)  # jaw pose parameter
        return expr_param, jaw_pose

class BoxNet(nn.Module):
    def __init__(self, feat_dim=768):
        super(BoxNet, self).__init__()
        self.joint_num = len(smpl_x.pos_joint_part['body'])
        self.deconv = make_deconv_layers([feat_dim + self.joint_num * cfg.output_hm_shape[0], 256, 256, 256])
        self.bbox_center = make_conv_layers([256, 3], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.lhand_size = make_linear_layers([256, 256, 2], relu_final=False)
        self.rhand_size = make_linear_layers([256, 256, 2], relu_final=False)
        self.face_size = make_linear_layers([256, 256, 2], relu_final=False)

    def forward(self, img_feat, joint_hm):
        joint_hm = joint_hm.view(joint_hm.shape[0], joint_hm.shape[1] * cfg.output_hm_shape[0], cfg.output_hm_shape[1], cfg.output_hm_shape[2])
        img_feat = torch.cat((img_feat, joint_hm), 1)
        img_feat = self.deconv(img_feat)

        # bbox center
        bbox_center_hm = self.bbox_center(img_feat)
        bbox_center = soft_argmax_2d(bbox_center_hm)
        lhand_center, rhand_center, face_center = bbox_center[:, 0, :], bbox_center[:, 1, :], bbox_center[:, 2, :]

        # bbox size
        lhand_feat = sample_joint_features(img_feat, lhand_center[:, None, :].detach())[:, 0, :]
        lhand_size = self.lhand_size(lhand_feat)
        rhand_feat = sample_joint_features(img_feat, rhand_center[:, None, :].detach())[:, 0, :]
        rhand_size = self.rhand_size(rhand_feat)
        face_feat = sample_joint_features(img_feat, face_center[:, None, :].detach())[:, 0, :]
        face_size = self.face_size(face_feat)

        lhand_center = lhand_center / 8
        rhand_center = rhand_center / 8
        face_center = face_center / 8
        return lhand_center, lhand_size, rhand_center, rhand_size, face_center, face_size

class BoxSizeNet(nn.Module):
    def __init__(self):
        super(BoxSizeNet, self).__init__()
        self.lhand_size = make_linear_layers([256, 256, 2], relu_final=False)
        self.rhand_size = make_linear_layers([256, 256, 2], relu_final=False)
        self.face_size = make_linear_layers([256, 256, 2], relu_final=False)

    def forward(self, box_fea):
        # box_fea: [bs, 3, C]
        lhand_size = self.lhand_size(box_fea[:, 0])
        rhand_size = self.rhand_size(box_fea[:, 1])
        face_size = self.face_size(box_fea[:, 2])
        return lhand_size, rhand_size, face_size

class HandRoI(nn.Module):
    def __init__(self, feat_dim=768, upscale=4):
        super(HandRoI, self).__init__()
        self.upscale = upscale
        if upscale==1:
            self.deconv = make_conv_layers([feat_dim, feat_dim], kernel=1, stride=1, padding=0, bnrelu_final=False)
            self.conv = make_conv_layers([feat_dim, feat_dim], kernel=1, stride=1, padding=0, bnrelu_final=False)
        elif upscale==2:
            self.deconv = make_deconv_layers([feat_dim, feat_dim//2])
            self.conv = make_conv_layers([feat_dim//2, feat_dim], kernel=1, stride=1, padding=0, bnrelu_final=False)
        elif upscale==4:
            self.deconv = make_deconv_layers([feat_dim, feat_dim//2, feat_dim//4])
            self.conv = make_conv_layers([feat_dim//4, feat_dim], kernel=1, stride=1, padding=0, bnrelu_final=False)
        elif upscale==8:
            self.deconv = make_deconv_layers([feat_dim, feat_dim//2, feat_dim//4, feat_dim//8])
            self.conv = make_conv_layers([feat_dim//8, feat_dim], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def forward(self, img_feat, lhand_bbox, rhand_bbox):
        lhand_bbox = torch.cat((torch.arange(lhand_bbox.shape[0]).float().cuda()[:, None], lhand_bbox),
                               1)  # batch_idx, xmin, ymin, xmax, ymax
        rhand_bbox = torch.cat((torch.arange(rhand_bbox.shape[0]).float().cuda()[:, None], rhand_bbox),
                               1)  # batch_idx, xmin, ymin, xmax, ymax
        img_feat = self.deconv(img_feat)
        lhand_bbox_roi = lhand_bbox.clone()
        lhand_bbox_roi[:, 1] = lhand_bbox_roi[:, 1] / cfg.input_body_shape[1] * cfg.output_hm_shape[2] * self.upscale
        lhand_bbox_roi[:, 2] = lhand_bbox_roi[:, 2] / cfg.input_body_shape[0] * cfg.output_hm_shape[1] * self.upscale
        lhand_bbox_roi[:, 3] = lhand_bbox_roi[:, 3] / cfg.input_body_shape[1] * cfg.output_hm_shape[2] * self.upscale
        lhand_bbox_roi[:, 4] = lhand_bbox_roi[:, 4] / cfg.input_body_shape[0] * cfg.output_hm_shape[1] * self.upscale
        assert (cfg.output_hm_shape[1]*self.upscale, cfg.output_hm_shape[2]*self.upscale) == (img_feat.shape[2], img_feat.shape[3])
        lhand_img_feat = roi_align(img_feat, lhand_bbox_roi, (cfg.output_hand_hm_shape[1], cfg.output_hand_hm_shape[2]), 1.0, 0, 'avg', False)
        lhand_img_feat = torch.flip(lhand_img_feat, [3])  # flip to the right hand

        rhand_bbox_roi = rhand_bbox.clone()
        rhand_bbox_roi[:, 1] = rhand_bbox_roi[:, 1] / cfg.input_body_shape[1] * cfg.output_hm_shape[2] * self.upscale
        rhand_bbox_roi[:, 2] = rhand_bbox_roi[:, 2] / cfg.input_body_shape[0] * cfg.output_hm_shape[1] * self.upscale
        rhand_bbox_roi[:, 3] = rhand_bbox_roi[:, 3] / cfg.input_body_shape[1] * cfg.output_hm_shape[2] * self.upscale
        rhand_bbox_roi[:, 4] = rhand_bbox_roi[:, 4] / cfg.input_body_shape[0] * cfg.output_hm_shape[1] * self.upscale
        rhand_img_feat = roi_align(img_feat, rhand_bbox_roi, (cfg.output_hand_hm_shape[1], cfg.output_hand_hm_shape[2]), 1.0, 0, 'avg', False)
        hand_img_feat = torch.cat((lhand_img_feat, rhand_img_feat))  # [bs, c, cfg.output_hand_hm_shape[2]*scale, cfg.output_hand_hm_shape[1]*scale]
        hand_img_feat = self.conv(hand_img_feat)
        return hand_img_feat
    
class FaceRoI(nn.Module):
    def __init__(self, feat_dim=768, upscale=4):
        super(FaceRoI, self).__init__()
        self.upscale = upscale
        if upscale==1:
            self.deconv = make_conv_layers([feat_dim, feat_dim], kernel=1, stride=1, padding=0, bnrelu_final=False)
            self.conv = make_conv_layers([feat_dim, feat_dim], kernel=1, stride=1, padding=0, bnrelu_final=False)
        elif upscale==2:
            self.deconv = make_deconv_layers([feat_dim, feat_dim//2])
            self.conv = make_conv_layers([feat_dim//2, feat_dim], kernel=1, stride=1, padding=0, bnrelu_final=False)
        elif upscale==4:
            self.deconv = make_deconv_layers([feat_dim, feat_dim//2, feat_dim//4])
            self.conv = make_conv_layers([feat_dim//4, feat_dim], kernel=1, stride=1, padding=0, bnrelu_final=False)
        elif upscale==8:
            self.deconv = make_deconv_layers([feat_dim, feat_dim//2, feat_dim//4, feat_dim//8])
            self.conv = make_conv_layers([feat_dim//8, feat_dim], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def forward(self, img_feat, face_bbox):
        face_bbox = torch.cat((torch.arange(face_bbox.shape[0]).float().cuda()[:, None], face_bbox),
                               1)  # batch_idx, xmin, ymin, xmax, ymax
        img_feat = self.deconv(img_feat)
        face_bbox_roi = face_bbox.clone()
        face_bbox_roi[:, 1] = face_bbox_roi[:, 1] / cfg.input_body_shape[1] * cfg.output_hm_shape[2] * self.upscale
        face_bbox_roi[:, 2] = face_bbox_roi[:, 2] / cfg.input_body_shape[0] * cfg.output_hm_shape[1] * self.upscale
        face_bbox_roi[:, 3] = face_bbox_roi[:, 3] / cfg.input_body_shape[1] * cfg.output_hm_shape[2] * self.upscale
        face_bbox_roi[:, 4] = face_bbox_roi[:, 4] / cfg.input_body_shape[0] * cfg.output_hm_shape[1] * self.upscale
        assert (cfg.output_hm_shape[1]*self.upscale, cfg.output_hm_shape[2]*self.upscale) == (img_feat.shape[2], img_feat.shape[3])
        face_img_feat = roi_align(img_feat, face_bbox_roi, (cfg.output_face_hm_shape[1], cfg.output_face_hm_shape[2]), 1.0, 0, 'avg', False)
        face_img_feat = self.conv(face_img_feat)
        return face_img_feat
    
    
class ResNetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(ResNetBlock2D, self).__init__()
        self.groupnorm1 = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        self.silu1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.silu2 = nn.SiLU()
        self.groupnorm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Residual connection
        self.residual_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, model_input):
        x, condition = model_input
        residual = self.residual_connection(x)

        out = self.groupnorm1(x)
        out = self.silu1(out)
        out = self.conv1(out)

        out = self.silu2(out)
        out = self.groupnorm2(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out += residual

        return (out, condition)

class Transformer2DModel(nn.Module):
    def __init__(self, in_channels, condition_channels, img_shape, condition_length, num_heads=8, dropout_rate=0.2):
        super(Transformer2DModel, self).__init__()

        # GroupNorm, Conv2d, and Linear for condition
        self.condition_groupnorm = nn.GroupNorm(num_groups=32, num_channels=condition_channels)
        self.condition_conv = nn.Conv2d(condition_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.condition_linear = nn.Linear(condition_length, img_shape[0] * img_shape[1])

        # Cross-Attention
        self.cross_attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)

        # LayerNorm, linear, and Dropout
        self.groupnorm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.groupnorm2 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.linear = nn.Linear(img_shape[0] * img_shape[1], img_shape[0] * img_shape[1])
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, model_input):
        x, condition = model_input
        batch, inner_dim, height, width = x.shape
        residual = x
        
        # GroupNorm, Conv2d, and Linear for condition
        c = condition.clone()
        c = self.condition_groupnorm(c)
        c = self.condition_linear(c)
        c = c.view(batch, -1, height, width)
        c = self.condition_conv(c)
        # 使用 interpolate 函数进行缩小
        # c = F.interpolate(c, size=(height, width), mode='bilinear', align_corners=False)
        c = c.view(batch, inner_dim, height * width)

        # Norm
        x = self.groupnorm1(x)

        # Cross-Attention
        x = x.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim) 
        c = c.permute(0, 2, 1)
        x, _ = self.cross_attention(x, c, c)
        x = x.permute(0, 2, 1).reshape(batch, inner_dim, height * width)
        
        x += residual.reshape(batch, inner_dim, height * width)

        # LayerNorm, and Dropout
        x = self.groupnorm2(x)
        x = self.gelu(x)
        x = self.dropout(x).reshape(batch, inner_dim, height * width)
        x = self.linear(x).reshape(batch, inner_dim, height, width)

        return (x, condition)

class Downsample2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, model_input):
        x, condition = model_input
        x = self.conv(x)
        return (x, condition)

class Upsample2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample2D, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, model_input):
        x, condition = model_input
        x = self.upconv(x)
        return (x, condition)

class UNet(nn.Module):
    def __init__(self, condition_length, shape):
        super(UNet, self).__init__()
        # Downsample blocks
        self.downsample1 = nn.Sequential(
            ResNetBlock2D(3, 64),
            Transformer2DModel(64, cfg.feat_dim, (shape[0], shape[1]), condition_length),
            Downsample2D(64, 128)
        )
        self.downsample2 = nn.Sequential(
            ResNetBlock2D(128, 256),
            Transformer2DModel(256, cfg.feat_dim, (shape[0]//2, shape[1]//2), condition_length),
            Downsample2D(256, 256)
        )

        # Middle block
        self.middle = nn.Sequential(
            ResNetBlock2D(256, 512),
            Transformer2DModel(512, cfg.feat_dim, (shape[0]//4, shape[1]//4), condition_length),
            ResNetBlock2D(512, 256)
        )

        # Upsample blocks
        self.upsample1 = nn.Sequential(
            ResNetBlock2D(512, 256),
            Transformer2DModel(256, cfg.feat_dim, (shape[0]//4, shape[1]//4), condition_length),
            Upsample2D(256, 128)
        )
        self.upsample2 = nn.Sequential(
            ResNetBlock2D(256, 128),
            Transformer2DModel(128, cfg.feat_dim, (shape[0]//2, shape[1]//2), condition_length),
            Upsample2D(128, 64)
        )

        # Output layer
        self.output_layer = nn.Conv2d(64, 3, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, model_input):
        x, condition = model_input
        # Downsample path
        skip1, condition = self.downsample1((x, condition))
        skip2, condition = self.downsample2((skip1, condition))

        # Middle path
        middle_out, condition = self.middle((skip2, condition))

        # Upsample path
        upsample1_out, condition = self.upsample1((torch.cat([middle_out, skip2], dim=1), condition))
        upsample2_out, condition = self.upsample2((torch.cat([upsample1_out, skip1], dim=1), condition))

        # Output layer
        output = self.output_layer(upsample2_out)
        return output
    
class UVOut(nn.Module):
    def __init__(self, num_classes=10, input_channels=3, input_size=64):
        super(UVOut, self).__init__()
        # 使用预训练的ResNet-18模型
        resnet18 = models.resnet18(pretrained=False)
        
        # 修改第一层卷积层以适应新的输入大小
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(resnet18.conv1.weight[:, :input_channels, :, :])
        
        # 提取ResNet-18模型的前卷积部分
        self.features = nn.Sequential(*list(resnet18.children())[1:-2])
        
        # 添加自定义的全局平均池化层和分类层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 前向传播
        x = self.conv1(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        return x