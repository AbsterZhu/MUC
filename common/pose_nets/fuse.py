import torch
import os
import copy
import cv2
import trimesh
import numpy as np

from torch import nn
from smplx import SMPLX
from common.utils.human_models import smpl_x
from common.pose_nets.config import cfg
from common.pose_nets.layer import make_conv_layers, make_linear_layers
from common.pose_nets.module import UNet, UVOut
from common.pose_nets.module import FaceRoI
from pytorchltr.loss import LambdaNDCGLoss2
from common.pose_nets.loss import CoordLoss, ParamLoss, CELoss



class KLDivergenceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(KLDivergenceLoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, depths, preds):
        # 假设preds已经通过softmax或类似函数转换成概率分布
        # 计算基于深度的目标分布
        target_probs = 1 / (depths + self.epsilon)
        target_probs /= target_probs.sum(dim=1, keepdim=True)  # 按行归一化
        
        # 计算KL散度
        kl_div = target_probs * torch.log(target_probs / (preds + self.epsilon))
        return kl_div.sum()



class Fusion_Net(nn.Module):
    def __init__(self, encoder, data_info):
        super(Fusion_Net, self).__init__()
        
        self.encoder = encoder
        self.data_info = data_info
        
        self.body_joint_num = len(smpl_x.pos_joint_part['body'])
        self.hm_shape = cfg.output_hm_shape
        self.hand_hm_shape = cfg.output_hand_hm_shape
        self.face_hm_shape = cfg.output_face_hm_shape
        
        if not cfg.froze:
            for param in self.encoder.parameters():
                param.requires_grad = True
        else:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        self.body_conv = make_conv_layers(
            [cfg.feat_dim, self.body_joint_num + 2], 
            kernel=1, stride=1, padding=0, bnrelu_final=True
        )
        self.body_code = make_linear_layers([cfg.feat_dim, 512], relu_final=True)
        self.body_att = make_linear_layers(
            [(self.body_joint_num + 2) * (512+self.hm_shape[1]*self.hm_shape[2]),
             1024], relu_final=True
        )
        self.body_out = make_linear_layers([1024, 512, 66], relu_final=False)
        
        self.hand_conv = make_conv_layers([cfg.feat_dim, 2], kernel=1, stride=1, padding=0, bnrelu_final=True)
        self.hand_code = make_linear_layers([cfg.feat_dim, self.hand_hm_shape[1]*self.hand_hm_shape[2]], relu_final=True)
        hand_matt_size = (4) * (self.hand_hm_shape[1]*self.hand_hm_shape[2]) * 2
        self.hand_att = make_linear_layers([hand_matt_size, 512, 90], relu_final=False)
        
        self.face_roi_net = FaceRoI(feat_dim=cfg.feat_dim, upscale=cfg.upscale)
        
        self.expr_code = make_linear_layers([3, 10], relu_final=False)
        self.expr_unet = UNet(10, cfg.output_face_uv_shape)
        self.expr_out = UVOut()
        
        self.shape_code = make_linear_layers([2, 10], relu_final=False)
        self.shape_unet = UNet(10, cfg.output_body_uv_shape)
        self.shape_out = UVOut()
        
        self.body_model = SMPLX(
            model_path=os.path.join(cfg.human_model_path, 'smplx'),
            gender='neutral',
            use_pca=False,
            flat_hand_mean=False,
            use_face_contour=True,
            create_expression=True,
            create_jaw_pose=True,
        )
        
        self.smplx_layer = copy.deepcopy(smpl_x.layer['neutral']).cuda()
        
        J_regressor = copy.deepcopy(smpl_x.layer['neutral']).J_regressor
        self.J_regressor = torch.cat((J_regressor[1:22, :], J_regressor[25:, :], J_regressor[22, :].view((1, -1))), dim=0)
        
        self.loss_fuc = torch.nn.L1Loss()
        self.loss_fuc_rank = KLDivergenceLoss()
        self.coord_loss = CoordLoss()
                
        uv_map_mask = cv2.resize(cv2.imread(cfg.uv_mask_path), cfg.input_body_uv_shape)
        self.uv_map_mask = (torch.tensor(uv_map_mask) / 255).permute(2, 0, 1)
        self.uv_mesh = trimesh.load(cfg.uv_mesh_path)
        self.uv_mesh.merge_vertices(merge_tex=True)
        self.uv_mesh_vt = self.uv_mesh.visual.uv
        self.uv_mesh_vt[:,1] = 1 - self.uv_mesh_vt[:,1]
        self.uv_mesh_vt = (self.uv_mesh_vt * self.uv_map_mask.shape[1]).astype(np.int16)
        
        self.uv_head_bbox = [0, 0, cfg.input_face_uv_shape[1], cfg.input_face_uv_shape[0]]
        
        
    def get_smplx_dic(self, smplx_params):
        batch_size = smplx_params.shape[0]
        shape = torch.zeros((batch_size, 10)).float().cuda()
        zero_pose = torch.zeros((batch_size, 3)).float().cuda()
        expr = torch.zeros((batch_size, 10)).float().cuda()
        
        body_pose = smplx_params[:, :63]
        if smplx_params.shape[1] < 108:
            lhand = torch.zeros((batch_size, 45)).float()
            rhand = torch.zeros((batch_size, 45)).float()
            jaw = torch.zeros((batch_size, 3)).float()
        else:
            lhand = smplx_params[:, 63:63+45]
            rhand = smplx_params[:, 63+45:63+45+45]
            jaw = smplx_params[:, 63+45+45:63+45+45+3]
        
        dict = {'betas': shape, 'body_pose': body_pose, 'global_orient': zero_pose,
                'right_hand_pose': rhand.cuda(), 'left_hand_pose': lhand.cuda(),
                'jaw_pose': jaw.cuda(), 'leye_pose': zero_pose, 'reye_pose': zero_pose,
                'expression': expr}
        
        return dict
    
    
    def get_coord(self, root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expr, cam_trans, mode):
        batch_size = root_pose.shape[0]
        zero_pose = torch.zeros((1, 3)).float().cuda().repeat(batch_size, 1)  # eye poses
        output = self.smplx_layer(betas=shape, body_pose=body_pose, global_orient=root_pose, right_hand_pose=rhand_pose,
                                  left_hand_pose=lhand_pose, jaw_pose=jaw_pose, leye_pose=zero_pose,
                                  reye_pose=zero_pose, expression=expr)
        # camera-centered 3D coordinate
        mesh_cam = output.vertices
        if mode == 'test' and cfg.testset == 'AGORA':  # use 144 joints for AGORA evaluation
            joint_cam = output.joints
        else:
            joint_cam = output.joints[:, smpl_x.joint_idx, :]

        # project 3D coordinates to 2D space
        if mode == 'train' and len(cfg.trainset_3d) == 1 and cfg.trainset_3d[0] == 'AGORA' and len(
                cfg.trainset_2d) == 0:  # prevent gradients from backpropagating to SMPLX paraemter regression module
            x = (joint_cam[:, :, 0].detach() + cam_trans[:, None, 0]) / (
                    joint_cam[:, :, 2].detach() + cam_trans[:, None, 2] + 1e-4) * cfg.focal[0] + cfg.princpt[0]
            y = (joint_cam[:, :, 1].detach() + cam_trans[:, None, 1]) / (
                    joint_cam[:, :, 2].detach() + cam_trans[:, None, 2] + 1e-4) * cfg.focal[1] + cfg.princpt[1]
        else:
            x = (joint_cam[:, :, 0] + cam_trans[:, None, 0]) / (joint_cam[:, :, 2] + cam_trans[:, None, 2] + 1e-4) * \
                cfg.focal[0] + cfg.princpt[0]
            y = (joint_cam[:, :, 1] + cam_trans[:, None, 1]) / (joint_cam[:, :, 2] + cam_trans[:, None, 2] + 1e-4) * \
                cfg.focal[1] + cfg.princpt[1]
        x = x / cfg.input_body_shape[1] * cfg.output_hm_shape[2]
        y = y / cfg.input_body_shape[0] * cfg.output_hm_shape[1]
        joint_proj = torch.stack((x, y), 2)

        # root-relative 3D coordinates
        root_cam = joint_cam[:, smpl_x.root_joint_idx, None, :]
        joint_cam = joint_cam - root_cam
        mesh_cam = mesh_cam + cam_trans[:, None, :]  # for rendering
        joint_cam_wo_ra = joint_cam.clone()

        # left hand root (left wrist)-relative 3D coordinatese
        lhand_idx = smpl_x.joint_part['lhand']
        lhand_cam = joint_cam[:, lhand_idx, :]
        lwrist_cam = joint_cam[:, smpl_x.lwrist_idx, None, :]
        lhand_cam = lhand_cam - lwrist_cam
        joint_cam = torch.cat((joint_cam[:, :lhand_idx[0], :], lhand_cam, joint_cam[:, lhand_idx[-1] + 1:, :]), 1)

        # right hand root (right wrist)-relative 3D coordinatese
        rhand_idx = smpl_x.joint_part['rhand']
        rhand_cam = joint_cam[:, rhand_idx, :]
        rwrist_cam = joint_cam[:, smpl_x.rwrist_idx, None, :]
        rhand_cam = rhand_cam - rwrist_cam
        joint_cam = torch.cat((joint_cam[:, :rhand_idx[0], :], rhand_cam, joint_cam[:, rhand_idx[-1] + 1:, :]), 1)

        # face root (neck)-relative 3D coordinates
        face_idx = smpl_x.joint_part['face']
        face_cam = joint_cam[:, face_idx, :]
        neck_cam = joint_cam[:, smpl_x.neck_idx, None, :]
        face_cam = face_cam - neck_cam
        joint_cam = torch.cat((joint_cam[:, :face_idx[0], :], face_cam, joint_cam[:, face_idx[-1] + 1:, :]), 1)

        return joint_proj, joint_cam, joint_cam_wo_ra, mesh_cam
        
        
    def forward(self, multi_view_inputs, multi_view_targets, multi_view_meta_info, model_status):
        view_num = len(multi_view_inputs.keys())
        out = {}
        
        global_smplx_pose = None
        global_smplx_root = None
        global_smplx_shape = None
        global_smplx_expr = None
        
        global_pose_weight = None
        global_shape_weight = None
        global_expr_weight = None
        
        global_shape_condition = None
        global_expr_condition = None
        
        global_body_depth_joint = None
        
        view_count = 0
        for view in multi_view_inputs.keys():
            single_loss, single_out = self.encoder(multi_view_inputs[view], 
                                                   multi_view_targets[view], 
                                                   multi_view_meta_info[view], 
                                                   model_status)
            
            if view_count == 0:
                global_body_depth_joint = multi_view_targets[view]["joint_img"][:, smpl_x.pos_joint_part['body'], 2].unsqueeze(0)
            else:
                global_body_depth_joint = torch.cat((global_body_depth_joint, 
                                                     multi_view_targets[view]["joint_img"][:, smpl_x.pos_joint_part['body'], 2].unsqueeze(0)), 0)
            
            if model_status == 'train':
                if view_count == 0:
                    out["loss_joint_cam"] = single_loss["joint_cam"]
                    out["loss_smplx_joint_cam"] = single_loss["smplx_joint_cam"]
                    out["loss_joint_proj"] = single_loss["joint_proj"]
                    out["loss_joint_img"] = single_loss["joint_img"]
                    out["loss_smplx_joint_img"] = single_loss["smplx_joint_img"]
                    
                    if "lhand_bbox" in single_loss.keys():
                        out["loss_lhand_bbox"] = single_loss["lhand_bbox"]
                    if "rhand_bbox" in single_loss.keys():
                        out["loss_rhand_bbox"] = single_loss["rhand_bbox"]
                    if "face_bbox" in single_loss.keys():
                        out["loss_face_bbox"] = single_loss["face_bbox"]
                else:
                    out["loss_joint_cam"] += single_loss["joint_cam"]
                    out["loss_smplx_joint_cam"] += single_loss["smplx_joint_cam"]
                    out["loss_joint_proj"] += single_loss["joint_proj"]
                    out["loss_joint_img"] += single_loss["joint_img"]
                    out["loss_smplx_joint_img"] += single_loss["smplx_joint_img"]
                    
                    if "lhand_bbox" in single_loss.keys():
                        if "loss_lhand_bbox" not in out.keys():
                            out["loss_lhand_bbox"] = single_loss["lhand_bbox"]
                        else:
                            out["loss_lhand_bbox"] += single_loss["lhand_bbox"]
                    if "rhand_bbox" in single_loss.keys():
                        if "loss_rhand_bbox" not in out.keys():
                            out["loss_rhand_bbox"] = single_loss["rhand_bbox"]
                        else:
                            out["loss_rhand_bbox"] += single_loss["rhand_bbox"]
                    if "face_bbox" in single_loss.keys():
                        if "loss_face_bbox" not in out.keys():
                            out["loss_face_bbox"] = single_loss["face_bbox"]
                        else:
                            out["loss_face_bbox"] += single_loss["face_bbox"]
                
            out["view_" + view] = single_out
            
            # feature
            img_feat = single_out['img_feat']
            batch_size = img_feat.shape[0]
            hand_feat = single_out['hand_feat']
            body_pose_token = single_out['body_pose_token']
            hand_pose_token = single_out['hand_pose_token']
            cam_token = single_out['cam_token'].view(batch_size, 1, -1)
            jaw_pose_token = single_out['jaw_pose_token'].view(batch_size, 1, -1)
            expr_token = single_out['expr_token'].view(batch_size, 1, -1)
            shape_token = single_out['shape_token'].view(batch_size, 1, -1)
            
            # output
            smplx_root_pose = single_out['smplx_root_pose'].view(batch_size, 1, -1)
            smplx_body_pose = single_out['smplx_body_pose'].view(batch_size, 1, -1)
            smplx_lhand_pose = single_out['smplx_lhand_pose'].view(batch_size, 1, -1)
            smplx_rhand_pose = single_out['smplx_rhand_pose'].view(batch_size, 1, -1)
            smplx_jaw_pose = single_out['smplx_jaw_pose'].view(batch_size, 1, -1)
            smplx_shape = single_out['smplx_shape'].view(batch_size, 1, -1)
            smplx_expr = single_out['smplx_expr'].view(batch_size, 1, -1)
            
            smplx_pose = torch.cat((smplx_body_pose, smplx_lhand_pose, smplx_rhand_pose, smplx_jaw_pose), 2)
            
            if view_count == 0:
                global_smplx_pose = smplx_pose
                global_smplx_root = smplx_root_pose
                global_smplx_shape = smplx_shape
                global_smplx_expr = smplx_expr
            else:
                global_smplx_pose = torch.cat((global_smplx_pose, smplx_pose), 1)
                global_smplx_root = torch.cat((global_smplx_root, smplx_root_pose), 1)
                global_smplx_shape = torch.cat((global_smplx_shape, smplx_shape), 1)
                global_smplx_expr = torch.cat((global_smplx_expr, smplx_expr), 1)
            
            # body
            body_feat = self.body_conv(img_feat).view(batch_size, self.body_joint_num + 2, -1)
            body_pose_token = torch.cat((body_pose_token, jaw_pose_token, cam_token), 1)
            body_pose_token = self.body_code(body_pose_token)
            body_attention = self.body_att(torch.cat((body_pose_token, body_feat), 2).view(batch_size, -1))
            pose_weight = self.body_out(body_attention).view(1, batch_size, -1) # v*b*66
            
            # hand
            hand_feat = self.hand_conv(hand_feat).view(batch_size * 2, 2, -1)
            lhand_feat = hand_feat[:batch_size]
            rhand_feat = hand_feat[batch_size:]
            hand_feat = torch.cat((lhand_feat, rhand_feat), 1)
            hand_pose_token = torch.cat((hand_pose_token[:, :1, :], cam_token,
                                         hand_pose_token[:, 1:, :], cam_token), 1)
            hand_pose_token = self.hand_code(hand_pose_token)
            hand_attention = self.hand_att(torch.cat((hand_pose_token, hand_feat), 2).view(1, batch_size, -1))
            
            # whole body
            pose_weight = torch.cat((pose_weight[:, :, :63], hand_attention, pose_weight[:, :, 63:]), 2) # v*b*156
            if view_count == 0:
                global_pose_weight = pose_weight
            else:
                global_pose_weight = torch.cat((global_pose_weight, pose_weight), 0)
                
                
            # shape & expression
            if cfg.srn_loss:
                # expression
                expr_token = torch.cat((expr_token, jaw_pose_token, cam_token), 1).permute(0, 2, 1)
                expr_condition = self.expr_code(expr_token).view(1, batch_size, -1, 10)
                
                # body shape
                shape_token = torch.cat((shape_token, cam_token), 1).permute(0, 2, 1)
                shape_condition = self.shape_code(shape_token).view(1, batch_size, -1, 10)
                
                if view_count == 0:
                    global_shape_condition = shape_condition
                    global_expr_condition = expr_condition
                else:
                    global_shape_condition = torch.cat((global_shape_condition, shape_condition), 0)
                    global_expr_condition = torch.cat((global_expr_condition, expr_condition), 0)
                
                shape_weight = torch.ones((1, batch_size, 10)).float().cuda()
                expr_weight = torch.ones((1, batch_size, 10)).float().cuda()
                
            else:
                pass
                shape_weight = torch.ones((1, batch_size, 10)).float().cuda()
                expr_weight = torch.ones((1, batch_size, 10)).float().cuda()
                
            if view_count == 0:
                global_shape_weight = shape_weight
                global_expr_weight = expr_weight
            else:
                global_shape_weight = torch.cat((global_shape_weight, shape_weight), 0)
                global_expr_weight = torch.cat((global_expr_weight, expr_weight), 0)
                
            view_count += 1
        
        global_smplx_pose = global_smplx_pose.permute(1, 0, 2)
        global_smplx_root = global_smplx_root.permute(1, 0, 2)
        global_smplx_shape = global_smplx_shape.permute(1, 0, 2)
        global_smplx_expr = global_smplx_expr.permute(1, 0, 2)
        
        global_shape_weight = nn.Softmax(dim=0)(global_shape_weight)
        global_expr_weight = nn.Softmax(dim=0)(global_expr_weight)
        global_pose_weight = nn.Softmax(dim=0)(global_pose_weight)
        
        global_smplx_pose = torch.sum(global_smplx_pose * global_pose_weight, 0)
        
        
        if model_status == 'train':
            smplx_index = [0, 1, 3, 4, 6, 7, 9,  10, 11, 14, 15, 16, 17, 18, 19, 20]
            joint_index = [1, 2, 3, 4, 5, 6, 14, 17, 7,  24, 8,  9,  10, 11, 12, 13]
            # 按照joint_index提取对应的关节点深度
            rank_labe = global_body_depth_joint[:, :, joint_index]
            # 减去最小值normalize
            rank_labe = (rank_labe - torch.min(rank_labe)) # v*b*x
            
            rank_score = torch.mean(global_pose_weight.reshape(view_num, batch_size, -1, 3), dim=3) # v*b*x
            rank_score = rank_score[:, :, smplx_index] # v*b*x
            
            for view_idx in range(rank_labe.shape[0]):
                temp_rank_label = rank_labe[view_idx]
                temp_rank_score = nn.Softmax(dim=1)(rank_score[view_idx])
                if view_idx == 0:
                    all_jrn_loss = self.loss_fuc_rank(temp_rank_label, temp_rank_score)
                else:
                    all_jrn_loss += self.loss_fuc_rank(temp_rank_label, temp_rank_score)
            
            
            out["jrn_loss"] = all_jrn_loss / rank_labe.shape[0]
            
        
        if cfg.srn_loss:
            global_shape_uv_map = None
            global_expr_uv_map = None
            
            global_smplx_pose_repeat = global_smplx_pose.unsqueeze(0).expand(view_num, batch_size, -1)
            global_smplx_pose_repeat = global_smplx_pose_repeat.reshape(view_num * batch_size, -1)
            global_smplx_shape = global_shape_weight.reshape(view_num * batch_size, -1)
            global_smplx_expr = global_expr_weight.reshape(view_num * batch_size, -1)
            global_shape_condition = global_shape_condition.reshape(view_num * batch_size, -1, 10)
            global_expr_condition = global_expr_condition.reshape(view_num * batch_size, -1, 10)
            
            body_para_dict = self.get_smplx_dic(global_smplx_pose_repeat)
            body_para_dict['betas'] = global_smplx_shape
            body_para_dict['expression'] = global_smplx_expr
            
            pred_vertex = self.body_model(return_verts=True, 
                                          return_full_pose=True, 
                                          **body_para_dict).vertices.detach().cpu().numpy()
            for i in range(view_num * batch_size):
                self.uv_mesh.vertices = pred_vertex[i]
                pred_mesh_vn = self.uv_mesh.vertex_normals
                shape_uv_map = np.zeros((3, cfg.input_body_uv_shape[0], cfg.input_body_uv_shape[1]))
                for vert_idx, vert_point in enumerate(self.uv_mesh_vt):
                    pred_vn = pred_mesh_vn[vert_idx]
                    shape_uv_map[:, vert_point[1], vert_point[0]] = pred_vn
                
                expr_uv_map = copy.deepcopy(shape_uv_map[:, 
                                            self.uv_head_bbox[1]:self.uv_head_bbox[3],
                                            self.uv_head_bbox[0]:self.uv_head_bbox[2]])
                
                shape_uv_map = nn.functional.interpolate(torch.from_numpy(shape_uv_map).unsqueeze(0).cuda(), 
                                                         size=cfg.output_body_uv_shape, 
                                                         mode='bilinear', align_corners=False).float()
                expr_uv_map = nn.functional.interpolate(torch.from_numpy(expr_uv_map).unsqueeze(0).cuda(),
                                                        size=cfg.output_face_uv_shape,
                                                        mode='bilinear', align_corners=False).float()
                
                if i == 0:
                    global_shape_uv_map = shape_uv_map
                    global_expr_uv_map = expr_uv_map
                else:
                    global_shape_uv_map = torch.cat((global_shape_uv_map, shape_uv_map), 0)
                    global_expr_uv_map = torch.cat((global_expr_uv_map, expr_uv_map), 0)
            
            global_shape_uv_weight = self.shape_unet((global_shape_uv_map, global_shape_condition))
            global_expr_uv_weight = self.expr_unet((global_expr_uv_map, global_expr_condition))
            
            global_shape_weight = self.shape_out(global_shape_uv_weight)
            global_expr_weight = self.expr_out(global_expr_uv_weight)
            
            global_shape_uv_map = global_shape_uv_map.reshape(view_num, batch_size, 3, 
                                                              cfg.output_body_uv_shape[0], cfg.output_body_uv_shape[1])
            global_expr_uv_map = global_expr_uv_map.reshape(view_num, batch_size, 3, 
                                                            cfg.output_face_uv_shape[0], cfg.output_face_uv_shape[1])
            global_shape_uv_weight = nn.Softmax(dim=0)(global_shape_uv_weight.reshape(view_num, batch_size, 3, 
                                                                    cfg.output_body_uv_shape[0], 
                                                                    cfg.output_body_uv_shape[1]))
            global_expr_uv_weight = nn.Softmax(dim=0)(global_expr_uv_weight.reshape(view_num, batch_size, 3,
                                                                  cfg.output_face_uv_shape[0], 
                                                                  cfg.output_face_uv_shape[1]))
            global_shape_weight = nn.Softmax(dim=0)(global_shape_weight.reshape(view_num, batch_size, 10))
            global_expr_weight = nn.Softmax(dim=0)(global_expr_weight.reshape(view_num, batch_size, 10))
            
            global_smplx_shape = global_smplx_shape.reshape(view_num, batch_size, 10)
            global_smplx_expr = global_smplx_expr.reshape(view_num, batch_size, 10)
            
            global_shape_uv_map = torch.sum(global_shape_uv_map * global_shape_uv_weight, 0)
            global_expr_uv_map = torch.sum(global_expr_uv_map * global_expr_uv_weight, 0)
            
            global_smplx_shape = torch.sum(global_shape_weight * global_smplx_shape, 0)
            global_smplx_expr = torch.sum(global_expr_weight * global_smplx_expr, 0)
        else:
            global_shape_weight = nn.Softmax(dim=0)(global_shape_weight.reshape(view_num, batch_size, 10))
            global_expr_weight = nn.Softmax(dim=0)(global_expr_weight.reshape(view_num, batch_size, 10))
            
            global_smplx_shape = torch.sum(global_shape_weight * global_smplx_shape, 0)
            global_smplx_expr = torch.sum(global_expr_weight * global_smplx_expr, 0)
            
        out["smplx_pose"] = global_smplx_pose
        out["smplx_root"] = global_smplx_root
        out["smplx_shape"] = global_smplx_shape
        out["smplx_expr"] = global_smplx_expr
        
        if model_status == 'train':
            for key in out.keys():
                if "loss" in key:
                    if key == "jrn_loss":
                        continue
                    # print(f"{key}")
                    out[key] = out[key] / view_num
            
            out["loss_smplx_pose"] = abs(global_smplx_pose - multi_view_targets["0"]["smplx_pose"][:, 3:]) * getattr(cfg, 'smplx_pose_weight', 1.0)
            out["loss_smplx_shape"] = abs(global_smplx_shape - multi_view_targets["0"]["smplx_shape"])
            out["loss_smplx_expr"] = abs(global_smplx_expr - multi_view_targets["0"]["smplx_expr"])
            
            smplx_kps_3d_weight = getattr(cfg, 'smplx_kps_3d_weight', 1.0)
            smplx_kps_3d_weight = getattr(cfg, 'smplx_kps_weight', smplx_kps_3d_weight) # old config
            
            smplx_kps_2d_weight = getattr(cfg, 'smplx_kps_2d_weight', 1.0)
            net_kps_2d_weight = getattr(cfg, 'net_kps_2d_weight', 1.0)
            
            view_count = 0
            for key in out.keys():
                if "view_" in key:
                    view = key.split("_")[1]
                    joint_proj, joint_cam, joint_cam_wo_ra, mesh_cam = self.get_coord(
                        out[key]["smplx_root_pose"], 
                        global_smplx_pose[:, :63], 
                        global_smplx_pose[:, 63:63+45], 
                        global_smplx_pose[:, 63+45:63+45+45], 
                        global_smplx_pose[:, 63+45+45:63+45+45+3],
                        global_smplx_shape, 
                        global_smplx_expr, 
                        out[key]["cam_trans"], 
                        "train")
                    
                    if view_count == 0:
                        fuse_loss_joint_cam = self.coord_loss(joint_cam_wo_ra, multi_view_targets[view]["joint_cam"], multi_view_meta_info[view]["joint_valid"] * multi_view_meta_info[view]["is_3D"][:, None, None]) * smplx_kps_3d_weight
                        
                        fuse_loss_smplx_joint_cam = self.coord_loss(joint_cam, multi_view_targets[view]['smplx_joint_cam'], multi_view_meta_info[view]['smplx_joint_valid']) * smplx_kps_3d_weight
                        
                        fuse_loss_joint_proj = self.coord_loss(joint_proj, multi_view_targets[view]['joint_img'][:, :, :2], multi_view_meta_info[view]['joint_trunc']) * smplx_kps_2d_weight
                    else:
                        fuse_loss_joint_cam += self.coord_loss(joint_cam_wo_ra, multi_view_targets[view]["joint_cam"], multi_view_meta_info[view]["joint_valid"] * multi_view_meta_info[view]["is_3D"][:, None, None]) * smplx_kps_3d_weight
                        
                        fuse_loss_smplx_joint_cam += self.coord_loss(joint_cam, multi_view_targets[view]['smplx_joint_cam'], multi_view_meta_info[view]['smplx_joint_valid']) * smplx_kps_3d_weight
                        
                        fuse_loss_joint_proj += self.coord_loss(joint_proj, multi_view_targets[view]['joint_img'][:, :, :2], multi_view_meta_info[view]['joint_trunc']) * smplx_kps_2d_weight
                    
                    view_count += 1
            out["fuse_loss_joint_cam"] = fuse_loss_joint_cam
            out["fuse_loss_smplx_joint_cam"] = fuse_loss_smplx_joint_cam
            out["fuse_loss_joint_proj"] = fuse_loss_joint_proj
        
        return out
        
        
        
            
            