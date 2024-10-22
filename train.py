import os
import torch
import random
import argparse
import logging
import copy
import numpy as np
import torch.distributed
import torch.utils
import torchvision.transforms as transforms
from tqdm import tqdm
from dataset.Human36M.Human36M import Human36M
from dataset.RICH.RICH import RICH
from common.pose_nets.config import cfg
from common.utils.utils import init
from common.utils.human_models import smpl_x
from common.pose_nets.fuse import Fusion_Net
from common.pose_nets.smpler_x import get_model
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data.distributed import DistributedSampler
from common.utils.transforms import rigid_align


def get_smplx_dic(smplx_params):
    batch_size = smplx_params.shape[0]
    shape = torch.zeros((batch_size, 10)).float().cuda()
    zero_pose = torch.zeros((batch_size, 3)).float().cuda()
    expr = torch.zeros((batch_size, 10)).float().cuda()
    
    body_pose = smplx_params[:, :63].cuda()
    if cfg.dataset == 'human36m':
        lhand = torch.zeros((batch_size, 45)).float().cuda()
        rhand = torch.zeros((batch_size, 45)).float().cuda()
        jaw = torch.zeros((batch_size, 3)).float().cuda()
    else:
        lhand = smplx_params[:, 63:63+45].cuda()
        rhand = smplx_params[:, 63+45:63+45+45].cuda()
        jaw = smplx_params[:, 63+45+45:63+45+45+3].cuda()
    
    dict = {'betas': shape, 'body_pose': body_pose, 'global_orient': zero_pose,
            'right_hand_pose': rhand.cuda(), 'left_hand_pose': lhand.cuda(),
            'jaw_pose': jaw.cuda(), 'leye_pose': zero_pose, 'reye_pose': zero_pose,
            'expression': expr}
    
    return dict


def get_eval(gt_dict, pred_dict):
    gt_mesh = smplx_layer(**gt_dict).vertices.detach().cpu().numpy()[0]
    pred_mesh = smplx_layer(**pred_dict).vertices.detach().cpu().numpy()[0]

    if len(gt_mesh) != len(pred_mesh):
        raise ValueError("Meshes must have the same number of vertices")
    
    pred_mesh_align = rigid_align(pred_mesh, gt_mesh)
    mpvpe = np.mean(np.sqrt(np.sum((pred_mesh_align - gt_mesh) ** 2, 1)) * 1000)
    
    if cfg.dataset == 'human36m':
        human36m_J17_regressor = np.load(cfg.human36m_J17_regressor)
        eval_joints_list = (1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16)
        joint_gt_body = np.dot(human36m_J17_regressor, gt_mesh)[eval_joints_list, :]
        joint_out_body = np.dot(human36m_J17_regressor, pred_mesh)[eval_joints_list, :]
    elif cfg.dataset == 'rich':
        joint_gt_body = np.dot(smpl_x.j14_regressor, gt_mesh)
        joint_out_body = np.dot(smpl_x.j14_regressor, pred_mesh)
    
    joint_out_body_align = rigid_align(joint_out_body, joint_gt_body)
    mpjpe = np.mean(np.sqrt(np.sum((joint_out_body_align - joint_gt_body) ** 2, 1)) * 1000)

    return mpvpe, mpjpe


#---------------------------------------------------#
#   set seed
#---------------------------------------------------#
def seed_everything(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
#---------------------------------------------------#
#   set param
#---------------------------------------------------#
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids', default='0')
    parser.add_argument('--lr', type=str, dest='lr', default='3e-5')
    parser.add_argument('--dataset', type=str, choices=['human36m', 'rich'])
    parser.add_argument('--num_thread', type=int, default=8)
    parser.add_argument('--end_epoch', type=int, default=50)
    parser.add_argument('--froze', action=argparse.BooleanOptionalAction)
    parser.add_argument('--jrn_loss', action=argparse.BooleanOptionalAction)
    parser.add_argument('--full_test', action=argparse.BooleanOptionalAction)
    parser.add_argument('--srn_loss', action=argparse.BooleanOptionalAction)
    parser.add_argument('--encoder_setting', type=str, default='large', choices=['base', 'large', 'huge'])
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set proper gpu ids"

    if not args.lr:
        assert 0, "Please set learning rate"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args



def move_dict_to_device(data):
    for key in data.keys():
        if isinstance(data[key], dict):
            data[key] = move_dict_to_device(data[key])
        else:
            data[key] = data[key].cuda()
    return data


def train(net, train_set):
    net.train()
    running_loss = 0.0
    with tqdm(total=len(train_set), desc=f'Epoch {epoch}/{cfg.end_epoch}', unit='batch') as pbar:
        for data in train_set:
            pbar.set_description('Epoch %i' % epoch)
            
            multi_view_inputs, multi_view_targets, multi_view_meta_info = data
            multi_view_inputs = move_dict_to_device(multi_view_inputs)
            multi_view_targets = move_dict_to_device(multi_view_targets)
            multi_view_meta_info = move_dict_to_device(multi_view_meta_info)
            
            view_num = len(multi_view_inputs)
            
            optimizer.zero_grad()
            out = net(multi_view_inputs, multi_view_targets, multi_view_meta_info, 'train')
            
            loss_smplx_pose = out['loss_smplx_pose'].mean()
            loss_smplx_shape = out['loss_smplx_shape'].mean()
            loss_smplx_expr = out['loss_smplx_expr'].mean()
            loss_joint_cam = out['loss_joint_cam'].mean() / view_num
            loss_smplx_joint_cam = out['loss_smplx_joint_cam'].mean() / view_num
            loss_joint_proj = out['loss_joint_proj'].mean() / view_num
            loss_joint_img = out['loss_joint_img'].mean() / view_num
            loss_smplx_joint_img = out['loss_smplx_joint_img'].mean() / view_num
            
            fuse_loss_joint_cam = out['fuse_loss_joint_cam'].mean() / view_num
            fuse_loss_smplx_joint_cam = out['fuse_loss_smplx_joint_cam'].mean() / view_num
            fuse_loss_joint_proj = out['fuse_loss_joint_proj'].mean() / view_num
            
            jrn_loss = out['jrn_loss'].mean()
            
            loss = loss_smplx_pose + loss_smplx_shape + loss_smplx_expr + fuse_loss_joint_cam + fuse_loss_smplx_joint_cam + fuse_loss_joint_proj + loss_joint_img + loss_smplx_joint_img + loss_joint_cam + loss_smplx_joint_cam + loss_joint_proj
            
            # rich dataset
            if cfg.dataset == 'rich':
                if "loss_lhand_bbox" in out.keys():
                    loss_lhand_bbox = out["loss_lhand_bbox"].mean()
                    loss += loss_lhand_bbox
                if "loss_rhand_bbox" in out.keys():
                    loss_rhand_bbox = out["loss_rhand_bbox"].mean()
                    loss += loss_rhand_bbox
                if "loss_face_bbox" in out.keys():
                    loss_face_bbox = out["loss_face_bbox"].mean()
                    loss += loss_face_bbox
            
            # check negative loss
            for key in out.keys():
                if "loss" in key:
                    if out[key].mean() < 0:
                        logging.info(f"Epoch {epoch} negative loss: {key}")
                        logging.info(f"Epoch {epoch} negative loss: {out[key].mean()}")
            
            if cfg.jrn_loss:
                loss += jrn_loss
                
            loss.backward()
            
            running_loss += loss.item()
            optimizer.step()
            
            pbar.update(1)
    scheduler.step()
    logging.info(f"Epoch {epoch} loss: {running_loss / len(train_set)} lr: {scheduler.get_last_lr()[0]}")
    
    # clean cuda cache
    # torch.cuda.empty_cache()
    

@torch.no_grad()
def eval(net, test_set):
    mpvpe, mpjpe = 0, 0
    net.eval()
    with tqdm(total=len(test_set), desc=f'Epoch {epoch}/{cfg.end_epoch}', unit='batch') as pbar:
        for data in test_set:
            pbar.set_description('Epoch %i' % epoch)
            
            multi_view_inputs, multi_view_targets, multi_view_meta_info = data
            multi_view_inputs = move_dict_to_device(multi_view_inputs)
            multi_view_targets = move_dict_to_device(multi_view_targets)
            multi_view_meta_info = move_dict_to_device(multi_view_meta_info)
            
            out = net(multi_view_inputs, multi_view_targets, multi_view_meta_info, 'test')
            
            gt_smplx_param = multi_view_targets["0"]["smplx_pose"][:, 3:]
            gt_smplx_dict = get_smplx_dic(gt_smplx_param)
            gt_smplx_dict["betas"] = multi_view_targets["0"]["smplx_shape"]
            gt_smplx_dict["expression"] = multi_view_targets["0"]["smplx_expr"]
            
            pr_smplx_param = out["smplx_pose"]
            pr_smplx_dict = get_smplx_dic(pr_smplx_param)
            pr_smplx_dict["betas"] = out["smplx_shape"]
            pr_smplx_dict["expression"] = out["smplx_expr"]
            
            temp_mpvpe, temp_mpjpe = get_eval(gt_smplx_dict, pr_smplx_dict)
            mpvpe += temp_mpvpe
            mpjpe += temp_mpjpe
            
            pbar.update(1)
    logging.info(f"Epoch {epoch} mpvpe: {mpvpe / len(test_set)}")
    logging.info(f"Epoch {epoch} mpjpe: {mpjpe / len(test_set)}")
    
    return mpvpe / len(test_set), mpjpe / len(test_set)
            
    
if __name__ == "__main__":
    seed_everything()
    
    args = parse_args()
    cfg.set_args(args.gpu_ids, args.lr)
    cfg.set_additional_args(
        num_thread=args.num_thread,
        encoder_setting=args.encoder_setting,
        end_epoch=args.end_epoch,
        jrn_loss=args.jrn_loss,
        full_test=args.full_test,
        srn_loss=args.srn_loss,
        dataset=args.dataset,
        froze=args.froze
    )
    
    ckpt_path = init(cfg)
    logging.info(f"frozen: {cfg.froze}")
    logging.info(f"jrn_loss: {cfg.jrn_loss}")
    logging.info(f"srn_loss: {cfg.srn_loss}")
    logging.info(f"encoder_setting: {cfg.encoder_setting}")
    logging.info(f"dataset: {cfg.dataset}")
    
    encoder = get_model('test')
    # encoder = DataParallel(encoder)
    print('### Loading pretrained model from {}'.format(cfg.pretrained_model_path))
    ckpt = torch.load(cfg.pretrained_model_path)
    
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in ckpt['network'].items():
        k = k.replace('module.backbone', 'module.encoder').replace('body_rotation_net', 'body_regressor').replace(
            'hand_rotation_net', 'hand_regressor')
        k = k.replace('module.', '')
        new_state_dict[k] = v
    encoder.load_state_dict(new_state_dict, strict=False)
    
    net = Fusion_Net(encoder, cfg.dataset)
    # /public/home/zhuyt12022/muc/results/human36m_20240720_181052.pt
    # net.load_state_dict(torch.load('/public/home/zhuyt12022/muc/results/human36m_20240720_181052.pt'))
    net = net.cuda()
    
    
    if cfg.dataset == 'human36m':
        train_set = Human36M(transforms.ToTensor(), 'train')
        test_set = Human36M(transforms.ToTensor(), 'test')
        
        train_loader = torch.utils.data.DataLoader(train_set, 
                                                   batch_size=cfg.train_batch_size, 
                                                   num_workers=args.num_thread,
                                                   shuffle=True,
                                                   drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, 
                                                  batch_size=8, 
                                                  num_workers=args.num_thread, 
                                                  drop_last=False)
    elif cfg.dataset == 'rich':
        train_set = RICH(transforms.ToTensor(), 'train')
        val_set = RICH(transforms.ToTensor(), 'val')
        test_set = RICH(transforms.ToTensor(), 'test')
        
        train_loader = torch.utils.data.DataLoader(train_set, 
                                                   batch_size=cfg.train_batch_size, 
                                                   num_workers=cfg.num_thread,
                                                   shuffle=True,
                                                   drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_set,
                                                 batch_size=1,
                                                 num_workers=args.num_thread,
                                                 drop_last=False)
        test_loader = torch.utils.data.DataLoader(test_set, 
                                                  batch_size=1, 
                                                  num_workers=args.num_thread, 
                                                  drop_last=False)
        
        
    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.end_epoch, 1e-8)
    
    smplx_layer = copy.deepcopy(smpl_x.layer['neutral']).cuda()
    
    best_mpvpe = 1e3
    for epoch in range(1, args.end_epoch + 1):
        train(net, train_loader)
        if epoch % 5 == 0:
            torch.save(net.state_dict(), ckpt_path.replace('.pt', f'_{epoch}.pt'))
        if cfg.dataset == 'human36m':
            mpvpe, mpjpe = eval(net, test_loader)
            if mpvpe < best_mpvpe:
                best_mpvpe = mpvpe
                torch.save(net.state_dict(), ckpt_path)
                logging.info(f"Best model saved in {ckpt_path}")
        elif cfg.dataset == 'rich':
            mpvpe, mpjpe = eval(net, val_loader)
            if mpvpe < best_mpvpe:
                best_mpvpe = mpvpe
                torch.save(net.state_dict(), ckpt_path)
                logging.info(f"Best model saved in {ckpt_path}")
                mpvpe, mpjpe = eval(net, test_loader)
    