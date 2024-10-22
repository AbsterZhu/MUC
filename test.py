import os
import torch
import random
import argparse
import logging
import copy
import numpy as np
import trimesh
import torch.distributed
import torch.utils
import torchvision.transforms as transforms
from tqdm import tqdm
from dataset.Human36M.Human36M import Human36M
from dataset.RICH.RICH import RICH
from common.utils.human_models import smpl_x
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
    if smplx_params.shape[1] <= 63:
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


def get_eval(gt_dict, pred_dict, eval_result):
    mesh_gt = smplx_layer(**gt_dict).vertices.detach().cpu().numpy()[0]
    mesh_out = smplx_layer(**pred_dict).vertices.detach().cpu().numpy()[0]
    # import trimesh
    # print(pred_dict['jaw_pose'])
    # mesh = trimesh.Trimesh(vertices=pred_mesh, faces=smplx_layer.faces)
    # mesh.export('pr.obj')
    # exit()

    if len(mesh_gt) != len(mesh_out):
        raise ValueError("Meshes must have the same number of vertices")
    
    # MPVPE from all vertices
    pred_mesh_align = rigid_align(mesh_out, mesh_gt)
    eval_result['pa_mpvpe_all'].append(np.sqrt(np.sum((pred_mesh_align - mesh_gt) ** 2, 1)) * 1000)
    pred_mesh_align = mesh_out - np.dot(smpl_x.J_regressor, mesh_out)[smpl_x.J_regressor_idx['pelvis'], None, :] + \
                             np.dot(smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['pelvis'], None, :]
    eval_result['mpvpe_all'].append(np.sqrt(np.sum((pred_mesh_align - mesh_gt) ** 2, 1)) * 1000)
    
    # MPVPE from hand vertices
    mesh_gt_lhand = mesh_gt[smpl_x.hand_vertex_idx['left_hand'], :]
    mesh_out_lhand = mesh_out[smpl_x.hand_vertex_idx['left_hand'], :]
    mesh_out_lhand_align = rigid_align(mesh_out_lhand, mesh_gt_lhand)
    mesh_gt_rhand = mesh_gt[smpl_x.hand_vertex_idx['right_hand'], :]
    mesh_out_rhand = mesh_out[smpl_x.hand_vertex_idx['right_hand'], :]
    mesh_out_rhand_align = rigid_align(mesh_out_rhand, mesh_gt_rhand)
    pa_mpvpe_hand = []
    pa_mpvpe_lhand = np.sqrt(np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1)) * 1000
    pa_mpvpe_hand.append(pa_mpvpe_lhand)
    pa_mpvpe_rhand = np.sqrt(np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1)) * 1000
    pa_mpvpe_hand.append(pa_mpvpe_rhand)
    eval_result['pa_mpvpe_hand'].append(np.mean(pa_mpvpe_hand))
    
    mesh_out_lhand_align = mesh_out_lhand - np.dot(smpl_x.J_regressor, mesh_out)[
                                            smpl_x.J_regressor_idx['lwrist'], None, :] + np.dot(
                                            smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['lwrist'], None, :]
    mesh_out_rhand_align = mesh_out_rhand - np.dot(smpl_x.J_regressor, mesh_out)[
                                            smpl_x.J_regressor_idx['rwrist'], None, :] + np.dot(
                                            smpl_x.J_regressor, mesh_gt)[smpl_x.J_regressor_idx['rwrist'], None, :]
                                            
    mpvpe_hand = []
    mpvpe_lhand = np.sqrt(np.sum((mesh_out_lhand_align - mesh_gt_lhand) ** 2, 1)) * 1000
    mpvpe_hand.append(mpvpe_lhand)
    mpvpe_rhand = np.sqrt(np.sum((mesh_out_rhand_align - mesh_gt_rhand) ** 2, 1)) * 1000
    mpvpe_hand.append(mpvpe_rhand)
    eval_result['mpvpe_hand'].append(np.mean(mpvpe_hand))
    
    # MPVPE from face vertices
    mesh_gt_face = mesh_gt[smpl_x.face_vertex_idx, :]
    mesh_out_face = mesh_out[smpl_x.face_vertex_idx, :]
    mesh_out_face_align = rigid_align(mesh_out_face, mesh_gt_face)
    eval_result['pa_mpvpe_face'].append(
            np.sqrt(np.sum((mesh_out_face_align - mesh_gt_face) ** 2, 1)) * 1000)
    mesh_out_face_align = mesh_out_face - np.dot(smpl_x.J_regressor, mesh_out)[smpl_x.J_regressor_idx['neck'],
                                            None, :] + np.dot(smpl_x.J_regressor, mesh_gt)[
                                                        smpl_x.J_regressor_idx['neck'], None, :]
    eval_result['mpvpe_face'].append(
            np.sqrt(np.sum((mesh_out_face_align - mesh_gt_face) ** 2, 1)) * 1000)
    
    # MPJPE from body joints
    if cfg.dataset == 'human36m':
        human36m_J17_regressor = np.load(cfg.human36m_J17_regressor)
        eval_joints_list = (1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16)
        joint_gt_body = np.dot(human36m_J17_regressor, mesh_gt)[eval_joints_list, :]
        joint_out_body = np.dot(human36m_J17_regressor, mesh_out)[eval_joints_list, :]
    else:
        joint_gt_body = np.dot(smpl_x.j14_regressor, mesh_gt)
        joint_out_body = np.dot(smpl_x.j14_regressor, mesh_out)
    joint_out_body_align = rigid_align(joint_out_body, joint_gt_body)
    eval_result['pa_mpjpe_body'].append(
        np.sqrt(np.sum((joint_out_body_align - joint_gt_body) ** 2, 1)) * 1000)
    
    # MPJPE from hand joints
    joint_gt_lhand = np.dot(smpl_x.orig_hand_regressor['left'], mesh_gt)
    joint_out_lhand = np.dot(smpl_x.orig_hand_regressor['left'], mesh_out)
    joint_out_lhand_align = rigid_align(joint_out_lhand, joint_gt_lhand)
    joint_gt_rhand = np.dot(smpl_x.orig_hand_regressor['right'], mesh_gt)
    joint_out_rhand = np.dot(smpl_x.orig_hand_regressor['right'], mesh_out)
    joint_out_rhand_align = rigid_align(joint_out_rhand, joint_gt_rhand)

    pa_mpjpe_hand = []
    pa_mpjpe_lhand = np.sqrt(np.sum((joint_out_lhand_align - joint_gt_lhand) ** 2, 1)) * 1000
    pa_mpjpe_hand.append(pa_mpjpe_lhand)
    pa_mpjpe_rhand = np.sqrt(np.sum((joint_out_rhand_align - joint_gt_rhand) ** 2, 1)) * 1000
    pa_mpjpe_hand.append(pa_mpjpe_rhand)
    eval_result['pa_mpjpe_hand'].append(np.mean(pa_mpjpe_hand))

    return eval_result, mesh_gt, mesh_out


#---------------------------------------------------#
#   设置种子
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
#   设置参数
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
            if key == "img_path" or key == "bbox":
                continue
            data[key] = data[key].cuda()
    return data


@torch.no_grad()
def eval(net, test_set):
    results = {
        "fusion_results": {'pa_mpvpe_all': [], 'pa_mpvpe_hand': [], 'pa_mpvpe_face': [], 
                           'mpvpe_all': [], 'mpvpe_hand': [], 'mpvpe_face': [], 
                           'pa_mpjpe_body': [], 'pa_mpjpe_hand': []},
        "single_results": {'pa_mpvpe_all': [], 'pa_mpvpe_hand': [], 'pa_mpvpe_face': [], 
                           'mpvpe_all': [], 'mpvpe_hand': [], 'mpvpe_face': [], 
                           'pa_mpjpe_body': [], 'pa_mpjpe_hand': []},
        "mean_results": {'pa_mpvpe_all': [], 'pa_mpvpe_hand': [], 'pa_mpvpe_face': [], 
                         'mpvpe_all': [], 'mpvpe_hand': [], 'mpvpe_face': [], 
                         'pa_mpjpe_body': [], 'pa_mpjpe_hand': []}
    }
    
    mesh_save_path = "/public_bme/data/zhuyt1128/smpl_mesh/"
    
    net.eval()
    with tqdm(total=len(test_set)) as pbar:
        test_count = 0
        for data in test_set:
            
            multi_view_inputs, multi_view_targets, multi_view_meta_info = data
            multi_view_inputs = move_dict_to_device(multi_view_inputs)
            multi_view_targets = move_dict_to_device(multi_view_targets)
            multi_view_meta_info = move_dict_to_device(multi_view_meta_info)
            
            out = net(multi_view_inputs, multi_view_targets, multi_view_meta_info, 'test')
            
            gt_smplx_param = multi_view_targets["0"]["smplx_pose"][:, 3:]
            gt_smplx_dict = get_smplx_dic(gt_smplx_param)
            gt_smplx_dict["betas"] = multi_view_targets["0"]["smplx_shape"]
            gt_smplx_dict["expression"] = multi_view_targets["0"]["smplx_expr"]
            
            # evaluate fusion results
            pr_smplx_param = out["smplx_pose"]
            pr_smplx_dict = get_smplx_dic(pr_smplx_param)
            pr_smplx_dict["betas"] = out["smplx_shape"]
            pr_smplx_dict["expression"] = out["smplx_expr"]
            
            results['fusion_results'], mesh_gt, mesh_fusion = get_eval(gt_smplx_dict, pr_smplx_dict, results['fusion_results'])
            
            # save mesh as obj
            mesh_gt_save = trimesh.Trimesh(vertices=mesh_gt, faces=smplx_layer.faces)
            mesh_gt_save.export(mesh_save_path + 'gt/' + f'{test_count}.obj')
            
            mesh_fusion_save = trimesh.Trimesh(vertices=mesh_fusion, faces=smplx_layer.faces)
            mesh_fusion_save.export(mesh_save_path + 'fusion/' + f'{test_count}.obj')
            
            
            # evaluate single view results
            single_smplx_dicts = []
            for key in out.keys():
                view_count = 0
                if "view_" in key:
                    # print(out[key].keys())
                    
                    if cfg.dataset == 'human36m':
                        single_smplx_param = out[key]["smplx_body_pose"]
                    else:
                        # print(out[key]["smplx_body_pose"].shape)
                        # exit()
                        single_smplx_param = torch.cat(
                            (out[key]["smplx_body_pose"],
                            out[key]["smplx_lhand_pose"],
                            out[key]["smplx_rhand_pose"],
                            out[key]["smplx_jaw_pose"]), dim=1)
                    single_smplx_dict = get_smplx_dic(single_smplx_param)
                    single_smplx_dict["betas"] = out[key]["smplx_shape"]
                    single_smplx_dict["expression"] = out[key]["smplx_expr"]
                    
                    single_smplx_dicts.append(single_smplx_dict)
                    
                    results['single_results'], mesh_gt, mesh_single = get_eval(gt_smplx_dict, single_smplx_dict, results['single_results'])
                    
                    mesh_single_save = trimesh.Trimesh(vertices=mesh_single, faces=smplx_layer.faces)
                    mesh_single_save.export(mesh_save_path + f'single/{test_count}_{view_count}.obj')
                    
                    view_count += 1
                        
            # evaluate mean results
            mean_smplx_dict = {}
            for view in single_smplx_dicts:
                for key in view.keys():
                    if key not in mean_smplx_dict:
                        mean_smplx_dict[key] = view[key]
                    else:
                        mean_smplx_dict[key] += view[key]
            for key in mean_smplx_dict.keys():
                mean_smplx_dict[key] /= len(single_smplx_dicts)
            
            results['mean_results'], mesh_gt, mesh_mean = get_eval(gt_smplx_dict, mean_smplx_dict, results['mean_results'])
            
            
            pbar.update(1)
            
            test_count += 1
            if test_count == 1:
                break
    
    return results
            
    
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
    # best h36m base vit
    net.load_state_dict(torch.load('/public/home/zhuyt12022/muc/results/rich_20240812_142950.pt'))
    net = net.cuda()
    
    
    if cfg.dataset == 'human36m':
        train_set = Human36M(transforms.ToTensor(), 'train')
        test_set = Human36M(transforms.ToTensor(), 'test')

        test_loader = torch.utils.data.DataLoader(test_set, 
                                                  batch_size=8, 
                                                  num_workers=args.num_thread, 
                                                  drop_last=False)
    else:
        train_set = RICH(transforms.ToTensor(), 'train')
        test_set = RICH(transforms.ToTensor(), 'test')

        test_loader = torch.utils.data.DataLoader(test_set, 
                                                  batch_size=1, 
                                                  num_workers=args.num_thread, 
                                                  drop_last=False)
    
    smplx_layer = copy.deepcopy(smpl_x.layer['neutral']).cuda()
    
    results = eval(net, test_loader)
    
    # print results
    for key in results.keys():
        logging.info(f"Results for {key}")
        for metric in results[key].keys():
            logging.info(f"{metric}: {np.mean(results[key][metric])}")
        logging.info("\n")
            
    