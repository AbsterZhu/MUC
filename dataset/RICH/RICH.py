import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
import time
import tqdm
import random
from pycocotools.coco import COCO
from dataset.humandata import Cache
from common.pose_nets.config import cfg
from common.utils.human_models import smpl_x
from common.utils.preprocessing import load_img, process_bbox, augmentation, process_db_coord, process_human_model_output, \
    get_fitting_error_3D
from common.utils.transforms import world2cam, cam2pixel, rigid_align
from humandata import HumanDataset


torch.multiprocessing.set_sharing_strategy('file_system')

KPS2D_KEYS = ['keypoints2d', 'keypoints2d_smplx', 'keypoints2d_smpl', 'keypoints2d_original']
KPS3D_KEYS = ['keypoints3d_cam', 'keypoints3d', 'keypoints3d_smplx','keypoints3d_smpl' ,'keypoints3d_original'] 
# keypoints3d_cam with root-align has higher priority, followed by old version key keypoints3d
# when there is keypoints3d_smplx, use this rather than keypoints3d_original

hands_meanr = np.array([ 0.11167871, -0.04289218,  0.41644183,  0.10881133,  0.06598568,
        0.75622   , -0.09639297,  0.09091566,  0.18845929, -0.11809504,
       -0.05094385,  0.5295845 , -0.14369841, -0.0552417 ,  0.7048571 ,
       -0.01918292,  0.09233685,  0.3379135 , -0.45703298,  0.19628395,
        0.6254575 , -0.21465237,  0.06599829,  0.50689423, -0.36972436,
        0.06034463,  0.07949023, -0.1418697 ,  0.08585263,  0.63552827,
       -0.3033416 ,  0.05788098,  0.6313892 , -0.17612089,  0.13209307,
        0.37335458,  0.8509643 , -0.27692273,  0.09154807, -0.49983943,
       -0.02655647, -0.05288088,  0.5355592 , -0.04596104,  0.27735803]).reshape(15, -1)
hands_meanl = np.array([ 0.11167871,  0.04289218, -0.41644183,  0.10881133, -0.06598568,
       -0.75622   , -0.09639297, -0.09091566, -0.18845929, -0.11809504,
        0.05094385, -0.5295845 , -0.14369841,  0.0552417 , -0.7048571 ,
       -0.01918292, -0.09233685, -0.3379135 , -0.45703298, -0.19628395,
       -0.6254575 , -0.21465237, -0.06599829, -0.50689423, -0.36972436,
       -0.06034463, -0.07949023, -0.1418697 , -0.08585263, -0.63552827,
       -0.3033416 , -0.05788098, -0.6313892 , -0.17612089, -0.13209307,
       -0.37335458,  0.8509643 ,  0.27692273, -0.09154807, -0.49983943,
        0.02655647,  0.05288088,  0.5355592 ,  0.04596104, -0.27735803]).reshape(15, -1)



class RICH(HumanDataset):
    def __init__(self, transform, data_split):
        super(RICH, self).__init__(transform, data_split)

        self.use_cache = getattr(cfg, 'use_cache', False)
        self.root_path = osp.join(cfg.data_dir, 'RICH', 'preprocessed_datasets')
        if self.data_split == 'train':
            self.annot_path_cache = osp.join(cfg.data_dir, 'cache', 'rich_train.npz')
        elif self.data_split == 'val':
            self.annot_path_cache = osp.join(cfg.data_dir, 'cache', 'rich_val.npz')
        elif self.data_split == 'test':
            self.annot_path_cache = osp.join(cfg.data_dir, 'cache', 'rich_test.npz')
        self.img_shape = None  # (h, w)
        self.cam_param = {}

        if self.use_cache and osp.isfile(self.annot_path_cache):
            print(f'[{self.__class__.__name__}] loading cache from {self.annot_path_cache}')
            self.datalist = self.load_cache(self.annot_path_cache)
        else:
            if self.use_cache:
                print(f'[{self.__class__.__name__}] Cache not found, generating cache...')

            if self.data_split == 'train':
                filename = os.path.join(self.root_path, 'rich_train.npz')
            elif self.data_split == 'val':
                filename = os.path.join(self.root_path, 'rich_val.npz')
            elif self.data_split == 'test':
                filename = os.path.join(self.root_path, 'rich_test.npz')

            self.img_dir = osp.join(self.root_path, '..')
            self.annot_path = filename

            # load data
            self.datalist = self.load_multi_view_data(
                train_sample_interval=20,
                test_sample_interval=15)

            if self.use_cache:
                self.save_cache(self.annot_path_cache, self.datalist)
            
    def load_multi_view_data(self, train_sample_interval=1, test_sample_interval=1):

        content = np.load(self.annot_path, allow_pickle=True)
        num_examples = len(content['image_path'])

        if 'meta' in content:
            meta = content['meta'].item()
            print('meta keys:', meta.keys())
        else:
            meta = None
            print('No meta info provided! Please give height and width manually')

        print(f'Start loading humandata {self.annot_path} into memory...\nDataset includes: {content.files}'); tic = time.time()
        image_path = content['image_path']

        if meta is not None and 'height' in meta:
            height = np.array(meta['height'])
            width = np.array(meta['width'])
            image_shape = np.stack([height, width], axis=-1)
        else:
            image_shape = None

        bbox_xywh = content['bbox_xywh']

        if 'smplx' in content:
            smplx = content['smplx'].item()
            as_smplx = 'smplx'
        elif 'smpl' in content:
            smplx = content['smpl'].item()
            as_smplx = 'smpl'
        elif 'smplh' in content:
            smplx = content['smplh'].item()
            as_smplx = 'smplh'

        # TODO: temp solution, should be more general. But SHAPY is very special
        elif self.__class__.__name__ == 'SHAPY':
            smplx = {}

        else:
            raise KeyError('No SMPL for SMPLX available, please check keys:\n'
                        f'{content.files}')

        print('Smplx param', smplx.keys())

        if 'lhand_bbox_xywh' in content and 'rhand_bbox_xywh' in content:
            lhand_bbox_xywh = content['lhand_bbox_xywh']
            rhand_bbox_xywh = content['rhand_bbox_xywh']
        else:
            lhand_bbox_xywh = np.zeros_like(bbox_xywh)
            rhand_bbox_xywh = np.zeros_like(bbox_xywh)

        if 'face_bbox_xywh' in content:
            face_bbox_xywh = content['face_bbox_xywh']
        else:
            face_bbox_xywh = np.zeros_like(bbox_xywh)

        decompressed = False
        if content['__keypoints_compressed__']:
            decompressed_kps = self.decompress_keypoints(content)
            decompressed = True

        keypoints3d = None
        valid_kps3d = False
        keypoints3d_mask = None
        valid_kps3d_mask = False
        for kps3d_key in KPS3D_KEYS:
            if kps3d_key in content:
                keypoints3d = decompressed_kps[kps3d_key][:, self.SMPLX_137_MAPPING, :3] if decompressed \
                else content[kps3d_key][:, self.SMPLX_137_MAPPING, :3]
                valid_kps3d = True

                if f'{kps3d_key}_mask' in content:
                    keypoints3d_mask = content[f'{kps3d_key}_mask'][self.SMPLX_137_MAPPING]
                    valid_kps3d_mask = True
                elif 'keypoints3d_mask' in content:
                    keypoints3d_mask = content['keypoints3d_mask'][self.SMPLX_137_MAPPING]
                    valid_kps3d_mask = True
                break

        for kps2d_key in KPS2D_KEYS:
            if kps2d_key in content:
                keypoints2d = decompressed_kps[kps2d_key][:, self.SMPLX_137_MAPPING, :2] if decompressed \
                    else content[kps2d_key][:, self.SMPLX_137_MAPPING, :2]

                if f'{kps2d_key}_mask' in content:
                    keypoints2d_mask = content[f'{kps2d_key}_mask'][self.SMPLX_137_MAPPING]
                elif 'keypoints2d_mask' in content:
                    keypoints2d_mask = content['keypoints2d_mask'][self.SMPLX_137_MAPPING]
                break

        mask = keypoints3d_mask if valid_kps3d_mask \
                else keypoints2d_mask

        print('Done. Time: {:.2f}s'.format(time.time() - tic))

        datalist = []
        all_data_dict = {}
        for i in tqdm.tqdm(range(int(num_examples))):
            img_path = osp.join(self.img_dir, image_path[i])
            img_shape = image_shape[i] if image_shape is not None else self.img_shape
            
            _, subject_id, cam_id, frame_id = image_path[i].split('/')
            frame_id = frame_id.split('_')[0]

            bbox = bbox_xywh[i][:4]

            if hasattr(cfg, 'bbox_ratio'):
                bbox_ratio = cfg.bbox_ratio * 0.833 # preprocess body bbox is giving 1.2 box padding
            else:
                bbox_ratio = 1.25
            bbox = process_bbox(bbox, img_width=img_shape[1], img_height=img_shape[0], ratio=bbox_ratio)
            if bbox is None: continue
            
            # subject/frame/cam
            if subject_id not in all_data_dict.keys():
                all_data_dict[str(subject_id)] = {}
            if frame_id not in all_data_dict[str(subject_id)].keys():
                all_data_dict[str(subject_id)][str(frame_id)] = {}
            if cam_id not in all_data_dict[str(subject_id)][str(frame_id)].keys():
                all_data_dict[str(subject_id)][str(frame_id)][str(cam_id)] = {}

            # hand/face bbox
            lhand_bbox = lhand_bbox_xywh[i]
            rhand_bbox = rhand_bbox_xywh[i]
            face_bbox = face_bbox_xywh[i]

            if lhand_bbox[-1] > 0:  # conf > 0
                lhand_bbox = lhand_bbox[:4]
                if hasattr(cfg, 'bbox_ratio'):
                    lhand_bbox = process_bbox(lhand_bbox, img_width=img_shape[1], img_height=img_shape[0], ratio=cfg.bbox_ratio)
                if lhand_bbox is not None:
                    lhand_bbox[2:] += lhand_bbox[:2]  # xywh -> xyxy
            else:
                lhand_bbox = None
            if rhand_bbox[-1] > 0:
                rhand_bbox = rhand_bbox[:4]
                if hasattr(cfg, 'bbox_ratio'):
                    rhand_bbox = process_bbox(rhand_bbox, img_width=img_shape[1], img_height=img_shape[0], ratio=cfg.bbox_ratio)
                if rhand_bbox is not None:
                    rhand_bbox[2:] += rhand_bbox[:2]  # xywh -> xyxy
            else:
                rhand_bbox = None
            if face_bbox[-1] > 0:
                face_bbox = face_bbox[:4]
                if hasattr(cfg, 'bbox_ratio'):
                    face_bbox = process_bbox(face_bbox, img_width=img_shape[1], img_height=img_shape[0], ratio=cfg.bbox_ratio)
                if face_bbox is not None:
                    face_bbox[2:] += face_bbox[:2]  # xywh -> xyxy
            else:
                face_bbox = None

            joint_img = keypoints2d[i]
            joint_valid = mask.reshape(-1, 1)
            # num_joints = joint_cam.shape[0]
            # joint_valid = np.ones((num_joints, 1))
            if valid_kps3d:
                joint_cam = keypoints3d[i]
            else:
                joint_cam = None

            smplx_param = {k: v[i] for k, v in smplx.items()}

            smplx_param['root_pose'] = smplx_param.pop('global_orient', None)
            smplx_param['shape'] = smplx_param.pop('betas', None)
            smplx_param['trans'] = smplx_param.pop('transl', np.zeros(3))
            smplx_param['lhand_pose'] = smplx_param.pop('left_hand_pose', None)
            smplx_param['rhand_pose'] = smplx_param.pop('right_hand_pose', None)
            smplx_param['expr'] = smplx_param.pop('expression', None)

            # TODO do not fix betas, give up shape supervision
            if 'betas_neutral' in smplx_param:
                smplx_param['shape'] = smplx_param.pop('betas_neutral')

            # TODO fix shape of poses
            if self.__class__.__name__ == 'Talkshow':
                smplx_param['body_pose'] = smplx_param['body_pose'].reshape(21, 3)
                smplx_param['lhand_pose'] = smplx_param['lhand_pose'].reshape(15, 3)
                smplx_param['rhand_pose'] = smplx_param['lhand_pose'].reshape(15, 3)
                smplx_param['expr'] = smplx_param['expr'][:10]

            if self.__class__.__name__ == 'BEDLAM':
                smplx_param['shape'] = smplx_param['shape'][:10]
                # manually set flat_hand_mean = True
                smplx_param['lhand_pose'] -= hands_meanl
                smplx_param['rhand_pose'] -= hands_meanr


            if as_smplx == 'smpl':
                smplx_param['shape'] = np.zeros(10, dtype=np.float32) # drop smpl betas for smplx
                smplx_param['body_pose'] = smplx_param['body_pose'][:21, :] # use smpl body_pose on smplx

            if as_smplx == 'smplh':
                smplx_param['shape'] = np.zeros(10, dtype=np.float32) # drop smpl betas for smplx

            if smplx_param['lhand_pose'] is None:
                smplx_param['lhand_valid'] = False
            else:
                smplx_param['lhand_valid'] = True
            if smplx_param['rhand_pose'] is None:
                smplx_param['rhand_valid'] = False
            else:
                smplx_param['rhand_valid'] = True
            if smplx_param['expr'] is None:
                smplx_param['face_valid'] = False
            else:
                smplx_param['face_valid'] = True

            if joint_cam is not None and np.any(np.isnan(joint_cam)):
                continue

            datalist.append({
                'img_path': img_path,
                'img_shape': img_shape,
                'bbox': bbox,
                'lhand_bbox': lhand_bbox,
                'rhand_bbox': rhand_bbox,
                'face_bbox': face_bbox,
                'joint_img': joint_img,
                'joint_cam': joint_cam,
                'joint_valid': joint_valid,
                'smplx_param': smplx_param,
                'smplx': smplx})
            
            all_data_dict[str(subject_id)][str(frame_id)][str(cam_id)] = {
                'img_path': img_path,
                'img_shape': img_shape,
                'bbox': bbox,
                'lhand_bbox': lhand_bbox,
                'rhand_bbox': rhand_bbox,
                'face_bbox': face_bbox,
                'joint_img': joint_img,
                'joint_cam': joint_cam,
                'joint_valid': joint_valid,
                'smplx_param': smplx_param,
                # 'smplx': smplx
            }

        # save memory
        del content, image_path, bbox_xywh, lhand_bbox_xywh, rhand_bbox_xywh, face_bbox_xywh, keypoints3d, keypoints2d
        
        multi_view_data_list = []
        for subject_id in all_data_dict.keys():
            for frame_id in all_data_dict[subject_id].keys():
                temp_data_list = []
                for cam_id in all_data_dict[subject_id][frame_id].keys():
                    temp_data_list.append(all_data_dict[subject_id][frame_id][cam_id])
                # 取前四个和后四个
                if len(temp_data_list) > 4 and self.data_split == 'train':
                    multi_view_data_list.append(temp_data_list[:4])
                    multi_view_data_list.append(temp_data_list[-4:])
                elif len(temp_data_list) == 4 and self.data_split == 'train':
                    multi_view_data_list.append(temp_data_list)
                elif self.data_split != 'train':
                    multi_view_data_list.append(temp_data_list)
                    
                    
        new_multi_view_data_list = []
        for i, multi_view_data in enumerate(multi_view_data_list):
            if self.data_split == 'train' and i % train_sample_interval != 0:
                continue
            if self.data_split == 'val' and i % train_sample_interval != 0:
                continue
            if self.data_split == 'test' and i % test_sample_interval != 0:
                continue
            
            new_multi_view_data_list.append(multi_view_data)
            
        multi_view_data_list = new_multi_view_data_list
                    

        temp_sample_interval = train_sample_interval if self.data_split == 'train' else test_sample_interval
        print(f'[{self.__class__.__name__} {self.data_split}] original size:', int(num_examples),
                '. Sample interval:', train_sample_interval,
                '. Sampled size:', len(multi_view_data_list))

        if (getattr(cfg, 'data_strategy', None) == 'balance' and self.data_split == 'train') or \
                getattr(cfg, 'eval_on_train', False):
            print(f'[{self.__class__.__name__}] Using [balance] strategy with datalist shuffled...')
            random.seed(2023)
            random.shuffle(multi_view_data_list)

            if getattr(cfg, 'eval_on_train', False):
                return multi_view_data_list[:10000]

        return multi_view_data_list
    
    def __getitem__(self, idx):
        try:
            data = copy.deepcopy(self.datalist[idx])
        except Exception as e:
            print(f'[{self.__class__.__name__}] Error loading data {idx}')
            print(e)
            exit(0)
            
        multi_view_data = self.datalist[idx]
        multi_view_inputs = {}
        multi_view_targets = {}
        multi_view_meta_info = {}

        # print(f"!!!!!!: {len(multi_view_data)}")
        for cam_index, data in enumerate(multi_view_data):
            img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']

            # img
            img = load_img(img_path)
            img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
            img = self.transform(img.astype(np.float32)) / 255.

            # h36m gt
            joint_cam = data['joint_cam']
            if joint_cam is not None:
                dummy_cord = False
                joint_cam = joint_cam - joint_cam[self.joint_set['root_joint_idx'], None, :]  # root-relative
            else:
                # dummy cord as joint_cam
                dummy_cord = True
                joint_cam = np.zeros((self.joint_set['joint_num'], 3), dtype=np.float32)

            joint_img = data['joint_img']
            joint_img = np.concatenate((joint_img[:, :2], joint_cam[:, 2:]), 1)  # x, y, depth
            if not dummy_cord: 
                joint_img[:, 2] = (joint_img[:, 2] / (cfg.body_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[0]  # discretize depth
            
            joint_img_aug, joint_cam_wo_ra, joint_cam_ra, joint_valid, joint_trunc = process_db_coord(
                joint_img, joint_cam, data['joint_valid'], do_flip, img_shape,
                self.joint_set['flip_pairs'], img2bb_trans, rot, self.joint_set['joints_name'], smpl_x.joints_name)
            
            # smplx coordinates and parameters
            smplx_param = data['smplx_param']
            smplx_joint_img, smplx_joint_cam, smplx_joint_trunc, smplx_pose, smplx_shape, smplx_expr, \
            smplx_pose_valid, smplx_joint_valid, smplx_expr_valid, smplx_mesh_cam_orig = process_human_model_output(
                smplx_param, self.cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx',
                joint_img=None if self.cam_param else joint_img,  # if cam not provided, we take joint_img as smplx joint 2d, which is commonly the case for our processed humandata
            )

            # TODO temp fix keypoints3d for renbody
            if 'RenBody' in self.__class__.__name__:
                joint_cam_ra = smplx_joint_cam.copy()
                joint_cam_wo_ra = smplx_joint_cam.copy()
                joint_cam_wo_ra[smpl_x.joint_part['lhand'], :] = joint_cam_wo_ra[smpl_x.joint_part['lhand'], :] \
                                                                + joint_cam_wo_ra[smpl_x.lwrist_idx, None, :]  # left hand root-relative
                joint_cam_wo_ra[smpl_x.joint_part['rhand'], :] = joint_cam_wo_ra[smpl_x.joint_part['rhand'], :] \
                                                                + joint_cam_wo_ra[smpl_x.rwrist_idx, None, :]  # right hand root-relative
                joint_cam_wo_ra[smpl_x.joint_part['face'], :] = joint_cam_wo_ra[smpl_x.joint_part['face'], :] \
                                                                + joint_cam_wo_ra[smpl_x.neck_idx, None,: ]  # face root-relative

            # change smplx_shape if use_betas_neutral
            # processing follows that in process_human_model_output
            if self.use_betas_neutral:
                smplx_shape = smplx_param['betas_neutral'].reshape(1, -1)
                smplx_shape[(np.abs(smplx_shape) > 3).any(axis=1)] = 0.
                smplx_shape = smplx_shape.reshape(-1)
                
            # SMPLX pose parameter validity
            # for name in ('L_Ankle', 'R_Ankle', 'L_Wrist', 'R_Wrist'):
            #     smplx_pose_valid[smpl_x.orig_joints_name.index(name)] = 0
            smplx_pose_valid = np.tile(smplx_pose_valid[:, None], (1, 3)).reshape(-1)
            # SMPLX joint coordinate validity
            # for name in ('L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel'):
            #     smplx_joint_valid[smpl_x.joints_name.index(name)] = 0
            smplx_joint_valid = smplx_joint_valid[:, None]
            smplx_joint_trunc = smplx_joint_valid * smplx_joint_trunc
            if not (smplx_shape == 0).all():
                smplx_shape_valid = True
            else: 
                smplx_shape_valid = False

            # hand and face bbox transform
            lhand_bbox, lhand_bbox_valid = self.process_hand_face_bbox(data['lhand_bbox'], do_flip, img_shape, img2bb_trans)
            rhand_bbox, rhand_bbox_valid = self.process_hand_face_bbox(data['rhand_bbox'], do_flip, img_shape, img2bb_trans)
            face_bbox, face_bbox_valid = self.process_hand_face_bbox(data['face_bbox'], do_flip, img_shape, img2bb_trans)
            if do_flip:
                lhand_bbox, rhand_bbox = rhand_bbox, lhand_bbox
                lhand_bbox_valid, rhand_bbox_valid = rhand_bbox_valid, lhand_bbox_valid
            lhand_bbox_center = (lhand_bbox[0] + lhand_bbox[1]) / 2.
            rhand_bbox_center = (rhand_bbox[0] + rhand_bbox[1]) / 2.
            face_bbox_center = (face_bbox[0] + face_bbox[1]) / 2.
            lhand_bbox_size = lhand_bbox[1] - lhand_bbox[0]
            rhand_bbox_size = rhand_bbox[1] - rhand_bbox[0]
            face_bbox_size = face_bbox[1] - face_bbox[0]

            inputs = {'img': img}
            targets = {'joint_img': joint_img_aug, # keypoints2d
                    'smplx_joint_img': joint_img_aug, #smplx_joint_img, # projected smplx if valid cam_param, else same as keypoints2d
                    'joint_cam': joint_cam_wo_ra, # joint_cam actually not used in any loss, # raw kps3d probably without ra
                    'smplx_joint_cam': smplx_joint_cam if dummy_cord else joint_cam_ra, # kps3d with body, face, hand ra
                    'smplx_pose': smplx_pose,
                    'smplx_shape': smplx_shape,
                    'smplx_expr': smplx_expr,
                    'lhand_bbox_center': lhand_bbox_center, 'lhand_bbox_size': lhand_bbox_size,
                    'rhand_bbox_center': rhand_bbox_center, 'rhand_bbox_size': rhand_bbox_size,
                    'face_bbox_center': face_bbox_center, 'face_bbox_size': face_bbox_size}
            meta_info = {'joint_valid': joint_valid,
                        'joint_trunc': joint_trunc,
                        'smplx_joint_valid': smplx_joint_valid if dummy_cord else joint_valid,
                        'smplx_joint_trunc': smplx_joint_trunc if dummy_cord else joint_trunc,
                        'smplx_pose_valid': smplx_pose_valid,
                        'smplx_shape_valid': float(smplx_shape_valid),
                        'smplx_expr_valid': float(smplx_expr_valid),
                        'is_3D': float(False) if dummy_cord else float(True), 
                        'lhand_bbox_valid': lhand_bbox_valid,
                        'rhand_bbox_valid': rhand_bbox_valid, 'face_bbox_valid': face_bbox_valid}
            
            multi_view_inputs[str(cam_index)] = inputs
            multi_view_targets[str(cam_index)] = targets
            multi_view_meta_info[str(cam_index)] = meta_info
            # print(cam_index)
            
        return multi_view_inputs, multi_view_targets, multi_view_meta_info