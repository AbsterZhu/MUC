import os
import os.path as osp
import sys

class Config:
    ## input, output size
    input_img_shape = (512, 384)
    input_body_shape = (256, 192)
    output_hm_shape = (16, 16, 12)
    input_hand_shape = (256, 256)
    output_hand_hm_shape = (16, 16, 16)
    output_face_hm_shape = (8, 8, 8)
    input_face_shape = (192, 192)
    
    input_body_uv_shape = (1024, 1024)
    input_face_uv_shape = (518, 600)
    output_body_uv_shape = (64, 64)
    output_face_uv_shape = (64, 64)
    focal = (5000, 5000)  # virtual focal lengths
    princpt = (input_body_shape[1] / 2, input_body_shape[0] / 2)  # virtual principal point position
    body_3d_size = 2
    hand_3d_size = 0.3
    face_3d_size = 0.3
    camera_3d_size = 2.5
    bbox_ratio = 1.25
    
    use_cache = True

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..', '..')
    data_dir = osp.join(root_dir, 'dataset')

    human_model_path = osp.join(root_dir, 'common', 'utils', 'human_model_files')
    smpl_kid_template_path = osp.join(human_model_path, 'smplx', 'smplx_kid_template.npy')
    human_mask_path = osp.join(root_dir, 'common', 'utils', 'human_model_files', 'smplx_uv', 'female_smplx.png')
    human_uv_path = osp.join(root_dir, 'common', 'utils', 'human_model_files', 'smplx_uv', 'smplx_model_300_20220615.obj')
    
    uv_map_path = osp.join(root_dir, 'common', 'utils', 'human_model_files', 'smplx_uv', 'smplx_uv.png')
    uv_mask_path = osp.join(root_dir, 'common', 'utils', 'human_model_files', 'smplx_uv', 'smplx_mask_1000.png')
    smplx_head_idx_path = osp.join(root_dir, 'common', 'utils', 'human_model_files', 'smplx', 'SMPL-X__FLAME_vertex_ids.npy')
    uv_mesh_path = osp.join(root_dir, 'common', 'utils', 'human_model_files', 'smplx_uv', 'smplx_uv.obj')
    
    calibration_path = osp.join(root_dir, 'dataset', 'RICH', 'scan_calibration')
    
    human36m_J17_regressor = osp.join(root_dir, 'common', 'utils', 'human_model_files', 'smplx', 'J_regressor_h36m_smplx.npy')

    ## model setting
    encoder_setting = 'large'
    testset = 'rich'
    pretrained_model_path = os.path.join(root_dir, 'common/pose_nets/pretrained_models/smpler_x_l32.pth.tar')
    upscale = 4
    hand_pos_joint_num = 20
    face_pos_joint_num = 72
    num_task_token = 24
    feat_dim = 1024
    num_noise_sample = 0
    encoder_config_file = os.path.join(root_dir, 'common', 'pose_nets', 'transformer_utils/configs/smpler_x/encoder/body_encoder_large.py')
    encoder_pretrained_model_path = os.path.join(root_dir, 'common', 'pose_nets', '../pretrained_models/vitpose_large.pth')
    train_batch_size = 4
    trainset_3d = ['human36m', 'rich']
    
    
    smplx_loss_weight = 1.0 #2 for agora_model for smplx shape
    smplx_pose_weight = 10.0

    smplx_kps_3d_weight = 100.0
    smplx_kps_2d_weight = 1.0
    net_kps_2d_weight = 1.0
    
    
    def set_additional_args(self, **kwargs):
        names = self.__dict__
        for k, v in kwargs.items():
            names[k] = v
        if self.encoder_setting == 'base':
            self.feat_dim = 768
            self.pretrained_model_path = os.path.join(cfg.root_dir, 'common/pose_nets/pretrained_models/smpler_x_b32.pth.tar')
            self.encoder_config_file = os.path.join(cfg.root_dir, 'common', 'pose_nets', 'transformer_utils/configs/smpler_x/encoder/body_encoder_base.py')
            self.encoder_pretrained_model_path = os.path.join(cfg.root_dir, 'common', 'pose_nets', '../pretrained_models/vitpose_base.pth')
        elif self.encoder_setting == 'large':
            self.feat_dim = 1024
            self.pretrained_model_path = os.path.join(cfg.root_dir, 'common/pose_nets/pretrained_models/smpler_x_l32.pth.tar')
            self.encoder_config_file = os.path.join(cfg.root_dir, 'common', 'pose_nets', 'transformer_utils/configs/smpler_x/encoder/body_encoder_large.py')
            self.encoder_pretrained_model_path = os.path.join(cfg.root_dir, 'common', 'pose_nets', '../pretrained_models/vitpose_large.pth')
        elif self.encoder_setting == 'huge':
            self.feat_dim = 1280
            self.pretrained_model_path = os.path.join(cfg.root_dir, 'common/pose_nets/pretrained_models/smpler_x_h32.pth.tar')
            self.encoder_config_file = os.path.join(cfg.root_dir, 'common', 'pose_nets', 'transformer_utils/configs/smpler_x/encoder/body_encoder_huge.py')
            self.encoder_pretrained_model_path = os.path.join(cfg.root_dir, 'common', 'pose_nets', '../pretrained_models/vitpose_huge.pth')
            
    def set_args(self, gpu_ids, lr=1e-4, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.lr = float(lr)
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))
            

cfg = Config()

## add some paths to the system root dir
sys.path.insert(0, cfg.root_dir)
# sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from common.utils.dir import add_pypath
add_pypath(osp.join(cfg.data_dir))
add_pypath(osp.join(cfg.root_dir, 'data'))
add_pypath(cfg.data_dir)
