# MUC: Mixture of Uncalibrated Cameras for Robust 3D Human Body Reconstruction (AAAI'2025)

The official implementation for the **AAAI 2025** paper \[[_MUC: Mixture of Uncalibrated Cameras for Robust 3D Human Body Reconstruction_](https://arxiv.org/pdf/2403.05055)\].

## News
- [2024-12] Training and testing code is released.

## Install
```bash
conda create -n muc python=3.9
conda activate muc
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch -y
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
pip install -r requirements.txt

# install mmpose
cd common/pose_nets/transformer_utils
pip install -v -e .
cd ../../..
```

## Preparation 
- Download all datasets          
  - [Human3.6M](http://vision.imar.ro/human3.6m/description.php)
  - [RICH](https://rich.is.tue.mpg.de/index.html)
- Download all external files from [OneDrive](https://1drv.ms/f/c/268d0615b5c36c55/EjLlyRa_RyRAt5m5ovluHXwBkx7pkbkMajLAytGql6MIxQ)
- Process RICH dataset into [HumanData](https://github.com/open-mmlab/mmhuman3d/blob/main/docs/human_data.md) format, code included in [OneDrive](https://1drv.ms/f/c/268d0615b5c36c55/EjLlyRa_RyRAt5m5ovluHXwBkx7pkbkMajLAytGql6MIxQ)
    ```
    python mmhuman3d/tools/convert_datasets.py \
        --datasets rich \
        --root_path dataset/RICH \
        --output_path dataset/RICH/preprocessed_datasets
    ```
- Download [SMPL-X](https://smpl-x.is.tue.mpg.de/) and [SMPL](https://smpl.is.tue.mpg.de/) body models.

The file structure should be like:
```
MUC/
├── common/
│   ├── pose_nets/
│   │    human_model_files/
│   │    └──smpler_x_b32.tar 
│   └── utils/
│       └── human_model_files/
│           ├── smpl/
│           │   ├──SMPL_NEUTRAL.pkl
│           │   ├──SMPL_MALE.pkl
│           │   └──SMPL_FEMALE.pkl
│           ├── smplx/
│           │   ├──J_regressor_h36m_smplx.npy
│           │   ├──MANO_SMPLX_vertex_ids.pkl
│           │   ├──SMPL-X__FLAME_vertex_ids.npy
│           │   ├──SMPLX_NEUTRAL.pkl
│           │   ├──SMPLX_to_J14.pkl
│           │   ├──SMPLX_NEUTRAL.npz
│           │   ├──SMPLX_MALE.npz
│           │   └──SMPLX_FEMALE.npz
│           └── smplx-uv/
│               ├──male_smplx.png
│               ├──smplx_human.png
│               ├──SMPLX_male.obj
│               ├──smplx_mask.png
│               ├──smplx_mask_1000.png
│               ├──smplx_uv.obj
│               └──smplx_uv.png
└── dataset/          
    ├── Human36M/
    │   ├── annotations/
    │   ├── images/
    │   └── SMPL-X/
    └── RICH/
        ├── preprocessed_datasets/
        │   ├──rich_test.npz
        │   ├──rich_train.npz
        │   └── rich_val.npz
        ├── scan_calibration/
        ├── test/
        ├── test_body/
        ├── train/
        ├── train_body/
        ├── val/
        └── val_body/
```

## Training
### For training on Human36M dataset
```
python train.py --lr 3e-5 --froze --jrn_loss --srn_loss --dataset human36m --no-full_test --num_thread 8 --end_epoch 50 --encoder_setting base --gpu 0
```
### For training on RICH dataset
```
python train.py --lr 3e-5 --froze --jrn_loss --srn_loss --dataset rich --no-full_test --num_thread 8 --end_epoch 50 --encoder_setting base --gpu 0
```

## Testing
### For testing on Human36M dataset
```
python test.py --froze --jrn_loss --srn_loss --dataset human36m --no-full_test --num_thread 8 --encoder_setting base --gpu 0
```
### For testing on RICH dataset
```
python test.py --froze --jrn_loss --srn_loss --dataset rich --no-full_test --num_thread 8 --encoder_setting base --gpu 0
```

## References
- [SMPLer-X](https://github.com/caizhongang/SMPLer-X)
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d)

## Citation
```bibtex
@article{zhu2024muc,
  title={MUC: Mixture of Uncalibrated Cameras for Robust 3D Human Body Reconstruction},
  author={Zhu, Yitao and Wang, Sheng and Xu, Mengjie and Zhuang, Zixu and Wang, Zhixin and Wang, Kaidong and Zhang, Han and Wang, Qian},
  journal={arXiv preprint arXiv:2403.05055},
  year={2024}
}
```
