python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 train.py --lr 1e-5 --no-froze --jrn_loss --srn_loss --dataset human36m --no-full_test --num_thread 8 --end_epoch 50 --encoder_setting base --gpu 0,1