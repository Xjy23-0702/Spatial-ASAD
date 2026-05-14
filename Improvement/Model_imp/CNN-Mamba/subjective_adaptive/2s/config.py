import torch

# models
pretrain_model_dir='./pretrain_model'
finetune_model_dir='./finetune_model'
dataset_name = 'KUL3_1D.mat'

device_ids = 5
device = torch.device(f"cuda:{device_ids}" if torch.cuda.is_available() else "cpu")
epoch_num = 100
finetune_epoch_num=50
batch_size = 192
sample_rate = 128
categorie_num = 2
sbnum = 16
trnum=8
kfold_num = 5
fine_ratio=0.2
lr=1e-3
finetune_lr=5e-5
weight_decay=0.01
torch_seed=2025
# the length of decision window
decision_window = 128*2
lr_temporal_conv = 1e-4   # 时间卷积层学习率
lr_spatial_conv = 1e-4    # 空间卷积层学习率
lr_mamba = 1e-3           # Mamba层学习率
lr_fc = 1e-3              # 全连接层学习率


