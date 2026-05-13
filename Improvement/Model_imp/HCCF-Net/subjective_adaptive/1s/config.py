import torch

# models
process_data_dir = '/disk1/jyxu/DaChuang'
result_dir='/disk1/jyxu/DaChuang/Improvement/Model_imp/log_variance'
pretrain_model_dir='/disk1/jyxu/DaChuang/Improvement/Model_imp/log_variance/pretrain_model'
finetune_model_dir='/disk1/jyxu/DaChuang/Improvement/Model_imp/log_variance/finetune_model'
dataset_name = 'KUL3_1D.mat'

device_ids = 5
device = torch.device(f"cuda:{device_ids}" if torch.cuda.is_available() else "cpu")
epoch_num = 100
finetune_epoch_num=50
batch_size = 256
sample_rate = 128
categorie_num = 2
sbnum = 16
trnum=8
kfold_num = 5
fine_ratio=0.2
lr=1e-3
finetune_lr=1e-4
weight_decay=0.01
torch_seed=2025
# the length of decision window
decision_window = 128 
csp_n_components = 3


