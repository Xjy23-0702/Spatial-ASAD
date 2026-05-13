import torch

# models
model_names = ['CNN_baseline','CNN_2D','DenseNet_37']
model_name = model_names[0] # you could change the code to other models by only changing the number
process_data_dir = '/disk1/jyxu/DaChuang/REFER_CODE'
result_dir='/disk1/jyxu/DaChuang/REFER_CODE/CNN-baseline_Subject_adaptive/10s'
pretrain_model_dir='/disk1/jyxu/DaChuang/REFER_CODE/CNN-baseline_Subject_adaptive/10s/pretrain_model'
finetune_model_dir='/disk1/jyxu/DaChuang/REFER_CODE/CNN-baseline_Subject_adaptive/10s/finetune_model'
if model_name == 'CNN_baseline':
    dataset_name = 'KUL3_1D.mat'
else:
    dataset_name = 'KUL3_2D.mat'

device_ids = 5
device = torch.device(f"cuda:{device_ids}" if torch.cuda.is_available() else "cpu")
epoch_num = 100
finetune_epoch_num=50
batch_size = 64
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
decision_window = 128*10



