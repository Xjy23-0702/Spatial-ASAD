import torch

# models
model_names = ['CNN_CSP_Riemannian', 'CNN_CovarianceEnhanced']
model_name = model_names[1]
if model_name == 'CNN_CSP_Riemannian':
    result_dir='./CNN_CSP_Riemannian'
    pretrain_model_dir='./CNN_CSP_Riemannian/pretrain_model'
    finetune_model_dir='./CNN_CSP_Riemannian/finetune_model'
elif model_name == 'CNN_CovarianceEnhanced':
    result_dir='./CNN_CovarianceEnhanced'
    pretrain_model_dir='./CNN_CovarianceEnhanced/pretrain_model'
    finetune_model_dir='./CNN_CovarianceEnhanced/finetune_model'
dataset_name = 'KUL3_1D.mat'

device_ids = 5
device = torch.device(f"cuda:{device_ids}" if torch.cuda.is_available() else "cpu")
epoch_num = 100
finetune_epoch_num=50
batch_size = 128
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
decision_window = 128*5 


csp_n_filters = 6  # CSP滤波器数量
csp_fusion_mode = 'concat'  # 融合方式: 'concat', 'add', 'attention'
csp_use_logm = True  # 是否使用精确矩阵对数
csp_eps = 1e-4  # 协方差正则化系数
