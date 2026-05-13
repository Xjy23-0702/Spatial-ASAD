from AADdataset import AADdataset_1point,AADdataset_1second
from torch.utils.data import DataLoader
from model import CNN_baseline
import tqdm
import torch
import config as cfg
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os

writer = SummaryWriter()

def coral_loss(source_features, target_features):
    d = source_features.size(1)
    
    # 减去均值
    source_features = source_features - torch.mean(source_features, dim=0)
    target_features = target_features - torch.mean(target_features, dim=0)
    
    # 计算协方差矩阵
    source_cov = torch.mm(source_features.T, source_features) / (source_features.size(0) - 1)
    target_cov = torch.mm(target_features.T, target_features) / (target_features.size(0) - 1)
    
    # 添加小的正则项保证数值稳定
    eye = torch.eye(d).to(cfg.device) * 1e-5
    source_cov += eye
    target_cov += eye
    
    # Frobenius范数
    loss = torch.norm(source_cov - target_cov, p='fro') ** 2
    loss = loss / (4 * d * d)  # 归一化
    
    return loss
# train the model for every subject
def pretrain_model(eegdata, eeglabel, sb):

# ----------------------initial model------------------------
    valid_loss_min = 100
    model = CNN_baseline().to(cfg.device)


    # Train and test using the current folded data
    x_train_val, y_train_val = eegdata, eeglabel
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_val, y_train_val, test_size=0.2 ,random_state=cfg.torch_seed)


    # get the dataset
    n_samples = len(x_train)
    n_target = int(n_samples * 0.2)
    indices = np.random.permutation(n_samples)
    source_indices = indices[n_target:]
    target_indices = indices[:n_target]
    
    x_source = x_train[source_indices]
    y_source = y_train[source_indices]
    x_target = x_train[target_indices]
    y_target = y_train[target_indices]
    
    # 创建数据集
    source_dataset = AADdataset_1second(x_source, y_source)
    target_dataset = AADdataset_1second(x_target, y_target)
    valid_dataset = AADdataset_1second(x_valid, y_valid)
    
    source_loader = DataLoader(source_dataset, batch_size=cfg.batch_size, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False)

    # set the criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


# ---------------------train and valid-----------

    for epoch in range(cfg.epoch_num):

        model.train()
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        
        num_correct = 0
        num_samples = 0
        train_loss = 0
        coral_loss_value = 0
        
        # CORAL权重：从0逐渐增加到0.05
        coral_weight = min(0.05, epoch / cfg.epoch_num * 0.05)
        
        num_batches = min(len(source_loader), len(target_loader))
        pbar = tqdm.tqdm(range(num_batches), desc=f"Epoch {epoch}/{cfg.epoch_num} [CORAL]", position=0, leave=True)
        # ---------------------train---------------------
        for batch_idx in pbar:
            # 获取源域数据
            try:
                x_s, y_s = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                x_s, y_s = next(source_iter)
            
            # 获取目标域数据
            try:
                x_t, y_t = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                x_t, y_t = next(target_iter)
            
            x_s = x_s.to(cfg.device)
            y_s = y_s.to(cfg.device)
            x_t = x_t.to(cfg.device)
            
            # 前向传播获取特征和预测
            # 源域数据
            pred_s = model(x_s)
            loss_label = criterion(pred_s, y_s)
            
            # 提取特征用于CORAL损失
            # 需要在模型中添加get_features方法
            if hasattr(model, 'get_features'):
                features_s = model.get_features(x_s)
                features_t = model.get_features(x_t)
                loss_coral = coral_loss(features_s, features_t)
                total_loss = loss_label + coral_weight * loss_coral
                coral_loss_value = loss_coral.item()
            else:
                total_loss = loss_label
                coral_loss_value = 0
            
            train_loss += loss_label.item()
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # 计算准确率
            _, predictions = pred_s.max(1)
            num_correct += (predictions == y_s).sum().item()
            num_samples += y_s.size(0)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss_label.item():.3f}',
                'coral': f'{coral_loss_value:.3f}',
                'acc': f'{100 * num_correct / num_samples:.1f}%'
            })
        
        train_acc = 100 * num_correct / num_samples
        avg_train_loss = train_loss / num_batches
        
        # 记录训练指标
        writer.add_scalar(f'sb{sb}/CORAL_Train_Loss', avg_train_loss, epoch)
        writer.add_scalar(f'sb{sb}/CORAL_Train_Accuracy', train_acc, epoch)
        
        # ---------------------验证---------------------
        model.eval()
        num_correct = 0
        num_samples = 0
        valid_loss = 0.0
        
        # 验证集也用tqdm
        with torch.no_grad():
            for eeg, label in tqdm.tqdm(valid_loader, desc="Validating", position=0, leave=True):
                eeg = eeg.to(cfg.device)
                label = label.to(cfg.device)
                pred = model(eeg)
                loss = criterion(pred, label)
                valid_loss += loss.item()
                
                _, predictions = pred.max(1)
                num_correct += (predictions == label).sum().item()
                num_samples += label.size(0)
        
        avg_valid_loss = valid_loss / len(valid_loader)
        valid_acc = 100 * num_correct / num_samples
        
        # 记录验证指标
        writer.add_scalar(f'sb{sb}/CORAL_Valid_Loss', avg_valid_loss, epoch)
        writer.add_scalar(f'sb{sb}/CORAL_Valid_Accuracy', valid_acc, epoch)
        
        print(f"Epoch {epoch}:")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Valid - Loss: {avg_valid_loss:.4f}, Acc: {valid_acc:.2f}%")
        
        # 保存最佳模型
        if avg_valid_loss < valid_loss_min:
            valid_loss_min = avg_valid_loss
            savedir = cfg.pretrain_model_dir
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            saveckpt = os.path.join(savedir, f'pretrain_sb{sb}_coral.ckpt')
            torch.save(model.state_dict(), saveckpt)
            print(f" Best model saved (valid_loss: {valid_loss_min:.4f})")

def finetune_model(eegdata, eeglabel, sb, patience=10):

    model = CNN_baseline().to(cfg.device)


    savedir1 = cfg.pretrain_model_dir
    saveckpt = os.path.join(savedir1, f'pretrain_sb{sb}_coral.ckpt')
    model.load_state_dict(torch.load(saveckpt))
    

    for param in model.conv_layer.parameters():
        param.requires_grad = False

    train_dataset = AADdataset_1second(eegdata, eeglabel)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=cfg.finetune_lr, 
        weight_decay=cfg.weight_decay
    )
    best_train_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(cfg.finetune_epoch_num):
        model.train()
        train_loss = 0.0
        
        for iter, (eeg, label) in enumerate(tqdm.tqdm(train_loader, position=0, leave=True), start=1):
            eeg = eeg.to(cfg.device)
            label = label.to(cfg.device)

            pred = model(eeg)
            loss = criterion(pred, label)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / iter
        
        # 计算训练准确率
        model.eval()
        num_correct = 0
        num_samples = 0
        with torch.no_grad():
            for eeg, label in train_loader:
                eeg = eeg.to(cfg.device)
                label = label.to(cfg.device)
                pred = model(eeg)
                _, predictions = pred.max(1)
                num_correct += (predictions == label).sum()
                num_samples += predictions.size(0)
        train_acc = float(num_correct) / float(num_samples) * 100
        writer.add_scalar(f'sb{sb}/Finetune_Loss', avg_train_loss, epoch)
        writer.add_scalar(f'sb{sb}/Finetune_Accuracy', train_acc, epoch)
        print(f"Finetune - subject: {sb}, epoch: {epoch}, "
              f"loss: {avg_train_loss:.4f}, acc: {train_acc:.2f}%")

        # 保存最佳模型和早停
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            patience_counter = 0
            savedir2 = cfg.finetune_model_dir
            if not os.path.exists(savedir2):
                os.makedirs(savedir2)
            saveckpt = os.path.join(savedir2, f'finetune_sb{sb}.ckpt') 
            torch.save(model.state_dict(), saveckpt)
            print(f"   Best model saved (loss: {best_train_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   Early stopping triggered at epoch {epoch}")
                break


def test_model(eegdata, eeglabel, sb):

# ----------------------initial model------------------------


    model = CNN_baseline().to(cfg.device)


    # test using the current folded data
    x_test, y_test = eegdata, eeglabel

    # tough the train and valid process exist difference
    # the test_data is same,one second by one second
    test_dataset = AADdataset_1second(x_test, y_test)
    # test the data one by one
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg.batch_size//16, shuffle=False)
# -------------------------test--------------------------------------------
    # after some epochs, test model
    savedir = cfg.finetune_model_dir
    saveckpt = os.path.join(savedir, f'finetune_sb{sb}.ckpt') 
    test_acc = 0
    model.load_state_dict(torch.load(saveckpt))
    model.eval()
    total_num = 0
    for iter, (eeg, label) in enumerate(tqdm.tqdm(test_loader, position=0, leave=True), start=1):
        with torch.no_grad():

            eeg = eeg.to(cfg.device)
            label = label.to(cfg.device)
            pred = model(eeg)

            _, predictions = pred.max(1)
            correct = (predictions == label).sum().item()
            test_acc += correct
            total_num += predictions.size(0)

    res = 100 * test_acc / total_num
    print(' Testid %d test accuracy: %.3f %%' % (sb, res))

    return res                         
    