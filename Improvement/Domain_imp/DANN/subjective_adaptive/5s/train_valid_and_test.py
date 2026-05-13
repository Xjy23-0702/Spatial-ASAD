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


# train the model for every subject
def pretrain_model(eegdata, eeglabel, sb):

# ----------------------initial model------------------------

    model = CNN_baseline().to(cfg.device)


    # Train and test using the current folded data
    x_train_val, y_train_val = eegdata, eeglabel
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_val, y_train_val, test_size=0.2 ,random_state=cfg.torch_seed)

    n_samples = len(x_train)
    n_target = int(n_samples * 0.2)
    indices = np.random.permutation(n_samples)
    source_indices = indices[n_target:]
    target_indices = indices[:n_target]
    
    x_source = x_train[source_indices]
    y_source = y_train[source_indices]
    x_target = x_train[target_indices]

    # get the dataset
    source_dataset = AADdataset_1second(x_source, y_source)
    target_dataset = AADdataset_1second(x_target, y_source[:len(x_target)])
    valid_dataset = AADdataset_1second(x_valid, y_valid)
    
    source_loader = DataLoader(source_dataset, batch_size=cfg.batch_size, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False)

    # train_dataset = AADdataset_1second(x_train, y_train)
    # valid_dataset = AADdataset_1second(x_valid, y_valid)

    # train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True)#增加batch维度，自动调用__getitem__
    # valid_loader = DataLoader(dataset=valid_dataset, batch_size=cfg.batch_size, shuffle=True)


    # set the criterion and optimizer
    criterion_label = nn.CrossEntropyLoss()
    criterion_domain = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_valid_loss = float('inf')
# ---------------------train and valid-----------

    for epoch in range(cfg.epoch_num):

        # train the model
        # num_correct = 0
        # num_samples = 0
        # train_loss = 0
        model.train()

        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        
        total_loss_label = 0
        total_loss_domain = 0
        num_correct = 0
        num_samples = 0
        # 动态调整GRL的alpha参数（从0逐渐增加到1）
        p = epoch / cfg.epoch_num
        alpha = 2. / (1. + np.exp(-10 * p)) - 1 
        num_batches = min(len(source_loader), len(target_loader))
        pbar = tqdm.tqdm(range(num_batches), desc=f"Epoch {epoch}/{cfg.epoch_num} [DANN]", position=0, leave=True)
        # ---------------------train---------------------
        for batch_idx in pbar:
            try:
                x_s, y_s = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                x_s, y_s = next(source_iter)
            
            try:
                x_t, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                x_t, _ = next(target_iter)
            
            x_s = x_s.to(cfg.device)
            y_s = y_s.to(cfg.device)
            x_t = x_t.to(cfg.device)
            

            domain_label_s = torch.zeros(x_s.size(0), 1).to(cfg.device)
            domain_label_t = torch.ones(x_t.size(0), 1).to(cfg.device)
            

            label_out_s, domain_out_s = model(x_s, alpha=alpha, return_domain=True)

            _, domain_out_t = model(x_t, alpha=alpha, return_domain=True)
            

            loss_label = criterion_label(label_out_s, y_s)
            loss_domain = criterion_domain(domain_out_s, domain_label_s) + \
                         criterion_domain(domain_out_t, domain_label_t)
            
            total_loss = loss_label + 0.5 * loss_domain  # 域损失权重可调
            

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            total_loss_label += loss_label.item()
            total_loss_domain += loss_domain.item()
            
            # 计算源域分类准确率（用于监控）
            _, predictions = label_out_s.max(1)
            num_correct += (predictions == y_s).sum().item()
            num_samples += y_s.size(0)
            
            # 更新进度条显示
            pbar.set_postfix({
                'loss_label': f'{loss_label.item():.3f}',
                'loss_domain': f'{loss_domain.item():.3f}',
                'acc': f'{100 * num_correct / num_samples:.1f}%'
            })
        
        avg_label_loss = total_loss_label / num_batches
        avg_domain_loss = total_loss_domain / num_batches
        train_acc = 100 * num_correct / num_samples
        

        writer.add_scalar(f'sb{sb}/DANN_Train_Loss_Label', avg_label_loss, epoch)
        writer.add_scalar(f'sb{sb}/DANN_Train_Loss_Domain', avg_domain_loss, epoch)
        writer.add_scalar(f'sb{sb}/DANN_Train_Accuracy', train_acc, epoch)
        
        # ---------------------valid---------------------
        model.eval()
        num_correct = 0
        num_samples = 0
        valid_loss = 0.0
        
        with torch.no_grad():
            for eeg, label in tqdm.tqdm(valid_loader, desc="Validating", position=0, leave=True):
                eeg = eeg.to(cfg.device)
                label = label.to(cfg.device)
                
                pred = model(eeg, alpha=0, return_domain=False)
                loss = criterion_label(pred, label)
                valid_loss += loss.item()
                
                _, predictions = pred.max(1)
                num_correct += (predictions == label).sum().item()
                num_samples += label.size(0)
        
        avg_valid_loss = valid_loss / len(valid_loader)
        valid_acc = 100 * num_correct / num_samples
        

        writer.add_scalar(f'sb{sb}/DANN_Valid_Loss', avg_valid_loss, epoch)
        writer.add_scalar(f'sb{sb}/DANN_Valid_Accuracy', valid_acc, epoch)
        
        print(f"Epoch {epoch}:")
        print(f"  Train - Loss_label: {avg_label_loss:.4f}, Loss_domain: {avg_domain_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Valid - Loss: {avg_valid_loss:.4f}, Acc: {valid_acc:.2f}%")

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            savedir = cfg.pretrain_model_dir
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            saveckpt = os.path.join(savedir, f'pretrain_sb{sb}_dann.ckpt')
            torch.save(model.state_dict(), saveckpt)
            print(f" Best model saved (valid_loss: {best_valid_loss:.4f})")
    
    return model

def finetune_model(eegdata, eeglabel, sb, patience=10):

    model = CNN_baseline().to(cfg.device)


    savedir1 = cfg.pretrain_model_dir
    saveckpt = os.path.join(savedir1, f'pretrain_sb{sb}_dann.ckpt')
    model.load_state_dict(torch.load(saveckpt))
    

    hard_subjects = [0,3,5,6,7,8,10,11,14]
    if sb in hard_subjects:
        for param in model.feature_extractor.parameters():
            param.requires_grad = True
        finetune_lr = cfg.finetune_lr * 0.1
        print(f"Subject {sb} is hard, unfreezing feature extractor")
    else:
        for param in model.feature_extractor.parameters():
            param.requires_grad = False
        finetune_lr = cfg.finetune_lr
    
    train_dataset = AADdataset_1second(eegdata, eeglabel)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=finetune_lr,
        weight_decay=cfg.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    
    best_train_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(cfg.finetune_epoch_num):
        model.train()
        train_loss = 0.0
        num_correct = 0
        num_samples = 0
        
        for eeg, label in tqdm.tqdm(train_loader, desc=f"Finetuning sb{sb}"):
            eeg = eeg.to(cfg.device)
            label = label.to(cfg.device)
            
            pred = model(eeg, alpha=0, return_domain=False)
            loss = criterion(pred, label)
            train_loss += loss.item()
            
            _, predictions = pred.max(1)
            num_correct += (predictions == label).sum().item()
            num_samples += label.size(0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_loss = train_loss / len(train_loader)
        train_acc = 100 * num_correct / num_samples
        
        writer.add_scalar(f'sb{sb}/DANN_Finetune_Loss', avg_loss, epoch)
        writer.add_scalar(f'sb{sb}/DANN_Finetune_Accuracy', train_acc, epoch)
        
        print(f"Finetune - subject: {sb}, epoch: {epoch}, loss: {avg_loss:.4f}, acc: {train_acc:.2f}%")
        
        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            patience_counter = 0
            savedir2 = cfg.finetune_model_dir
            if not os.path.exists(savedir2):
                os.makedirs(savedir2)
            saveckpt = os.path.join(savedir2, f'finetune_sb{sb}_dann.ckpt')
            torch.save(model.state_dict(), saveckpt)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
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
    saveckpt = os.path.join(savedir, f'finetune_sb{sb}_dann.ckpt')
    test_acc = 0
    model.load_state_dict(torch.load(saveckpt))
    model.eval()
    total_num = 0
    for iter, (eeg, label) in enumerate(tqdm.tqdm(test_loader, position=0, leave=True), start=1):
        with torch.no_grad():

            eeg = eeg.to(cfg.device)
            label = label.to(cfg.device)
            pred = model(eeg, alpha=0, return_domain=False)

            _, predictions = pred.max(1)
            correct = (predictions == label).sum().item()
            test_acc += correct
            total_num += predictions.size(0)

    res = 100 * test_acc / total_num
    print(' Testid %d test accuracy: %.3f %%' % (sb, res))

    return res                         
    