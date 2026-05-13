from AADdataset import AADdataset_1point,AADdataset_1second
from torch.utils.data import DataLoader
from model import CNN_CSP
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
def pretrain_model(eegdata, eeglabel, sb, csp_filters=None):

# ----------------------initial model------------------------
    valid_loss_min = 100
    model = CNN_CSP(
        n_channels=64,
        csp_components=cfg.csp_n_components
    ).to(cfg.device)
    if csp_filters is not None:
        model.set_csp_filters(csp_filters)
    else:
        print('CSP_filter not exist!')

    # Train and test using the current folded data
    x_train_val, y_train_val = eegdata, eeglabel
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_val, y_train_val, test_size=0.2 ,random_state=cfg.torch_seed)


    # get the dataset
    train_dataset = AADdataset_1second(x_train, y_train)
    valid_dataset = AADdataset_1second(x_valid, y_valid)


    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True)#增加batch维度，自动调用__getitem__
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=cfg.batch_size, shuffle=True)


    # set the criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


# ---------------------train and valid-----------

    for epoch in range(cfg.epoch_num):

        # train the model
        num_correct = 0
        num_samples = 0
        train_loss = 0
        model.train()
        # ---------------------train---------------------
        for iter, (eeg, label) in enumerate(tqdm.tqdm(train_loader, position=0, leave=True), start=1):
            running_loss = 0.0
            # get the input
            eeg = eeg.to(cfg.device)
            label = label.to(cfg.device)

            pred = model(eeg)
            loss = criterion(pred, label)
            train_loss += loss

            # backward
            optimizer.zero_grad()  # clear the grad
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            _, predictions = pred.max(1)
            num_correct += (predictions == label).sum()
            num_samples += predictions.size(0)

        decoder_answer = float(num_correct) / float(num_samples) * 100

        # Record the results of training
        trainloss = 'sb' + str(sb) + '/Train_Loss'
        train_decoder_answer = 'sb' + str(sb) + '/Train_decoder_answer'
        writer.add_scalar(trainloss, train_loss / iter, epoch // 1)
        writer.add_scalar(train_decoder_answer, decoder_answer, epoch // 1)


        # ---------------------valid---------------------
        num_correct = 0
        num_samples = 0
        valid_loss = 0.0
        model.eval()
        for iter, (eeg, label) in enumerate(tqdm.tqdm(valid_loader, position=0, leave=True), start=1):
            with torch.no_grad():
                eeg = eeg.to(cfg.device)
                label = label.to(cfg.device)
                pred = model(eeg)
                loss = criterion(pred, label)
                valid_loss = loss + valid_loss
                _, predictions = pred.max(1)
                num_correct += (predictions == label).sum()
                num_samples += predictions.size(0)

        decoder_answer = float(num_correct) / float(num_samples) * 100

        validloss = 'sb' + str(sb) + '/Valid_Loss'
        valid_decoder_answer = 'sb' + str(sb) + '/Valid_decoder_answer'
        writer.add_scalar(validloss, valid_loss / iter, epoch // 1)
        writer.add_scalar(valid_decoder_answer, decoder_answer, epoch // 1)
        print(f"testid: {sb}epoch: {epoch},\n"
                f"valid loss: {valid_loss / iter} , valid_decoder_answer: {decoder_answer}%\n")

        # Please note that for the densenet model,
        # the result presented here is a classification accuracy of 1/128s rather than 1s
        if valid_loss_min>valid_loss / iter:
            valid_loss_min = valid_loss / iter
            savedir = cfg.pretrain_model_dir
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            saveckpt = os.path.join(savedir, f'pretrain_sb{sb}.ckpt') 
            torch.save(model.state_dict(), saveckpt)

def finetune_model(eegdata, eeglabel, sb, csp_filters=None, patience=10):

    model = CNN_CSP(
        n_channels=64,
        csp_components=cfg.csp_n_components
    ).to(cfg.device)

    savedir1 = cfg.pretrain_model_dir
    saveckpt = os.path.join(savedir1, f'pretrain_sb{sb}.ckpt') 
    model.load_state_dict(torch.load(saveckpt))
    
    if csp_filters is not None:
        model.set_csp_filters(csp_filters)
    else:
        print('CSP_filter not exist!')

    for param in model.conv_layer.parameters():
        param.requires_grad = False
    model.csp_filters.requires_grad = False
        
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


def test_model(eegdata, eeglabel, sb, csp_filters=None):

# ----------------------initial model------------------------
    model = CNN_CSP(
            n_channels=64,
            csp_components=cfg.csp_n_components
        ).to(cfg.device)

    savedir = cfg.finetune_model_dir
    saveckpt = os.path.join(savedir, f'finetune_sb{sb}.ckpt') 
    model.load_state_dict(torch.load(saveckpt))

    if csp_filters is not None:
        model.set_csp_filters(csp_filters)
    else:
        print('CSP_filter not exist!')

    # test using the current folded data
    x_test, y_test = eegdata, eeglabel

    # tough the train and valid process exist difference
    # the test_data is same,one second by one second
    test_dataset = AADdataset_1second(x_test, y_test)
    # test the data one by one
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg.batch_size//16, shuffle=False)
# -------------------------test--------------------------------------------
    # after some epochs, test model
    test_acc = 0
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
    