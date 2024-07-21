import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch import optim
import numpy as np
import copy

from torch.distributions import Categorical
import matplotlib.pyplot as plt

from sklearn.metrics import auc

if __name__ == '__main__':
    from tools import prob2pr, plot_single_rel_graph, get_best_f, get_accuracy
    from tools import AverageMeter, accuracy

import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def model_performance(model, dataloader):
    criterion = nn.BCELoss().to(device)
    
    probs = []
    labels = []

    with torch.no_grad():
         for batch, (features, tgt, prob, _) in enumerate(dataloader):
            features=features.to(device)
            features = torch.cat((features, prob.unsqueeze(dim=1).to(device)),dim=1)
            tgt=tgt.to(device)
            pred = model(features)
            pred = pred.squeeze(1)
            # pred = (pred-min(pred).item())/(max(pred).item()-min(pred).item())
            loss = criterion(pred, tgt).item()
          
            probs.extend(pred.tolist())
            labels.extend(tgt.tolist())

    p,r,t = prob2pr(probs, labels)
    auc1 = auc(r,p)
    acc, f = get_accuracy(probs, labels)
    loss = criterion(torch.tensor(probs), torch.tensor(labels))
    
    ece = plot_single_rel_graph(probs, labels)
    best_f, best_acc = get_best_f(p, r, t)
    print(f'ece:{ece},auc:{auc1}, acc:{acc}, f"{f}, loss:{loss}, best f:{best_f}, best_acc:{best_acc}')
        
    return auc1, loss, best_f

class SpeechDataGenerator(Dataset):

    def __init__(self, datapath):
        # read from .npy file and get the list of dictionary
        datalist = []
        for path in datapath:
            data = np.load(path, allow_pickle=True).tolist()
            datalist.extend(data)
        self.data = datalist
        self.embed_len = len(self.data[0]['e_label_logits'])
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]

        prob = np.asarray(item['prob'], dtype=np.float32)
        idx = np.asarray(item['idx'], dtype=np.intc)
        features = np.asarray(item['e_label_logits'], dtype=np.float32)

        features = torch.from_numpy(features).detach().to(device)
        target = item['op_tgts']
        target = np.asarray(target, dtype=np.float32)
        target = torch.from_numpy(target).detach().to(device)
        return features, target, prob,idx

class ConfidenceModel(nn.Module):
    # a simple fully connected layer applied to logits             
    def __init__(self,):
        super().__init__()
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.1)
                m.bias.data.fill_(0.0)
                m.weight.data[0][-1] = 1.0
                # print(m.weight.data.shape)
                # print(m.weight.data[0])
                    
        self.fc_layer = nn.Sequential(
                                    #   nn.Dropout(p = 0.5),
                                      nn.Linear(768+1,1), # todo: input size uncertain
                                    #   nn.ReLU(),
                                    #   nn.Linear(300,1),
                                      nn.Sigmoid(),
                                      )
        self.fc_layer.apply(init_weights)
                                     
    def forward(self,x):
        out = self.fc_layer(x)
        return out

def normalized_cross_entropy(c,p):
    H_c = torch.mean(Categorical(probs = c).entropy())
    criterion = nn.BCELoss()
    H_c_p =criterion(p, c.float())
    nce = (H_c - H_c_p)/H_c
    return nce

def rank_loss(pred,tgt):
    target = tgt
    ranked, idx = torch.sort(pred)
    # import pdb; pdb.set_trace()
    ranked = ranked/ranked.shape[0] - 0.5/ranked.shape[0]
    
    criterion = nn.MSELoss().to(device)
    loss =criterion(ranked, target[idx])
    
    return loss

def train_loop(dataloader, model, loss_fn, optimizer):
    losses = AverageMeter()
    accs = AverageMeter()
    for batch, (features, tgt, prob, _) in enumerate(dataloader):
        # Compute prediction and loss
        features=features.to(device)
        features = torch.cat((features, prob.unsqueeze(dim=1).to(device)),dim=1)
        tgt=tgt.to(device)
        
        pred = model(features)
        pred = pred.squeeze(1)
        # pred = (pred-min(pred).item())/(max(pred).item()-min(pred).item())
        loss = loss_fn(pred, tgt)
        
        # import pdb; pdb.set_trace()
        
        acc = accuracy(pred,tgt)
        accs.update(acc, features.size(0))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss, features.size()[0])
        # if batch%100==0:
        #     print(f'Loss at iteration {batch} is {losses.val} ({losses.avg})')
    print(f'Train loss {losses.avg}, accuracy {accs.avg}')
    
    return losses.avg

def test_loop(dataloader, model, loss_fn):
    losses = AverageMeter()
    accs = AverageMeter()
    
    with torch.no_grad():
        for batch, (features, tgt, prob, _) in enumerate(dataloader):
            features=features.to(device)
            features = torch.cat((features, prob.unsqueeze(dim=1).to(device)),dim=1)
            tgt=tgt.to(device)
            pred = model(features)
            pred = pred.squeeze(1)
            # pred = (pred-min(pred).item())/(max(pred).item()-min(pred).item())
            loss = loss_fn(pred, tgt).item()

            acc = accuracy(pred,tgt)
            accs.update(acc, features.size(0))

            losses.update(loss, features.size(0))
            
            # if batch%100==0:
            #     print(f'Loss at iteration {batch} is {losses.val} ({losses.avg})')
           
        print(f'test loss {losses.avg}, acc {accs.avg}')
        
    return losses.avg

def main(args):
    confidence_model = ConfidenceModel().to(device)
    idx= args.train_file_idx
    
    folder_bea = '/home/zl437/rds/hpc-work/gector/CEM/data/bea_nucle'
    folder_fce = '/home/zl437/rds/hpc-work/gector/CEM/data/fce'

    train_files = [folder_fce, folder_bea]
    
    train_file = [train_files[idx]+'/train_emb.npy',
                  train_files[idx]+'/train_emb_spec.npy']
    test_file = [train_files[idx]+'/test_emb.npy']

    train_dataset = SpeechDataGenerator(train_file) 
    test_dataset = SpeechDataGenerator(test_file)
    
    batch_size = 400
    # batch_size = 20000

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(confidence_model.parameters(), lr=1e-3, weight_decay=1e-5, betas=(0.9, 0.98), eps=1e-9)

    train_loss = []
    test_loss = []
    aucs = []
    best_fs = []
    models = []
    
    if args.load_pre_trained:
        checkpoint = torch.load(args.model_path, map_location=device)
        confidence_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']

        test_loss.append(loss)
        aucs.append(checkpoint['auc'])
        best_fs.append(checkpoint['fscore'])
        
        models.append(checkpoint['model_state_dict'])
    
    epochs = 100
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        tr_ls=train_loop(train_dataloader, confidence_model, criterion, optimizer)
        tst_ls=test_loop(test_dataloader, confidence_model, criterion)

        auc, loss, best_f = model_performance(confidence_model, test_dataloader) # print model performance
        copy_model = copy.deepcopy(confidence_model)
        
        aucs.append(auc)
        best_fs.append(best_f)
        train_loss.append(tr_ls)
        test_loss.append(tst_ls)
        models.append(copy_model.state_dict())
    # optimizer = optim.Adam(confidence_model.parameters(), lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.98), eps=1e-9)
    # for t in range(40):
    #     print(f"Epoch {t+1}\n-------------------------------")
    #     train_loss.append(train_loop(train_dataloader, confidence_model, criterion, optimizer))
    #     test_loss.append(test_loop(test_dataloader, confidence_model, criterion))
    print("Done!")
    
    # pick the epoch that gives best AUC
    auc_array = np.array(aucs)
    idx = np.argmax(auc_array)
    print(f'best auc:{auc_array[idx]}, loss:{test_loss[idx]}, best f:{best_fs[idx]}')
    
    plt.plot(range(len(test_loss)), test_loss)
    plt.plot(range(len(aucs)), aucs)
    plt.plot(range(len(best_fs)), best_fs)
    
    plt.xlabel('epochs')

    plt.legend(['test loss', 'AUC', 'F 0.5'])
    plt.savefig('loss_epoch.png')
    
    if args.model_path:
        torch.save({
                'model_state_dict': models[idx],
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss[idx],
                'auc':auc_array[idx],
                'fscore':best_fs[idx],
                }, args.model_path)
    # torch.save(confidence_model, )
    
    test_output_model(args.model_path, test_dataloader, criterion)

def test_output_model(modelpath, testloader, criterion):
    confidence_model = ConfidenceModel().to(device)
    
    checkpoint = torch.load(modelpath)
    confidence_model.load_state_dict(checkpoint['model_state_dict'])
   
    test_loop(testloader, confidence_model, criterion)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path',
                        help='Path to save output model.',
                        default='/home/zl437/rds/hpc-work/gector/CEM/model/cm_fce.pt'
                        )
    parser.add_argument('--train_file_idx',
                        help='idx of path to the train file.',
                        type=int,
                        default=0
                        )
    parser.add_argument('--load_pre_trained',
                    type=int,
                    default=0
                    )
    args = parser.parse_args()
    main(args)