import os
from unittest.util import _MAX_LENGTH
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch import optim
import numpy as np
import torch.nn.functional as F

from torch.distributions import Categorical

import glob

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class SpeechDataGenerator(Dataset):
    '''
    todo: write .npy files and read, bucket?
    '''
    def __init__(self, datapath:list):
        # read from .npy file and get the list of dictionary
        datalist = []
        for path in datapath:
            data = np.load(path, allow_pickle=True).tolist()
            for stc_dict in data:
                edits_idxs = stc_dict['e_indices']
                labels_for_cme = stc_dict['op_tgts']
                edit_logits_labels = stc_dict['e_label_logits']
                for e, tgt, e_logits in zip(edits_idxs, labels_for_cme,edit_logits_labels):
                    new_dict = {
                        'embedding': stc_dict['embedding'],
                        'label_logits': stc_dict['label_logits'], 
                        'e_indices':e,
                        'op_tgts':tgt,
                        'e_label_logits': e_logits,
                    }
                    datalist.append(new_dict)
        self.data = datalist
        # print(len(self.data))
        # print(self.data)
        self.embed_len = len(self.data[0]['embedding'][0])
        self.label_len = len(self.data[0]['label_logits'][0])
        self.max_length = 128
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        sentence = self.data[idx]

        embed = torch.from_numpy(np.asarray(sentence['embedding'], dtype=np.float32)).detach().to(device)
        sentence_len = embed.size()[0]
        
        v_all = sentence['label_logits']
        v_all = torch.from_numpy(np.asarray(v_all, dtype=np.float32)).detach().to(device)
        v_all = F.pad(v_all,(0,0,0,self.max_length-sentence_len))
            
        target = sentence['op_tgts']
        target = torch.from_numpy(np.asarray(target, dtype=np.float32)).detach().to(device)
        
        edits_idx = sentence['e_indices']
        if edits_idx == 0:
            v_last = torch.zeros(self.label_len, device=device)
        else:
            v_last = v_all[edits_idx-1]
            
        e_i = embed[edits_idx]
        v_i = v_all[edits_idx]
        return [e_i, v_i, v_last, v_all, target]
    
class ConfidenceModelAttn(nn.Module):
    def __init__(self, max_len = 128, label_len = 5002, embed_len = 768):  # embedding length uncertain
        super().__init__()
        self.max_len = max_len # maximum sequence length
        self.embed_len = embed_len # length of embedding
        self.label_len = label_len # number of label classes
        
        self.attn = nn.Sequential(
                                nn.Linear(self.embed_len + self.label_len, self.max_len),
                                nn.Softmax()
                                )
        self.attn_combine = nn.Sequential(
                                nn.Linear(self.label_len * 2, self.label_len),
                                nn.ReLU()
                                )
        
        self.fc_layer = nn.Sequential(
                                nn.Linear(self.label_len,512), # todo: input size uncertain
                                nn.ReLU(),
                                nn.Linear(512,2),
                                nn.Softmax(dim=2)
                                )
                                     
    def forward(self, v_i, v_last, e_i, all_v):
        attn_weights = self.attn(torch.cat((e_i, v_last), 1))
        stc_len = all_v.size()[0]
        # if stc_len<self.max_len:
        #     attn_weights = attn_weights[:,:stc_len]
        # else:
        #     all_v = all_v[:self.max_len, :]
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 all_v)
        # print(attn_applied.size())
        # print(torch.cat((v_i, attn_applied[0]), 1).size())
        attn_combined = self.attn_combine(torch.cat((v_i.unsqueeze(1), attn_applied), 2))
        out = self.fc_layer(attn_combined).squeeze()
        return out
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size, device=device)
    
def normalized_cross_entropy(c,p):
    H_c = torch.mean(Categorical(probs = c).entropy())
    criterion = nn.BCELoss()
    H_c_p =criterion(p, c.float())
    nce = (H_c - H_c_p)/H_c
    return nce

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    train_loss = 0
    for batch, [e_i, v_i, v_last, v_all, tgt] in enumerate(dataloader):
        # preprocess dataset
        # embed = torch.squeeze(embed)
        # v_all = torch.squeeze(v_all)
        # target = torch.squeeze(target, 0)
        # print(f'embed:{e_i.size()}, v_all:{v_all.size()}, target:{tgt}, v_i:{v_i.size()}')
        
        # Compute prediction and loss
        pred = model(v_i, v_last, e_i, v_all)
        # pred = pred.squeeze(1)
        try:
            loss = loss_fn(pred, tgt)
        except ValueError:
            # except only one value batch, make sure tgt and pred have the same sizes
            loss = loss_fn(torch.squeeze(tgt), torch.squeeze(pred))
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # try:
        #     nce_val = normalized_cross_entropy(tgt, pred)
        # except ValueError:
        #     # except only one value batch, make sure tgt and pred have the same sizes
        #     nce_val = normalized_cross_entropy(torch.squeeze(tgt), torch.squeeze(pred))
        train_loss += loss
        # if batch%100==0:
        #     print(f'Loss at iteration {batch} is {loss.item()}')
        #     print(f'NCE Loss at iteration {batch} is {nce_val.item()}')
    print(f'Train NCE loss {train_loss/size}')

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    
    with torch.no_grad():
        for batch, [e_i, v_i, v_last, v_all, tgt] in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(v_i, v_last, e_i, v_all)
            try:
                loss = loss_fn(pred, tgt)
            except:
                # except only one value batch, make sure tgt and pred have the same sizes
                loss = loss_fn(torch.squeeze(tgt), torch.squeeze(pred))
                print(f'pred:{pred}')
            # loss = loss_fn(pred, tgt)
            test_loss += loss.item()
            
            # nce_val = normalized_cross_entropy(y, pred)
            # nce_list.append(nce_val.item())
           
        print(f'test NCE loss {test_loss/size}')

def main():
    train_file = ['/home/zl437/rds/hpc-work/gector/CEM/data/bea_nucle/trainset1.npy',]
                #   '/home/zl437/rds/hpc-work/gector/CEM/data/bea_nucle/trainset2.npy',
                #   '/home/zl437/rds/hpc-work/gector/CEM/data/bea_nucle/trainset3.npy']
    test_file = ['/home/zl437/rds/hpc-work/gector/CEM/data/bea_nucle/testset1.npy',]
                #   '/home/zl437/rds/hpc-work/gector/CEM/data/bea_nucle/testset2.npy',
                #   '/home/zl437/rds/hpc-work/gector/CEM/data/bea_nucle/testset3.npy']
    batch_size = 35

    train_dataset = SpeechDataGenerator(train_file) 
    test_dataset = SpeechDataGenerator(test_file)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    confidence_model = ConfidenceModelAttn().to(device)

    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(confidence_model.parameters(), lr=0.0001, weight_decay=1e-05, betas=(0.9, 0.98), eps=1e-9)

    epochs = 7
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, confidence_model, criterion, optimizer)
        test_loop(test_dataloader, confidence_model, criterion)
    print("Done!")
    
    torch.save(confidence_model, '/home/zl437/rds/hpc-work/gector/CEM/model/confidence_model_attn.pt')

if __name__ == '__main__':
    main()