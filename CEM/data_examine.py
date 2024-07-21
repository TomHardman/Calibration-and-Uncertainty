import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

from cem_model import SpeechDataGenerator
from cem_w_self_a import SpeechDataGenerator as selfAttnDataGenerator
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import auc
from torch import nn
import torch
import pandas as pd
import torch.nn.functional as F

import os

from predict import predict_wo_a, predict_baseline, predict_w_self_a,reliability_graph
from tools import prob2pr, plot_single_rel_graph, get_best_f, get_accuracy

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_baseline(filename,col = 0):
    '''
    draw pr curve according to file 
    with first column to be the probability/confidence, second col to be the true labels
    '''
    with open(filename,'r',encoding='UTF-8') as f:
        lines = f.readlines()

    probs = [float(l.rstrip('\n').split()[col]) for l in lines]
    # print(probs)
    probs = np.array(probs)
    probs[probs==float('inf')] = 1e9
    labels = [int(l.rstrip('\n').split()[1]) for l in lines]
    
    idx = [int(l.rstrip('\n').split()[2]) for l in lines]


    return probs, labels, idx

def calibrated_baseline(filepath = '/home/zl437/rds/hpc-work/gector/fce-data/with_prob/calibrated_baseline_dev.npy'):
    data = np.load(filepath, allow_pickle=True)

    probs = data[0]
    labels = data[1]
    
    precisions, recalls, thresholds = prob2pr(probs, labels)
    rel1, accs, ECE1 = reliability_graph(probs, labels, n=20)
    AUC = auc(recalls, precisions)
    
    criterion = nn.BCELoss().to(device)
    loss1 = criterion(torch.tensor(probs), torch.tensor(labels))
    
    best_f, best_acc,_ = get_best_f(precisions, recalls, thresholds)
    
    acc1, f1 = get_accuracy(probs, labels, threshold=0.5)

    # plot_single_rel_graph(probs, labels, filename='calibrated_baseline.png')
    print(f'baseline ece:{ECE1}, auc:{AUC}, acc:{acc1}, f0.5:{f1},loss:{loss1}')
    print(f'precision:{best_acc},best f:{best_f}')
    
    return probs, labels, precisions, recalls, thresholds #, nums

def predict_wo_a_folder(datapaths, model_path_folder, batch_size = 1):
    from cem_model import ConfidenceModel
    
    files = os.listdir(model_path_folder)
    model_lists = []
    
    for f in files:
        model_path = model_path_folder+'/'+f
        
        model = ConfidenceModel().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model_lists.append(model)

    probs = []
    labels = []
    idxes = []
    
    dataset = SpeechDataGenerator(datapaths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    with torch.no_grad():
        for features, tgt, prob, idx in dataloader:
            # Compute prediction and loss
            features=features.to(device)
            features = torch.cat((features, prob.unsqueeze(dim=1).to(device)),dim=1)
            tgt=tgt.to(device)
            
            preds = []
            for m in model_lists:
                pred = m(features)
                preds.append(pred.unsqueeze(dim=0))
                
            # average over all preds
            preds = torch.cat(preds, dim=0)
            pred = torch.mean(preds, dim=0)
            
            probs.append(pred[0].item())
            labels.append(tgt[0].item())
            idxes.append(idx[0].item())
            
    return probs, labels, idxes

def predict_w_self_a_folder(datapath, model_path_folder, batch_size = 100):
    # the average output from a set of models in a folder
    from cem_w_self_a import ConfidenceModelAttn
    from cem_w_self_a import collate_wrapper
    
    files = os.listdir(model_path_folder)
    model_lists = []
    
    for f in files:
        model_path = model_path_folder+'/'+f
        
        model_w_self_a = ConfidenceModelAttn().to(device)
        # import pdb;pdb.set_trace()
        checkpoint = torch.load(model_path, map_location=device)
        model_w_self_a.load_state_dict(checkpoint['model_state_dict'])
        loss = checkpoint['loss']
        
        model_lists.append(model_w_self_a)
    # print(loss)
    
    dataset = selfAttnDataGenerator(datapath)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_wrapper)
    
    probs = []
    labels = []

    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            # Compute prediction and loss
            mask = data.op_mask
            
            preds = []
            for m in model_lists:
                pred = m(data.inp, data.mask, data.prob)
                preds.append(pred.unsqueeze(dim=0))
                
            # average over all preds
            preds = torch.cat(preds, dim=0)
            pred = torch.mean(preds, dim=0)

            tgt = data.tgt
            pred = torch.masked_select(pred, mask)
            tgt = torch.masked_select(tgt, mask)
            
            if pred.size() != tgt.size():
                import pdb; pdb.set_trace()
          
            # pred = [pred[i].item() for i in range(len(mask)) if mask[i]]
            # tgt = [tgt[i].item() for i in range(len(mask)) if mask[i]]

            # import pdb; pdb.set_trace()
            probs.extend(pred.tolist())
            labels.extend(tgt.tolist())
        
    return probs, labels

def plot_rel_graph(preds, label1, probs, label2, legends = ['baseline', 'cem model'], plot_figure = False):
    rel1, accs, ECE1 = reliability_graph(preds, label1, n=20)
    rel2, accs, ECE2 = reliability_graph(probs, label2, n=20)
    m1_t = pd.DataFrame({
        legends[0] : rel1,
        legends[1] : rel2,
        'accuracy' : accs,
        })
    
    if plot_figure:
        fig, axs = plt.subplots()
        m1_t[[legends[0],legends[1]]].plot(kind='bar', legend=True)
        m1_t['accuracy'].plot(kind='line', marker='*', color='black', legend=True)
    
        plt.savefig("/home/zl437/rds/hpc-work/gector/CEM/rel_fce.png")
        
    return ECE1, ECE2

def baseline_only(datapath):
    criterion = nn.BCELoss().to(device)
    
    from cem_w_self_a import collate_wrapper
    dataset = selfAttnDataGenerator([datapath])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_wrapper)
    
    probs = []
    labels = []

    for batch, data in enumerate(dataloader):
        # Compute prediction and loss
        tgt = torch.masked_select(data.tgt, data.op_mask)
        prob = torch.masked_select(data.prob, data.op_mask)
        probs.extend(prob.tolist())
        labels.extend(tgt.tolist())
    
    print(len(probs))
    print(len(labels))  
    p,r, t = prob2pr(probs, labels)
    auc1 = auc(r,p)
    acc1, f1 = get_accuracy(probs, labels)
    loss1 = criterion(torch.tensor(probs), torch.tensor(labels))
    
    ECE1= plot_single_rel_graph(probs, labels)
    
    best_f, best_acc,_ = get_best_f(p, r, t)
    print(f'baseline ece:{ECE1}, auc:{auc1}, acc:{acc1}, f0.5:{f1},loss:{loss1}, best f:{best_f},acc:{best_acc}')
    
    return

    prob0=[]
    prob1 = []
    for pr,l in zip(probs,labels):
        if l>=0.5:
            # print(pr)
            prob1.append(pr)
        else:
            prob0.append(pr)
    
    plt.hist([prob0,prob1],20, range=[0.0,1.0], histtype='barstacked')
    plt.title('baseline')
    plt.legend(['zeros', 'ones'])
    
    plt.savefig('data_fce.png')
    
def predict_self_a_multiple(modelpaths, test_path, ax1):
    criterion = nn.BCELoss().to(device)
    
    for model_path in modelpaths:
        if model_path.endswith('.pt'):
            print('model file')
            probs, labels = predict_w_self_a(test_path, model_path)
        else: # it is a folder
            print('folder')
            probs, labels = predict_w_self_a_folder(test_path, model_path)
        p,r,t = prob2pr(probs, labels)
        auc1 = auc(r,p)
        acc, f = get_accuracy(probs, labels)
        loss = criterion(torch.tensor(probs), torch.tensor(labels))
        
        ece = plot_single_rel_graph(probs, labels)
        best_f, best_acc, _ = get_best_f(p, r, t)
        print(f'ece:{ece},auc:{auc1}, acc:{acc}, f"{f}, loss:{loss}, best f:{best_f}, best_acc:{best_acc}')
    
        # ax1.plot(r*100, p*100)
        
    return probs, labels
    
def main():
    # baseline_file = '/home/zl437/rds/hpc-work/gector/fce-data/with_prob/test_pred4pr.txt'
    
    datapath = ['/home/zl437/rds/hpc-work/gector/CEM/data/fce/dev_emb.npy']
    
    criterion = nn.BCELoss().to(device)
    
    # fig1,ax1 = plt.subplots(1)
    
    # probsb, labelsb, idxb = predict_baseline(datapath)
    # p,r, t = prob2pr(probsb, labelsb)
    
    probsb, labelsb, p, r, t =calibrated_baseline()
    plt.plot(r*100,p*100)
    auc1 = auc(r,p)
    acc1, f1 = get_accuracy(probsb, labelsb)
    loss1 = criterion(torch.tensor(probsb), torch.tensor(labelsb))
    
    ECE1= plot_single_rel_graph(probsb, labelsb)
    best_f, best_acc,_ = get_best_f(p, r, t)
    print(f'baseline ece:{ECE1}, auc:{auc1}, acc:{acc1}, f0.5:{f1},loss:{loss1}, best f:{best_f},acc:{best_acc}')
    
    model_set = '/home/zl437/rds/hpc-work/gector/CEM/model/cm_fce'
    probs, labels, _ = predict_wo_a_folder(datapath, model_set)
    # probs, labels, _ = predict_wo_a(datapath, model_set+'/model4.pt')
    p,r,t = prob2pr(probs, labels)
    plt.plot(r*100,p*100)
    auc1 = auc(r,p)
    acc1, f1 = get_accuracy(probs, labels)
    loss1 = criterion(torch.tensor(probs), torch.tensor(labels))
    ECE1= plot_single_rel_graph(probs, labels)
    
    best_f, best_acc, best_t = get_best_f(p, r, t)
    print(f'best threshold:{best_t}')
    print(f'no attn ece:{ECE1}, auc:{auc1}, acc:{acc1}, f0.5:{f1},loss:{loss1}, best f:{best_f},acc:{best_acc}')
    
    model_set = '/home/zl437/rds/hpc-work/gector/CEM/model/cm_fce_masked'
    probs, labels, _ = predict_wo_a_folder(datapath, model_set)
    # probs, labels, _ = predict_wo_a(datapath, model_set+'/model4.pt')
    p,r,t = prob2pr(probs, labels)
    plt.plot(r*100,p*100)
    
    auc1 = auc(r,p)
    acc1, f1 = get_accuracy(probs, labels)
    loss1 = criterion(torch.tensor(probs), torch.tensor(labels))
    ECE1= plot_single_rel_graph(probs, labels)
    
    best_f, best_acc, best_t = get_best_f(p, r, t)
    print(f'best threshold:{best_t}')
    print(f'no attn masked ece:{ECE1}, auc:{auc1}, acc:{acc1}, f0.5:{f1},loss:{loss1}, best f:{best_f},acc:{best_acc}')

    
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    # acc2, f2 = get_accuracy(probs, labels)
    # ax1.plot(r*100,p*100)
    
    # datapath = ['/home/zl437/rds/hpc-work/gector/CEM/data/fce/masking_self_attn/test.npy']
    datapath = ['/home/zl437/rds/hpc-work/gector/CEM/data/fce/self_attn_dev.npy']
    # datapath = ['/home/zl437/rds/hpc-work/gector/CEM/data/fce/masking_self_attn/train.npy']
    # datapath = ['/home/zl437/rds/hpc-work/gector/CEM/data/bea_nucle/test_selfAttn.npy']
    # probs, labels,_ = predict_wo_a(datapath, confidence_model)
    
    model_paths = ['/home/zl437/rds/hpc-work/gector/CEM/model/cm_fce_masked']
    
    model_folders = ['/home/zl437/rds/hpc-work/gector/CEM/self_attn_model/fce/new_masking']
    model_folders = ['/home/zl437/rds/hpc-work/gector/CEM/self_attn_model/fce_extra_layer_768/new_masking']

    # legends = ['baseline'] + [legends[i] for i in idx]
    
    probs, labels = predict_self_a_multiple(model_folders,datapath, ax1=None)
    p,r,t = prob2pr(probs, labels)
    plt.plot(r*100,p*100)
    plt.legend(['softmax prob', 'CEM pred', 'CEM trained w masking', 'self attn'])
    plt.savefig('PR_curve.png')
    return
    
    # ax1.legend(legends)
    # ax1.set_ylim([50,110])
    # fig1.savefig('pr_curve.png')
 
    prob0=[]
    prob1 = []
    for pr,l in zip(probsb,labelsb):
        if l>=0.5:
            # print(pr)
            prob1.append(pr)
        else:
            prob0.append(pr)
    
    fig2, ax2 = plt.subplots(1)
    # ax[0].hist([prob0,prob1],20, range=[0.0,1.0], histtype='barstacked')
    # ax[0].set_title('baseline')
    # ax[0].legend(['zeros', 'ones'])
    
    
    # probs, labels, idx = predict_wo_a(datapath, confidence_model)
    # p,r,t = prob2pr(probs, labels)
    
    prob0=[]
    prob1 = []
    for pr,l in zip(probs,labels):

        if l>=0.5:
            # print(pr)
            prob1.append(pr)
        else:
            prob0.append(pr)
    
    ax2.hist([prob0,prob1],20, range=[0.0,1.0], histtype='barstacked')
    ax2.set_title('prediction')
    ax2.legend(['zeros', 'ones'])
    
    # fig2.savefig('hist_fce.png')

def get_model_info():
    from cem_w_self_a import ConfidenceModelAttn
    model_w_a = ConfidenceModelAttn().to(device)
    checkpoint = torch.load('/home/zl437/rds/hpc-work/gector/CEM/model/cm_self_attn_fce.pt', map_location=device)
    model_w_a.load_state_dict(checkpoint['model_state_dict'])
    
    loss = checkpoint['loss']
    print(loss)
    print(model_w_a)
    
  
if __name__=='__main__':
    # baseline_only(datapath='/home/zl437/rds/hpc-work/gector/CEM/data/fce/masking_self_attn/test.npy')
    # rels, accs, ECE, nums = calibrated_baseline()
    # print(rels, accs, ECE, nums, sum(nums))
    main()
    # get_model_info()
    # import math
    
    # x1 = np.zeros([1,500],dtype=np.float64)
    # x1[0,0] = 60
    # x1[0,1] = 50
    # x2 = np.ones([1,500],dtype=np.float64)*math.log((math.exp(50)+498)/499)
    # x2[0,0] = 60
    
    # l1 = torch.tensor(x1, dtype=torch.float64)
    # l2 = torch.tensor(x2, dtype=torch.float64)
    # probs1 = F.softmax(l1/1.3,dim=-1, dtype=torch.float64)
    # base1 = F.softmax(l1,dim=-1, dtype=torch.float64)
    # diff1 = torch.max(probs1,dim=-1)[0]-torch.max(base1,dim=-1)[0]
    
    # probs2 = F.softmax(l2/1.3,dim=-1, dtype=torch.float64)
    # base2 = F.softmax(l2,dim=-1, dtype=torch.float64)
    # diff2 = torch.max(probs2,dim=-1)[0]-torch.max(base2,dim=-1)[0]
    
    # # import pdb; pdb.set_trace()
    # print(diff1, diff2)
    