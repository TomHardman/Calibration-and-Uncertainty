import torch
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd

from cem_model import SpeechDataGenerator
from cem_w_self_a import SpeechDataGenerator as selfAttnDataGenerator
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import auc
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
max_length = 128
label_length = 5002

def predict_wo_a(datapaths, modelpath, batch_size = 1):
    from cem_model import ConfidenceModel
    model = ConfidenceModel().to(device)
    checkpoint = torch.load(modelpath,map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

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
            pred = model(features)
            probs.append(pred[0].item())
            labels.append(tgt[0].item())
            idxes.append(idx[0].item())
            
    return probs, labels, idxes

def predict_baseline(datapaths, batch_size = 1, datagenerator = SpeechDataGenerator, dataloader = DataLoader):
    probs = []
    labels = []
    idxes = []
    
    dataset = SpeechDataGenerator(datapaths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(dataset.__len__())
    
    with torch.no_grad():
        for features, tgt, prob, idx in dataloader:
            # Compute prediction and loss
            probs.append(prob[0].item())
            labels.append(tgt[0].item())
            idxes.append(idx[0].item())             
            
    return probs, labels, idxes

def predict_w_a(test_file, model_w_a, batch_size = 1):
    dataset = SpeechDataGenerator(test_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    probs = []
    labels = []

    with torch.no_grad():
        for batch, [e_i, v_i, v_last, v_all, tgt] in enumerate(dataloader):
            # Compute prediction and loss
            pred = model_w_a.forward(v_i, v_last, e_i, v_all)
            probs.append(pred[0].item())
            labels.append(tgt[0][0].item())
        
    return probs, labels

def predict_w_self_a(datapath, model_path, batch_size = 100):
    from cem_window import ConfidenceModelAttn
    from cem_window import collate_wrapper
    
    model_w_self_a = ConfidenceModelAttn().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model_w_self_a.load_state_dict(checkpoint['model_state_dict'])
    loss = checkpoint['loss']
    # print(loss)
    
    dataset = selfAttnDataGenerator(datapath)
    dataloader1 = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_wrapper)
    dataloader2 = DataLoader(dataset, batch_size=2*batch_size, shuffle=False, collate_fn=collate_wrapper)
    
    probs = []
    labels = []

    with torch.no_grad():
        for batch, data in enumerate(dataloader1):
            # Compute prediction and loss
            mask = data.op_mask
            pred = model_w_self_a(data.inp, data.mask, data.prob)
            
            tgt = data.tgt
            pred = torch.masked_select(pred, mask)
            tgt = torch.masked_select(tgt, mask)
          
            # pred = [pred[i].item() for i in range(len(mask)) if mask[i]]
            # tgt = [tgt[i].item() for i in range(len(mask)) if mask[i]]

            probs.extend(pred.tolist())
            labels.extend(tgt.tolist())
    
    return probs, labels

def reliability_graph(probs, labels,n = 20):
    count = [0]*n
    ECE = 0
    nums = [0]*n
    for i in range(n):
        left = 1.0/n*i
        right = 1.0/n*(i+1)
        mid = 1.0/n*(i+0.5)
        # import pdb; pdb.set_trace()
        
        # correct = sum([l for p,l in zip(probs, labels) if p>left and p <=right])
        correct = sum(map(lambda p,l: p>left and p <=right and l >= 0.5, probs, labels))
        # incor = sum([1-l for p,l in zip(probs, labels) if p>left and p <=right])
        incor = sum(map(lambda p,l: p>left and p <=right and l <= 0.5, probs, labels))
        
        if correct+incor !=0:
            accuracy = correct/(correct+incor)
            ECE += abs(accuracy - mid)*(correct+incor)
        else:
            accuracy = 0
        
        count[i] = accuracy
        nums[i]=correct+incor
    
    return count, np.linspace(1.0/n/2, 1.0-1.0/n/2, n), ECE/len(probs)

def prob2pr(probs, labels):
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    prec_rec = list(zip(precisions,recalls, thresholds))
    # prec_rec = list(filter(lambda x: (x[0] != 0.0 and x[0] != 1.0 and x[1] != 0.0 and x[1] != 1.0),prec_rec))
    precisions = np.array(list(zip(*prec_rec))[0])
    recalls = np.array(list(zip(*prec_rec))[1])
    thresholds = np.array(list(zip(*prec_rec))[2])
    
    return precisions, recalls, thresholds

def draw_pr_curve(filename,col = 0):
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

    precisions, recalls, thresholds = precision_recall_curve(labels, probs)

    prec_rec = list(zip(precisions,recalls, thresholds))
    prec_rec = list(filter(lambda x: (x[0] != 0.0 and x[0] != 1.0 and x[1] != 0.0 and x[1] != 1.0),prec_rec))
    precisions = np.array(list(zip(*prec_rec))[0])
    recalls = np.array(list(zip(*prec_rec))[1])
    thresholds = np.array(list(zip(*prec_rec))[2])

    return precisions, recalls, thresholds, probs, labels

def bceLoss(probs, labels):
    criterion = nn.BCELoss().to(device)
    
    probs = torch.from_numpy(np.asarray(probs, dtype=np.float32)).detach().to(device)
    labels = torch.from_numpy(np.asarray(labels, dtype=np.float32)).detach().to(device)
    baseline_loss = criterion(probs, labels).item()
    
    return baseline_loss

if __name__ == '__main__':
    from cem_w_self_a import ConfidenceModelAttn
    model_w_a = ConfidenceModelAttn().to(device)
    checkpoint = torch.load('/home/zl437/rds/hpc-work/gector/CEM/model/cm_self_attn_fce.pt', map_location=device)
    model_w_a.load_state_dict(checkpoint['model_state_dict'])
    
    from cem_model import ConfidenceModel
    confidence_model = ConfidenceModel().to(device)
    checkpoint = torch.load('/home/zl437/rds/hpc-work/gector/CEM/model/cm_fce.pt',map_location=device)
    confidence_model.load_state_dict(checkpoint['model_state_dict'])
    
    # fig, axs = plt.subplots(2)
    # fig.tight_layout(pad=3.0)
  
    filename_fce_baseline = '/home/zl437/rds/hpc-work/gector/fce-data/with_prob/test_pred_class_fce.txt'
    filename_nucle_baseline = '/home/zl437/rds/hpc-work/gector/bea-data/nucle/with_prob/test_pred_bea.txt'
    # precisions, recalls, thresholds,probs, labels = draw_pr_curve(filename_nucle_baseline)
    # probs_0 = [probs[i] for i in range(len(labels)) if labels[i]>0.5]
    # probs_1 = [probs[i] for i in range(len(labels)) if labels[i]<0.5]    
    
    # rel1, accs = reliability_graph(probs, labels)
    # plt.plot(recalls*100,precisions*100)
    # area1 = auc(recalls, precisions)
    criterian = nn.BCELoss().to(device)
    datapath = ['/home/zl437/rds/hpc-work/gector/CEM/data/fce/test_emb.npy']
    # datapath = ['/home/zl437/rds/hpc-work/gector/CEM/data/bea_nucle/test_emb.npy']
    # datapath = ['/home/zl437/rds/hpc-work/gector/CEM/data/bea_nucle/test_emb.npy']
    preds, labels, _ = predict_baseline(datapath, confidence_model)
    p,r,t = prob2pr(preds, labels)
    plt.plot(r*100,p*100)
    area1 = auc(r, p)
    loss1 = criterian(torch.tensor(preds), torch.tensor(labels))

    # preds_0 = [preds[i] for i in range(len(labels)) if labels[i]>0.5]
    # preds_1 = [preds[i] for i in range(len(labels)) if labels[i]<0.5]
    # axs[0].hist([probs_0,preds_0],20)
    # axs[0].set_title('true label = 0')
    # axs[0].legend(['baseline','prediction'])
    # axs[1].hist([probs_1,preds_1],20)
    # axs[1].set_title('true label = 1')
    # axs[1].legend(['baseline','prediction'])


    
    # datapath = ['/home/zl437/rds/hpc-work/gector/CEM/data/bea_nucle/test_emb.npy']
    preds, labels,_ = predict_wo_a(datapath, confidence_model)
    loss2 = criterian(torch.tensor(preds), torch.tensor(labels))
    
    # preds, labels,_ = predict_wo_a(datapath, confidence_model)
    p,r,t = prob2pr(preds, labels)
    plt.plot(r*100,p*100)
    area2 = auc(r, p)
    print(f'baseline auc:{area1}, cem auc:{area2} \n baseline loss:{loss1}, cem:{loss2}')
    datapath = ['/home/zl437/rds/hpc-work/gector/CEM/data/fce/self_attn_test.npy']
    # datapath = ['/home/zl437/rds/hpc-work/gector/CEM/data/bea_nucle/train_selfAttn.npy']
    probs, labels = predict_w_self_a(datapath, model_w_a)
    p,r, _ = precision_recall_curve(labels, probs)
    plt.plot(r*100,p*100)
    
    plt.xlabel('Recall')
    plt.ylim((50,110))
    plt.ylabel('Precision')
    plt.title("PR curve")
    plt.legend(['baseline', 'no attention', 'with attention'])

    plt.savefig("/home/zl437/rds/hpc-work/gector/CEM/hist_fce.png")
    print("saved")
