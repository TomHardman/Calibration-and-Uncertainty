import torch
import numpy as np
from sklearn.metrics import precision_recall_curve,fbeta_score,precision_score
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

import os

device = "cuda" if torch.cuda.is_available() else "cpu"

def prob2pr(probs, labels):
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    # prec_rec = list(zip(precisions,recalls, thresholds))
    # prec_rec = list(filter(lambda x: (x[0] != 0.0 and x[0] != 1.0 and x[1] != 0.0 and x[1] != 1.0),prec_rec))
    # precisions = np.array(list(zip(*prec_rec))[0])
    # recalls = np.array(list(zip(*prec_rec))[1])
    # thresholds = np.array(list(zip(*prec_rec))[2])
    
    return precisions, recalls, thresholds

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

def plot_single_rel_graph(preds, label1, filename = 'rel_fce.png'):
    rel1, accs, ECE1 = reliability_graph(preds, label1, n=20)
        
    return ECE1

def get_best_f(precisions, recalls, thresholds):
    prec_rec = list(zip(precisions,recalls, thresholds))
    prec_rec = list(filter(lambda x: (x[0] != 0.0 and x[0] != 1.0 and x[1] != 0.0 and x[1] != 1.0),prec_rec))
    precisions = np.array(list(zip(*prec_rec))[0])
    recalls = np.array(list(zip(*prec_rec))[1])
    thresholds = np.array(list(zip(*prec_rec))[2])
    
    f_scores = 1.25*recalls*precisions/(recalls+0.25*precisions)
    best_t = thresholds[np.argmax(f_scores)]
    best_f = np.max(f_scores)
    best_acc = precisions[np.argmax(f_scores)]
    
    return best_f, best_acc, best_t

def get_accuracy(preds, labels, threshold = 0.5):
    preds = [1 if pred>threshold else 0 for pred in preds]
    
    acc = precision_score(labels, preds)
    Fscore = fbeta_score(labels, preds, beta=0.5)
             
    return acc, Fscore

def temp_anneal_binary_k(probs, labels, temp = np.linspace(1.1,2.0,20)):
    """do temperature annealing on a single probability in a binary classification task
    pick the best results according to BCE loss
    return calibrated probs"""
    criterion = nn.BCELoss().to(device)
    print(temp)
    
    prev_loss = criterion(torch.tensor(probs), torch.tensor(labels))
    
    min_loss = prev_loss
    optimal_t = 1.0
    
    for t in temp:
        cal_probs = []
        for p in probs:
            try:
                l = -np.log((1 / p) - 1)
            except:
                l = -np.log((1 / (p+1e-8)) - 1)
            
            p_new = torch.sigmoid(torch.tensor(l)/t)
            cal_probs.append(p_new)

        new_loss = criterion(
            torch.tensor(cal_probs,dtype=torch.float64), 
            torch.tensor(labels, dtype=torch.float64))
        
        if new_loss < min_loss:
            optimal_t = t
            min_loss = new_loss
        # print(new_loss)
    
    return probs, min_loss, optimal_t

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def accuracy_topk(output, target, k=1):
    """Computes the topk accuracy"""
    _, pred = torch.topk(output, k=k, dim=1, largest=True, sorted=True)

    res_total = 0
    for curr_k in range(k):
      curr_ind = pred[:,curr_k]
      num_eq = torch.eq(curr_ind, target).sum()
      acc = num_eq/len(output)
      res_total += acc
    return res_total*100

def accuracy(output, pred):
    correct = 0
    for o, p in zip(output.tolist(), pred.tolist()):
        if o>0.5 and int(p)==1 or o<=0.5 and int(p)==0:
             correct+=1
    if output.size(0)==0:
        # import pdb; pdb.set_trace()
        return 100
    else:
        return correct/output.size(0)*100