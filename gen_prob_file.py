import argparse

from utils.helpers import read_lines, normalize
from gector.gec_model import GecBERTModel

from utils.preprocess_data import convert_data_from_raw_files, my_get_prob_edits, my_get_keep_prob_edits, my_calibrated_prob, my_predict_sing_stc

from sklearn.metrics import precision_recall_curve,auc
import numpy as np
import random
import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import pandas as pd

from CEM.tools import prob2pr

device = "cuda" if torch.cuda.is_available() else "cpu"

def reliability_graph(probs, labels,n = 20):
    count = [0]*n
    ECE = 0
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
    
    return count, np.linspace(1.0/n/2, 1.0-1.0/n/2, n), ECE/len(probs)

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

def write_for_keep_edit_prob(model, target_path,source_path, output_path):
    # write the 1-keep probability
    [probs, labels, tags] = my_get_keep_prob_edits(target_file=target_path, source_file=source_path, model=model)
    lines = []
    for i in range(len(tags)):
        line = "{} {} {}\n".format(1-probs[i], labels[i], tags[i])
        lines.append(line)

    with open(output_path, 'w') as f:
        f.writelines(lines)
        
def write_for_edit_prob(model, target_path,source_path, output_path):
    [edits, probs, labels, logits_labels, pred_error_probs, labels_for_cem, embeddings] = my_get_prob_edits(target_file=target_path, source_file=source_path, model=model)

   # write for baseline file 
    lines = []
    for i in range(len(probs)):
        line = "{} {} {}\n".format(probs[i], labels[i], 1-logits_labels[i][0])
        lines.append(line)

    with open(output_path, 'w') as f:
        f.writelines(lines)

def prepare_calibration(model, target_path,source_path, output_path):
    [edits, labels, logits, all_probs]=my_calibrated_prob(target_file=target_path, source_file=source_path, model=model)
    
    lines = []
    for e, la, lo, all_p in zip(edits, labels, logits, all_probs):
        if len(e)!=0:
            dic = {'edits':e, 'labels':la, 'logits':lo, 'probs': all_p}
            lines.append(dic)
    np.save(output_path, lines)
        
def calibration(file, temperatures, outputfile):
    datas = np.load(file, allow_pickle=True).tolist()
    print(len(datas))
    
    Temp = temperatures
    criterian = nn.BCELoss().to(device)
    
    losses = []
    for T in Temp:
        probs_cal = []
        probs_base = []
        labels = []
        i=0
        n = 1750
        m=1860
        for data in datas:
            e = data['edits']
            label = data['labels']
            logit = data['logits']
            
            logit = torch.tensor(logit, dtype=torch.float64)
            probs = F.softmax(logit/T,dim=-1, dtype=torch.float64)
            probs_b = F.softmax(logit,dim=-1, dtype=torch.float64)
            
            
            max_vals, idx = torch.max(probs, dim=-1)
            if i<=n and i+len(label)>n:
                l1 = logit[e[n-i]]
            if i<=m and i+len(label)>m:
                l2 = logit[e[m-i]]
            
            i+= len(label)
            # import pdb; pdb.set_trace()
            e_probs = [max_vals[idx].item() for idx in e ]
            label = [l for l in label]
            
            max_vals, _ = torch.max(probs_b, dim=-1)
            b_probs = [max_vals[i].item() for i in e]
            
            probs_cal.extend(e_probs)
            probs_base.extend(b_probs)
            labels.extend(label)
        
        print(len(probs_cal))
        tensor_prob = torch.tensor(probs_cal,dtype=torch.float64)
        tensor_l = torch.tensor(labels,dtype = torch.float64)
        
        # _, idx1 = torch.max(torch.tensor(probs_cal))
        # _, idx2 = torch.max(torch.tensor(probs_base))
        # import pdb; pdb.set_trace()
        loss = criterian(tensor_prob, tensor_l)  
        losses.append(loss)
        
    print(f'losses:{losses}, temps:{temperatures}')
    np.save(outputfile, [probs_cal, labels])
    # import pdb; pdb.set_trace()
        
    return probs_cal, labels

def plot_single_rel_graph(preds, label1, filename = 'rel_fce.png', plot=True):
    precisions, recalls, thresholds = precision_recall_curve(label1, preds)
    plt.plot(recalls, precisions)
    plt.savefig('pr_curve_.png')
    AUC = auc(recalls, precisions)
    
    rel1, accs, ECE1 = reliability_graph(preds, label1, n=20)
    if plot:
        plt.plot(accs, accs,linestyle = '--',color = 'orange')
        print(accs, rel1)
        plt.bar(x=accs, height=rel1, width=accs[1]-accs[0], alpha=0.8)
        plt.ylabel('Accuracy')
        plt.legend('Perfect calibration','Output')

        plt.savefig("/home/zl437/rds/hpc-work/gector/CEM/"+filename)
        
    return ECE1, AUC

def main(args):
    # get all paths
    model = GecBERTModel(vocab_path=args.vocab_path,
                         model_paths=args.model_path,
                         max_len=args.max_len, min_len=args.min_len,
                         iterations=1,
                         min_error_probability=args.min_error_probability,
                         lowercase_tokens=args.lowercase_tokens,
                         model_name=args.transformer_model,
                         special_tokens_fix=args.special_tokens_fix,
                         log=False,
                         confidence=args.additional_confidence,
                         del_confidence=args.additional_del_confidence,
                         is_ensemble=args.is_ensemble,
                         weigths=args.weights)


    # stc = 'Finally, about the free three hours, my personal prefer to go shopping .'
    # my_predict_sing_stc(stc, model)

    target_path =  '/home/zl437/rds/hpc-work/gector/fce-data/dev/fce_target.corr'
    # target_path = '/home/zl437/rds/hpc-work/gector/fce-data/train/target-train.txt'
    # target_path = '/home/zl437/rds/hpc-work/gector/bea-data/nucle/target-test.txt'
    source_path = '/home/zl437/rds/hpc-work/gector/fce-data/dev/fce_test.inc'
    # source_path ='/home/zl437/rds/hpc-work/gector/fce-data/train/source-train.txt'
    # source_path='/home/zl437/rds/hpc-work/gector/bea-data/nucle/source-test.txt'
    output_path = '/home/zl437/rds/hpc-work/gector/fce-data/with_prob/test_logits_dev.npy'
    outputfile = '/home/zl437/rds/hpc-work/gector/fce-data/with_prob/calibrated_baseline_dev.npy'

    # write_for_edit_prob(model, target_path, source_path, output_path)
    prepare_calibration(model, target_path, source_path, output_path)
    probs, labels = calibration(output_path, temperatures=[1.325], outputfile=outputfile) # /1.325
    p,r, t = prob2pr(probs, labels)
    plt.plot(r*100, p*100)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.savefig('temp_annealed_pr.png')
    # probsb, labelsb = calibration(output_path, temperatures=[1.00], output_file=outputfile)
    ece, auc = plot_single_rel_graph(probs, labels, filename='calibrated_baseline.png', plot=False)
    print(ece, auc)
    
    # convert_data_from_raw_files(source_file=args.input_file,target_file=target_path,output_file="",chunk_size=50)

if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help='Path to the model file.', nargs='+',
                        default=['/home/zl437/rds/hpc-work/gector/data/roberta_1_gectorv2.th']
                        )
    parser.add_argument('--vocab_path',
                        help='Path to the model file.',
                        default='/home/zl437/rds/hpc-work/gector/data/output_vocabulary'  # to use pretrained models
                        )
    parser.add_argument('--input_file',
                        help='Path to the evalset file',
                        default='/home/zl437/rds/hpc-work/gector/fce-data/source-test.txt'
                        )
    parser.add_argument('--output_file',
                        help='Path to the output file',
                        default='/home/zl437/rds/hpc-work/gector/bea-data/nucle/with_prob/test_pred_bea_part.txt')
    parser.add_argument('--max_len',
                        type=int,
                        help='The max sentence length'
                             '(all longer will be truncated)',
                        default=50)
    parser.add_argument('--min_len',
                        type=int,
                        help='The minimum sentence length'
                             '(all longer will be returned w/o changes)',
                        default=3)
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of hidden unit cell.',
                        default=128)
    parser.add_argument('--lowercase_tokens',
                        type=int,
                        help='Whether to lowercase tokens.',
                        default=0)
    parser.add_argument('--transformer_model',
                        choices=['bert', 'gpt2', 'transformerxl', 'xlnet', 'distilbert', 'roberta', 'albert'
                                 'bert-large', 'roberta-large', 'xlnet-large'],
                        help='Name of the transformer model.',
                        default='roberta')
    parser.add_argument('--iteration_count',
                        type=int,
                        help='The number of iterations of the model.',
                        default=1)
    parser.add_argument('--additional_confidence',
                        type=float,
                        help='How many probability to add to $KEEP token.',
                        default=0)
    parser.add_argument('--additional_del_confidence',
                        type=float,
                        help='How many probability to add to $DELETE token.',
                        default=0)
    parser.add_argument('--min_error_probability',
                        type=float,
                        help='Minimum probability for each action to apply. '
                             'Also, minimum error probability, as described in the paper.',
                        default=0)
    parser.add_argument('--special_tokens_fix',
                        type=int,
                        help='Whether to fix problem with [CLS], [SEP] tokens tokenization. '
                             'For reproducing reported results it should be 0 for BERT/XLNet and 1 for RoBERTa.',
                        default=1)
    parser.add_argument('--is_ensemble',
                        type=int,
                        help='Whether to do ensembling.',
                        default=0)
    parser.add_argument('--weights',
                        help='Used to calculate weighted average', nargs='+',
                        default=None)
    parser.add_argument('--normalize',
                        help='Use for text simplification.',
                        action='store_true')
    args = parser.parse_args()
    main(args)