from tqdm import tqdm
import argparse
import os
import sys
import subprocess

from utils.preprocess_data import read_parallel_lines, align_sequences, my_equal_edits, check_fn
from align import align_no_file
from gector.gec_model import GecBERTModel, GECBERTModelSpeech
from utils.helpers import read_lines, normalize

import numpy as np
import random
import errant
import pickle
import torch

from pr_confidence import predict_probs_batch, calculate_thresholds

feature_length=768


def calc_entropies_batch(class_probs):
    """
    Calculates entropies given the softmax probability distribution across classes for
    a batch of data.
    
    class_probs should be of size (batch size, sentence length, 5001)
    """
    log_probs = torch.log(class_probs)
    log_probs = torch.where(torch.isinf(log_probs), 0.0, log_probs)
    entropies = torch.sum(-log_probs * class_probs, dim=2)
    return entropies


def get_p_at_index(stc_data, class_probs, i):
    """
    Given a list of predicticted ensemble sentence data dictionaries for a batch, 
    batch class probabilities for the single model and a specific index to for 
    the sentence of interest within the batch, this function returns the probabilities
    of the predicted tags from the ensemble model for the single model case 
    """
    idxs = stc_data[i]['tag_indexes']
    ensemble_probs = []
    assert(len(idxs)==len(class_probs[i]))
    
    for idx, token_dist in zip(idxs, class_probs[i]):
        ensemble_probs.append(token_dist[idx].tolist())
    
    return ensemble_probs


def get_p_of_decision(ensemble_data, source_data, model, batch_size=32, to_normalize=False,
                      temp=1.0, ignore_keep=False, speech=False):
    """
    Creates list of sentence data dictionaries that contain the probability of the tag predicted
    by the ensemble model at the output of the softmax layer for the single model
    """
    pred_data = []
    batch = []
    batch_data = []
    for sent, ensemble_stc_data in zip(source_data, ensemble_data):
        batch.append(sent.split())
        batch_data.append(ensemble_stc_data)
        sequences = model.preprocess(batch)
    
        if len(batch) == batch_size:
                pred_probs, pred_idx, error_probs, keep_probs, class_probs = predict_probs_batch(sequences, batch, model, temp, 
                                                                                                 ignore_keep, return_softmax_dist=True,
                                                                                                 speech=speech)
                for i in range(len(batch)):
                    stc_data = {'original sentence': batch[i],
                                'pred_probs': pred_probs[i],
                                'tag_indexes': pred_idx[i],
                                'prob_ensemble': get_p_at_index(batch_data, class_probs, i)
                                }
                    pred_data.append(stc_data)
                batch = []
                batch_data = []
        
    if batch:
        pred_probs, pred_idx, error_probs, keep_probs, class_probs = predict_probs_batch(sequences, batch, model, temp, 
                                                                                         ignore_keep, return_softmax_dist=True,
                                                                                         speech = speech)
        for i in range(len(batch)):
            stc_data = {'original sentence': batch[i],
                        'pred_probs': pred_probs[i],
                        'tag_indexes': pred_idx[i],
                        'prob_ensemble': get_p_at_index(batch_data, class_probs, i)
                        }
            pred_data.append(stc_data)
    
    return pred_data


def get_file_pred_data(source_data, model, batch_size=32, to_normalize=False, 
                       temp=1.0, ignore_keep=False, entropies=True, speech=False):
    """
    Returns list of dictionaries with each dict containing sentence (in split form), tag indexes, tag probabilities,
    error probability for sentence and keep probabilities for each token
    """
    pred_data = []
    batch = []

    for sent in source_data:
        batch.append(sent.split())
        sequences = model.preprocess(batch)
    
        if len(batch) == batch_size:
                pred_probs, pred_idx, error_probs, keep_probs, class_probs = predict_probs_batch(sequences, batch, model, temp, 
                                                                                                 ignore_keep, return_softmax_dist=True,
                                                                                                 speech=speech)
                for i in range(len(batch)):
                    if entropies:
                        stc_data = {'original sentence': batch[i],
                                    'pred_probs': pred_probs[i],
                                    'tag_indexes': pred_idx[i],
                                    'error_prob': error_probs[i],
                                    'keep_probs': keep_probs[i],
                                    'entropies': calc_entropies_batch(class_probs)[i]
                                    }
                    else:
                        stc_data = {'original sentence': batch[i],
                                    'pred_probs': pred_probs[i],
                                    'tag_indexes': pred_idx[i],
                                    'error_prob': error_probs[i],
                                    'keep_probs': keep_probs[i],
                                    }
                    
                    pred_data.append(stc_data)
                batch = []
        
    if batch:
        pred_probs, pred_idx, error_probs, keep_probs, class_probs = predict_probs_batch(sequences, batch, model, temp, 
                                                                                         ignore_keep, return_softmax_dist=True,
                                                                                         speech=speech)
        for i in range(len(batch)):
            if entropies:
                stc_data = {'original sentence': batch[i],
                            'pred_probs': pred_probs[i],
                            'tag_indexes': pred_idx[i],
                            'error_prob': error_probs[i],
                            'keep_probs': keep_probs[i],
                            'entropies': calc_entropies_batch(class_probs)[i]
                            }
            else:
                stc_data = {'original sentence': batch[i],
                            'pred_probs': pred_probs[i],
                            'tag_indexes': pred_idx[i],
                            'error_prob': error_probs[i],
                            'keep_probs': keep_probs[i],
                            }   
            pred_data.append(stc_data)
    
    return pred_data


def threshold_and_postprocess_batch(batch, model, t, pred_probs, pred_idx, error_probs, keep_probs, uncertainties):
    """
    Apply uncertainty thresholding to batch of sentences and then postprocess to generate predictions
    """
    if t is not None:  # If thresholding is required
        for i in range(len(batch)): # Iterate through sentences in batch
            for j in range(len(pred_probs[i])):  # Iterate through tokens in sentence
                # Determine which tags to keep based on the uncertainty threshold
                # Replace labels with zero and probabilities with the keep probability if confidence above threshold
                if uncertainties[i][j] > t:
                    pred_idx[i][j] = 0
                    pred_probs[i][j] = keep_probs[i][j]
    
    pred_batch = model.postprocess_batch(batch, pred_probs.tolist(),
                                        pred_idx.tolist(), error_probs)
    return pred_batch


def get_average_entropy_file(pred_data):
    """Calculate average in tag class distribution for given data set"""
    entropies = []
    for stc_data in pred_data:
        stc_length = len(stc_data['original sentence'])
        entropies.extend(stc_data['entropies'].tolist()[:stc_length+1])
    return np.mean(entropies)


def get_ensemble_uncertainty_data_jeff(ens_data, b_data, rb_data, xl_data):
    """
    Takes list of sentence data dictionaries for the ensemble and single
    models. Calculates data and knowledge uncertainty for each tag and 
    returns list of dictionaries with metrics for each sentence. Also
    returns sorted arrays for each type of uncertainty - these can be 
    used to calculate the values at which to threshold at.
    """
    uc_data = []
    total_ucs = []
    data_ucs = []
    knowledge_ucs = []
    
    for stc_ens, stc_b, stc_rb, stc_xl in zip(ens_data, b_data, rb_data, xl_data):
        
        total_uc = stc_ens['entropies']
        data_uc = (stc_b['entropies'] + stc_rb['entropies'] + stc_xl['entropies'])/3
        knowledge_uc = total_uc - data_uc

        stc_ens['total_uc'] = total_uc
        stc_ens['data_uc'] = data_uc
        stc_ens['knowledge_uc'] = knowledge_uc

        mask = torch.where(stc_ens['tag_indexes'] != 0)

        total_ucs.extend(total_uc[mask].tolist())
        data_ucs.extend(data_uc[mask].tolist())
        knowledge_ucs.extend(knowledge_uc[mask].tolist())

        uc_data.append(stc_ens)
    
    return uc_data, sorted(total_ucs, reverse=True), sorted(data_ucs, reverse=True), sorted(knowledge_ucs, reverse=True)


def get_ensemble_uncertainty_data(data):
    """
    Takes array of arrays of sentence data dictionaries for the ensemble and 
    single models. Calculates data and knowledge uncertainty for each tag and 
    returns list of dictionaries with metrics for each sentence

    Arguments:
    -------------
    Data - 2D array where each array within the 2D array is an array
    of sentence data dictionaries for each model. The array for the
    ensembled model must be at position zero.
    """
    uc_data = []
    total_ucs = []
    data_ucs = []
    knowledge_ucs = []
    
    for i in range(len(data[0])):

        stc_ens = data[0][i]
        
        total_uc = stc_ens['entropies']   # total uncertainty is uncertainty from ensemble model (entropy of expected)
        data_uc = sum([data[j][i]['entropies'] for j in range(1, len(data))])/(len(data) - 1)  # data uc is expected entropy
        knowledge_uc = total_uc - data_uc

        stc_ens['total_uc'] = total_uc
        stc_ens['data_uc'] = data_uc
        stc_ens['knowledge_uc'] = knowledge_uc

        # when working out distribution of uncertainties for thresholding, only consider those corresponding to non-keep tags
        mask = torch.where(stc_ens['tag_indexes'] != 0) 

        total_ucs.extend(total_uc[mask].tolist())
        data_ucs.extend(data_uc[mask].tolist())
        knowledge_ucs.extend(knowledge_uc[mask].tolist())

        uc_data.append(stc_ens)
    
    return uc_data, sorted(total_ucs, reverse=True), sorted(data_ucs, reverse=True), sorted(knowledge_ucs, reverse=True)


def get_pr(model, pred_data, source_stcs, target_stcs, t_arr, uc_type, dset,
           use_errant=True, log=False):
    #Initialise arrays to store thresholds and corresponds scores
    p_arr = []
    r_arr = []
    f_arr = []
    p0 = 0

    # Setup paths
    pred_path = f'experiments/uncertainty_temp/{dset}_{uc_type}_stc.pred'
    inc_path = f'experiments/uncertainty_temp/{dset}_{uc_type}_stc.inc'
    corr_path = f'experiments/uncertainty_temp/{dset}_{uc_type}_stc.corr'
    hyp_ann_path = f'experiments/uncertainty_temp/{dset}_{uc_type}_pred_ann.m2'
    ref_ann_path = f'experiments/uncertainty_temp/{dset}_{uc_type}_cor_ann.m2'
    
    anno_cmd_pred = ["errant_parallel", "-orig", inc_path, "-cor", pred_path, "-out", hyp_ann_path]
    anno_cmd_tgt = ["errant_parallel", "-orig", inc_path, "-cor", corr_path, "-out", ref_ann_path]
    compare_cmd = ["errant_compare", "-hyp", hyp_ann_path, "-ref", ref_ann_path]
    first_run = True # used to only compute target annotation file once
    
    for t in t_arr:
        print(f'Calculating at t = {t} for {uc_type}')
        pred_stcs = []
        tp = 0
        fp = 0
        fn = 0

        for stc_dict in pred_data:
            # unsqueeze method converts data for a single sentence to batch-wise form by adding extra dimension
            pred_probs = stc_dict['pred_probs'].unsqueeze(0).clone()
            pred_idx = stc_dict['tag_indexes'].unsqueeze(0).clone()
            error_prob = [stc_dict['error_prob']]
            keep_probs = stc_dict['keep_probs'].unsqueeze(0)
            uncertainties = stc_dict[f'{uc_type}'].unsqueeze(0)
            batch = [stc_dict['original sentence']]
            pred_sent = " ".join(threshold_and_postprocess_batch(batch, model, t, pred_probs, pred_idx, error_prob, keep_probs, uncertainties)[0])
            pred_stcs.append(pred_sent)
        
        source_stcs_al, pred_stcs_al, target_stcs_al = align_no_file(source_stcs, pred_stcs, target_stcs)

        with open(pred_path, 'w') as f:  # write predicted sentences to file
            f.write("\n".join(pred_stcs_al) + '\n')
        if first_run:
            with open(inc_path, 'w') as f:
                    f.write("\n".join(source_stcs_al) + '\n')
            with open(corr_path, 'w') as f:
                    f.write("\n".join(target_stcs_al) + '\n')
            subprocess.run(anno_cmd_tgt) # create target annotation file
            first_run = False
        
        subprocess.run(anno_cmd_pred) # create prediction annotation file
        output = subprocess.check_output(compare_cmd, universal_newlines=True) # get output from command line 

        if log:
            print(f't = {t}')
            print(output)
        
        # Parse command line output for precision, recall and f-score
        output_txt = "".join(output)
        lines = output_txt.split('\n')
        stats = lines[3].split('\t')
        p = stats[3]
        r = stats[4]
        f_05 = stats[5]

        if float(p) != 0 and float(p) < p0:
            # If alignment error is suspected 
            if log:
                print('Suspected alignment error - realigning annotation files')
            with open(inc_path, 'w') as f:
                    f.write("\n".join(source_stcs_al) + '\n')
            with open(corr_path, 'w') as f:
                    f.write("\n".join(target_stcs_al) + '\n')
            
            subprocess.run(anno_cmd_tgt) # recreate target annotation file
            subprocess.run(anno_cmd_pred) # create prediction annotation file
            output = subprocess.check_output(compare_cmd, universal_newlines=True)

            # Parse command line output for precision, recall and f-score
            output_txt = "".join(output)
            lines = output_txt.split('\n')
            stats = lines[3].split('\t')
            p = stats[3]
            r = stats[4]
            f_05 = stats[5]

            if log:
                print(f't = {t}')
                print(output)

        if not p_arr:
            # store precision at intial threshold so that it can be used to check for erroneous results
            p0 = float(p)

        # add metrics and thresholds to array
        p_arr.append(p)
        r_arr.append(r)
        f_arr.append(f_05)
    
    return p_arr, r_arr, t_arr, f_arr


def main(args, vis=False, log=True):
    method = args.method[0]
    speech = args.speech
    finetuned = args.finetuned
    combined = args.combined

    # initialise paths to all models and model names
    path_b = '../trained_models/bert_0_gectorv2.th'
    path_rb = '../trained_models/roberta_1_gectorv2.th'
    path_xl = '../trained_models/xlnet_0_gectorv2.th'
    path_b_f = '../trained_models/bert_0_gectorv2finetuned.th'
    path_rb_f = '../trained_models/roberta_1_gectorv2finetuned.th'
    path_xl_f = '../trained_models/xlnet_0_gectorv2finetuned.th'
    
    if combined:
        model_paths = [path_b, path_rb, path_xl, path_b_f, path_rb_f, path_xl_f]
    elif not finetuned:
        model_paths = [path_b, path_rb, path_xl]
    else:
        model_paths = [path_b_f, path_rb_f, path_xl_f]

    if not combined:
        model_names = ['bert', 'roberta', 'xlnet']
        spec_token_fix_params = [0, 1, 0]

    else:
        model_names = ['bert', 'roberta', 'xlnet', 'bert', 'roberta', 'xlnet']
        spec_token_fix_params = [0, 1, 0, 0, 1, 0]

    target_data = read_lines(args.target_file)
    source_data = read_lines(args.input_file)
    dset = args.output_dir.split('/')[-1]  # used for naming temp files to avoid interference while running parallel processes
    model_data = {}

    if log:
        print(f'Test data loaded, {len(source_data)} sentences')

    #generate predictions for ensemble model
    if not speech:
        model_ensemble = GecBERTModel(vocab_path=args.vocab_path,
                                      model_paths=model_paths,
                                      max_len=args.max_len, min_len=args.min_len,
                                      iterations=args.iteration_count,
                                      min_error_probability=args.min_error_probability,
                                      lowercase_tokens=args.lowercase_tokens,
                                      model_name='bert',
                                      special_tokens_fix=0,
                                      log=False,
                                      confidence=args.additional_confidence,
                                      del_confidence=args.additional_del_confidence,
                                      is_ensemble=1,
                                      weigths=args.weights)
    else:
        model_ensemble = GECBERTModelSpeech(vocab_path=args.vocab_path,
                                            model_paths=model_paths,
                                            max_len=args.max_len, min_len=args.min_len,
                                            iterations=args.iteration_count,
                                            min_error_probability=args.min_error_probability,
                                            lowercase_tokens=args.lowercase_tokens,
                                            model_name='bert',
                                            special_tokens_fix=0,
                                            log=False,
                                            confidence=args.additional_confidence,
                                            del_confidence=args.additional_del_confidence,
                                            is_ensemble=1,
                                            weigths=args.weights)
    if log:
        print(f'Generating prediction data for ensemble model')
    
    model_data['ensemble'] = get_file_pred_data(source_data,  model_ensemble, batch_size=args.batch_size, 
                                                to_normalize=False, temp=1.0, ignore_keep=False, speech=speech)

    # iterate through models
    for model_path, model_name, stf in zip(model_paths, model_names, spec_token_fix_params):
        if not speech:
            model = GecBERTModel(vocab_path=args.vocab_path,
                                model_paths=[model_path],
                                max_len=args.max_len, min_len=args.min_len,
                                iterations=args.iteration_count,
                                min_error_probability=args.min_error_probability,
                                lowercase_tokens=args.lowercase_tokens,
                                model_name=model_name,
                                special_tokens_fix=stf,
                                log=False,
                                confidence=args.additional_confidence,
                                del_confidence=args.additional_del_confidence,
                                is_ensemble=0,
                                weigths=args.weights)
        else:
            model = GECBERTModelSpeech(vocab_path=args.vocab_path,
                                     model_paths=[model_path],
                                     max_len=args.max_len, min_len=args.min_len,
                                     iterations=args.iteration_count,
                                     min_error_probability=args.min_error_probability,
                                     lowercase_tokens=args.lowercase_tokens,
                                     model_name=model_name,
                                     special_tokens_fix=stf,
                                     log=False,
                                     confidence=args.additional_confidence,
                                     del_confidence=args.additional_del_confidence,
                                     is_ensemble=0,
                                     weigths=args.weights)

        if combined:
            if model_name in model_data:
                model_name = model_name + '_ft'

        if log:
            print(f'Generating prediction data for {model_name} model')
        
        if method == 1:
            pred_data = get_file_pred_data(source_data,  model, batch_size=args.batch_size, 
                                           to_normalize=False, temp=1.0, ignore_keep=False, speech=speech)
    
            model_data[model_name] = pred_data
        
        elif method == 2:
            decision_probs_model = get_p_of_decision(pred_data_ensemble, source_data, model, batch_size=args.batch_size,
                                                     temp=1.0, ignore_keep=False, speech=speech)
            model_data[model_name] = decision_probs_model
        
        if log:
            print('Prediction data generated')

    if method == 1:
        if log:
            print('Calculating uncertainty data using predictions from all models')
 
        model_data = list(model_data.values())
        uc_data, total_ucs, data_ucs, knowledge_ucs = get_ensemble_uncertainty_data(model_data)

        thresh_arr_tot = calculate_thresholds(total_ucs, args.n_steps)
        thresh_arr_know = calculate_thresholds(knowledge_ucs, args.n_steps)
        thresh_arr_data = calculate_thresholds(data_ucs, args.n_steps)

        uncertainty_types = ['total_uc', 'knowledge_uc', 'data_uc']
        thresh_arrs = [thresh_arr_tot, thresh_arr_know, thresh_arr_data]

        for uc_type, thresh_arr in zip(['data_uc'], [thresh_arr_data]):
            if log:
                print(f'Beginning {uc_type} thresholding')
        
            p, r, t, f = get_pr(model_ensemble, uc_data, source_data, target_data, thresh_arr, uc_type, dset, log=True)
            pr_data = zip(t, p, r, f)
            output_file = args.output_dir + f'/{uc_type}.pkl'

            with open(output_file, 'wb') as f:
                pickle.dump(pr_data, f)
        
    
    if method == 2: # method 2 calculates mean variances for each file
        if log:
            print('Calculating variances')
        var_arr = []
        for stc_ens, stc_b, stc_rb, stc_xl in zip(pred_data_ensemble, model_data['bert'],
                                                  model_data['roberta'], model_data['xlnet']):
            for i in range(min([51, len(stc_ens['original sentence']) + 1])):
                probs = [stc_b['prob_ensemble'][i], stc_rb['prob_ensemble'][i], stc_xl['prob_ensemble'][i]]
                var_arr.append(np.var(probs))
        
        print(f'Mean Variance: {np.mean(var_arr)}')
                

if __name__ == '__main__':
     # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--method',
                        help='Method for uncertainty estimation', nargs='+',
                        type=int,
                        default=1)
    parser.add_argument('--vocab_path',
                        help='Path to the model file.',
                        default='data/output_vocabulary'  # to use pretrained models
                        )
    parser.add_argument('--input_file',
                        help='Path to the evalset file',
                        required=True)
    parser.add_argument('--target_file',
                        help='Path to target sentences',
                        required=True)
    parser.add_argument('--output_dir',
                        help='Path to output directory',
                        required=True)
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
                        default=32)
    parser.add_argument('--lowercase_tokens',
                        type=int,
                        help='Whether to lowercase tokens.',
                        default=0)
    parser.add_argument('--iteration_count',
                        type=int,
                        help='The number of iterations of the model.',
                        default=5)
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
                        default=0.0)
    parser.add_argument('--weights',
                        help='Used to calculate weighted average', nargs='+',
                        default=None)
    parser.add_argument('--normalize',
                        help='Use for text simplification.',
                        action='store_true')
    parser.add_argument('--speech',
                         help='Use model amended for speech data',
                         type=int,
                         default=0)
    parser.add_argument('--n_steps',
                         help='Number of steps to threshold at',
                         type=int,
                         default=100)
    parser.add_argument('--finetuned',
                         help='Whether to use finetuned models',
                         type=int,
                         default=0)
    parser.add_argument('--combined',
                         help='Whether to combine using finetuned and original models',
                         type=int,
                         default=0)
    args = parser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/uncertainty.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    main(args)