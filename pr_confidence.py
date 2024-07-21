"""
This file is used for generating pr data for a given model - examples in gector_cmds.txt
The program generates predictions and annotation files as it runs in experiments/ct folder -
the naming of these files is based on the model_name parameter passed to the program when
it is run. If you are trying to generate predictions for multiple models in parallel, 
make sure you change the model_name parameter for each process in order to stop the parallel 
processes from interfering with each other

It generates outputs as a zipped structure containing threshold, precision, recall, f-score,
which it saves in a pickle file
"""

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
from numpy.random import default_rng
import Levenshtein
import errant
import pickle

import torch
import torch.nn.functional as F

feature_length=768


def softmax_with_temperature_batch(logits_tensor, temperature=1.0, ignore_keep=False):
    if temperature != 1.0:
        # Apply temperature annealing
        scaled_logits = logits_tensor / temperature
        softmax_output = F.softmax(scaled_logits, dim=-1)
    else:
        softmax_output = F.softmax(logits_tensor, dim=-1)

    keep_probs = softmax_output[:, :, 0].clone()  # Save the probabilities at index 0

    if ignore_keep:
        # Set probability at index zero (keep tag) to zero
        softmax_output[:, :, 0] = 0

    return softmax_output, keep_probs


def calculate_thresholds(vals, n_steps):
    """
    Given an array of data, returns an array that represents the values to be thresholded at
    based on the idea that each thresholding increment should affect an equal number of tokens
    """
    indexes = np.linspace(0, len(vals)-2, n_steps).astype(int)
    thresholds = [vals[idx] for idx in indexes]
    return thresholds

       
def predict_probs_batch(sequences, batch, model, temp, ignore_keep, return_softmax_dist=False, speech=False):
    """
    Predicts tag probabilities for a batch of sequences
    """
    preds, idx, error_probs, logit_labels = model.predict(sequences, return_logits=True)
    annealed_probs = torch.zeros_like(logit_labels[0])
    keep_probs = torch.zeros_like(torch.tensor(preds)).cuda()

    for n in range(len(model.model_weights)): # iterate through models to ensemble class probabilties
        #  extract logits for each model and put through softmax layer with annealing
        logits_model = logit_labels[n]

        annealed_probs_sm, keep_probs_sm = softmax_with_temperature_batch(
                                            logits_model, temperature=temp, 
                                            ignore_keep=ignore_keep)
        
        annealed_probs += annealed_probs_sm * model.model_weights[n] / sum(model.model_weights)
        keep_probs += keep_probs_sm * model.model_weights[n] / sum(model.model_weights)
    
    if speech:
        annealed_probs = model.remove_punctuation_edits(annealed_probs)

    pred_probs, pred_idx = torch.max(annealed_probs, dim=-1)  # get max tag probabilities and corresponding tag indices

    if not return_softmax_dist:
        return pred_probs, pred_idx, error_probs, keep_probs
    
    else:
        return pred_probs, pred_idx, error_probs, keep_probs, annealed_probs


def threshold_and_postprocess_batch(batch, model, t, pred_probs, pred_idx, error_probs, keep_probs, asr_probs=None, asr_labels=None,
                                    threshold_mode = 'asr_only', threshold_const=0, alpha=1, beta=1, stc_correct=False):
    if t is not None:  # If thresholding is required
        for i in range(len(batch)): # Iterate through sentences in batch
            for j in range(len(pred_probs[i])):  # Iterate through tokens in sentence
                # Determine which tags to keep based on the confidence threshold
                # Replace labels with zero and probabilities with the keep probability if confidence below threshold
                if not asr_probs:
                    if pred_probs[i][j] < t:
                        pred_idx[i][j] = 0
                        pred_probs[i][j] = keep_probs[i][j]
                elif threshold_mode == 'asr_only':
                    if pred_probs[i][j] < threshold_const or asr_probs[i][j] < t:
                        pred_idx[i][j] = 0
                        pred_probs[i][j] = keep_probs[i][j]
                elif threshold_mode == 'gec_only':
                    if pred_probs[i][j] < t or asr_probs[i][j] < threshold_const:
                        pred_idx[i][j] = 0
                        pred_probs[i][j] = keep_probs[i][j]
                elif threshold_mode == 'weighted_product':
                    if asr_probs[i][j] ** alpha * pred_probs[i][j] ** beta < t:
                        pred_idx[i][j] = 0
                        pred_probs[i][j] = keep_probs[i][j]
                elif threshold_mode == 'upper_bound':
                    if pred_probs[i][j] < threshold_const or asr_labels[i][j] < t:
                        pred_idx[i][j] = 0
                        pred_probs[i][j] = keep_probs[i][j]
            if threshold_mode == 'upper_bound_stc': # if asr error in sentence replace entire sentence with keep tags
                if not stc_correct[i]:
                    pred_idx[i] = torch.Tensor([0 for k in range(len(pred_idx[i]))])
                    pred_probs[i] = torch.Tensor([keep_probs[i][k] for k in range(len(keep_probs[i]))])

    pred_batch = model.postprocess_batch(batch, pred_probs.tolist(),
                                        pred_idx.tolist(), error_probs)
    return pred_batch


def predict_file_with_threshold(test_data, model, t, batch_size=1, to_normalize=False, 
                           temp=1.0, ignore_keep=True, log=False, speech=False):
    """
    Predict output sentences for an entire file, with option for thresholding and temperature
    annealing
    """
    batch = []
    predictions = []
    probs_file = []
    idx_file = []
    k = 0

    for sent in test_data:
        batch.append(sent.split())
        sequences = model.preprocess(batch)
        
        if len(batch) == batch_size:
            print(f'batch{k} preds_start')
            pred_probs, pred_idx, error_probs, keep_probs = predict_probs_batch(sequences, batch, model, 
                                                                                temp, ignore_keep, speech=speech)

            print(f'batch{k} preds done')
            pred_batch = threshold_and_postprocess_batch(batch, model, t, pred_probs, 
                                                         pred_idx, error_probs, keep_probs)
            predictions.extend(pred_batch)
            probs_file.extend(pred_probs)
            idx_file.extend(pred_idx)
            batch = []
            k += 1

    pred_probs, pred_idx, error_probs, keep_probs = predict_probs_batch(sequences, batch, model, temp, ignore_keep, speech=speech)
    pred_batch = threshold_and_postprocess_batch(batch, model, t, pred_probs, pred_idx, error_probs, keep_probs)
    predictions.extend(pred_batch)
    probs_file.extend(pred_probs)
    idx_file.extend(pred_idx)

    predictions = [" ".join(stc) for stc in predictions]
    
    return predictions, probs_file, idx_file
        

def get_file_pred_data(source_data, output_file, model, batch_size=32, to_normalize=False, 
                       temp=1.0, ignore_keep=True, log=True, speech=False):
    """
    Returns list of dictionaries with each dict containing sentence (in split form), tag indexes, tag probabilities,
    error probability for sentence and keep probabilities for each token
    """
    pred_data = []
    batch = []
    k = 0
    for sent in source_data:
        batch.append(sent.split())
        sequences = model.preprocess(batch)
    
        if len(batch) == batch_size:
                pred_probs, pred_idx, error_probs, keep_probs = predict_probs_batch(sequences, batch, 
                                                                                    model, temp, ignore_keep, speech=speech)
                for i in range(len(batch)):
                    stc_data = {'original sentence': batch[i],
                                'pred_probs': pred_probs[i],
                                'tag_indexes': pred_idx[i],
                                'error_prob': error_probs[i],
                                'keep_probs': keep_probs[i],
                                }
                    pred_data.append(stc_data)
                k += 1
                batch = []
        
    if batch:
        pred_probs, pred_idx, error_probs, keep_probs = predict_probs_batch(sequences, batch, 
                                                                                    model, temp, ignore_keep, speech=speech)
        for i in range(len(batch)):
            stc_data = {'original sentence': batch[i],
                        'pred_probs': pred_probs[i],
                        'tag_indexes': pred_idx[i],
                        'error_prob': error_probs[i],
                        'keep_probs': keep_probs[i],
                            }
        pred_data.append(stc_data)
    
    return pred_data


def get_conf_arr_from_pred_data(pred_data):
    """
    Given array of stc data dictionaries containing data for each sentence, 
    returns array of confidences associated with non-KEEP tags to be used
    for setting the thresholds.
    """
    c_arr = []
    for stc_data in pred_data:
        pred_probs = stc_data['pred_probs']
        mask = torch.where(stc_data['tag_indexes'] != 0)
        c_arr.extend(pred_probs[mask].tolist())
    
    return sorted(c_arr)


def get_conf_arr_asr(asr_data, pred_data, prod=False, alpha=1, beta=1):
    """
    Takes data formatted as list of lists where each inner list represents a single
    sentence and is of form [(word1, asr  word error prob), (word2, asr word error prob) ...]
    Returns sorted list of word confidences to be used for setting the thresholds.
    """
    c_arr = []
    for stc_data_asr, stc_pred in zip(asr_data, pred_data):
        asr_correctness_probs = [1 - word[1] for word in stc_data_asr]
        mask = torch.where(stc_pred['tag_indexes'][:len(asr_correctness_probs)] != 0)
        if not prod:
            c_arr.extend(np.array(asr_correctness_probs)[mask[0].tolist()])
        else:
            prod_arr = np.array(asr_correctness_probs[:len(stc_pred['pred_probs'])])**alpha * np.array(stc_pred['pred_probs'].tolist()[:len(asr_correctness_probs)])**beta
            c_arr.extend(prod_arr[mask[0].tolist()])

    
    return sorted(c_arr)


def get_stc_edits(annotator, src, tgt):
    src = annotator.parse(src)
    tgt = annotator.parse(tgt)
    edits = annotator.annotate(src, tgt)
    return edits


def equal_edits(edit_hyp, edit_ref, asr_shifts=None):
    if asr_shifts is None:
        asr_shifts = np.zeros((2, 50))
    
    if all([edit_hyp.o_start + asr_shifts[0][edit_hyp.o_start] == edit_ref.o_start, 
            edit_hyp.o_end + asr_shifts[1][edit_hyp.o_end] == edit_ref.o_end, 
            edit_hyp.c_str == edit_ref.c_str,
            edit_hyp.o_str == edit_ref.o_str]):
        return True
    return False


def get_pr_conf(args, model, pred_data, source_stcs, target_stcs, t_arr, model_name,
                log=False, asr_data=None, corr_inp_stcs=None):
    #Initialise arrays to store thresholds and corresponding scores
    p_arr = []
    r_arr = []
    f_arr = []
    p0 = 0

    # Setup paths
    pred_path = f'experiments/ct/{model_name}_stc.pred'
    inc_path = f'experiments/ct/{model_name}_stc.inc'
    corr_path = f'experiments/ct/{model_name}_stc.corr'
    corr_inp_path = f'experiments/ct/{model_name}_stc.corr_inp'
    hyp_ann_path = f'experiments/ct/{model_name}_pred_ann.m2'
    ref_ann_path = f'experiments/ct/{model_name}_cor_ann.m2'
    
    # Setup cmds
    anno_cmd_pred = ["errant_parallel", "-orig", inc_path, "-cor", pred_path, "-out", hyp_ann_path]
    if not args.spoken_gec:
        anno_cmd_tgt = ["errant_parallel", "-orig", inc_path, "-cor", corr_path, "-out", ref_ann_path]
    else:
        anno_cmd_tgt = ["errant_parallel", "-orig", corr_inp_path, "-cor", corr_path, "-out", ref_ann_path]
    compare_cmd = ["errant_compare", "-hyp", hyp_ann_path, "-ref", ref_ann_path]
    first_run = True # used to only compute target annotation file once
    
    for t in t_arr:
        print(f'Calculating at t = {t}')
        pred_stcs = []
        tp = 0
        fp = 0
        fn = 0
        offset = 0

        for i, stc_dict in enumerate(pred_data):
            # unsqueeze method converts data for a single sentence to batch-wise form by adding extra dim
            pred_probs = stc_dict['pred_probs'].unsqueeze(0)
            pred_idx = stc_dict['tag_indexes'].unsqueeze(0)
            error_prob = [stc_dict['error_prob']]
            keep_probs = stc_dict['keep_probs'].unsqueeze(0)
            batch = [stc_dict['original sentence']]

            if args.spoken_gec:
                words = [pair[0] for pair in asr_data[i + offset]]
                
                while words[0] != batch[0][0]:
                    offset += 1
                    words = [pair[0] for pair in asr_data[i + offset]]
                
                stc_asr_confs = [1] + [1 - conf[1] for conf in asr_data[i + offset]]
                stc_asr_labels = [1] + [int(abs(label[2] - 1)) for label in asr_data[i + offset]]

                assert(len(batch[0]) + 1 == len(stc_asr_confs))
                
                stc_asr_confs.extend([1 for _ in range(len(pred_probs[0]) - len(stc_asr_confs))])
                stc_asr_labels.extend([1 for _ in range(len(pred_probs[0]) - len(stc_asr_labels))])
                stc_asr_confs = [stc_asr_confs]
                stc_asr_labels = [stc_asr_labels]
                stc_correctness = [batch[0] == corr_inp_stcs[i + offset].split(' ')]
      
            else:
                stc_asr_confs = stc_asr_labels = stc_correctness =  None

            pred_sent = " ".join(threshold_and_postprocess_batch(batch, model, t, pred_probs, pred_idx, error_prob, keep_probs,
                                                                 stc_asr_confs, stc_asr_labels, threshold_mode=args.spoken_gec_threshold_mode,
                                                                 threshold_const=args.threshold_constant, alpha=args.alpha,
                                                                 beta=args.beta, stc_correct=stc_correctness)[0])
            pred_stcs.append(pred_sent)
        
        if args.spoken_gec:
            source_stcs_al, pred_stcs_al, target_stcs_al, corr_inp_stcs_al = align_no_file(source_stcs, pred_stcs, target_stcs, 
                                                                                           trans_inp_sent=corr_inp_stcs, spoken_gec=True)
            if not len(source_stcs_al) == len(pred_stcs_al) == len(target_stcs_al) == len(corr_inp_stcs_al):
                print('Unexplained alignment error - skipping current iteration')
                continue
        else:
            source_stcs_al, pred_stcs_al, target_stcs_al = align_no_file(source_stcs, pred_stcs, target_stcs)

        if not args.errant_mod:
            with open(pred_path, 'w') as f:  # write predicted sentences to file
                f.write("\n".join(pred_stcs_al) + '\n')
            if first_run:
                with open(inc_path, 'w') as f:
                        f.write("\n".join(source_stcs_al) + '\n')
                with open(corr_path, 'w') as f:
                        f.write("\n".join(target_stcs_al) + '\n')
                if args.spoken_gec:
                    with open(corr_inp_path, 'w') as f:
                        f.write("\n".join(corr_inp_stcs_al) + '\n')
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
                if args.spoken_gec:
                    with open(corr_inp_path, 'w') as f:
                        f.write("\n".join(corr_inp_stcs_al) + '\n')
                
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

        
        elif args.errant_mod:
            print('Using modified errant')
            tp = 0
            fp = 0
            ref_tot = 0
            annotator = errant.load('en')
            
            if t == 0.0:
                correct_edits_baseline = set()
                correct_edits_ub = set()
            
            for i, (stc_asr, stc_pred, stc_tgt, stc_trans) in enumerate(zip(source_stcs_al, pred_stcs_al, target_stcs_al, corr_inp_stcs_al)):
                hyp_edits = get_stc_edits(annotator, stc_asr, stc_pred)
                ref_edits = get_stc_edits(annotator, stc_trans, stc_tgt)
                ref_tot += len(ref_edits)

                # find operations to get from transcription to asr such that we can manually adjust spans of edits to match
                asr_editops = Levenshtein.opcodes(stc_asr.split(' '), stc_trans.split(' '))
                asr_shifts = np.zeros((2, len(stc_asr.split(' ')) + 1))
                ins_del = False
                
                for opcode, asr_start, asr_stop, trans_start, trans_stop in asr_editops:
                    if opcode == 'insert':
                        asr_shifts[0, asr_start] += trans_stop - trans_start
                        asr_shifts[1, asr_start+1] += trans_stop - trans_start
                        ins_del = True
                    elif opcode == 'delete':
                        asr_shifts[:, asr_stop] += - (asr_stop - asr_start)
                        ins_del = True
                
                asr_shifts = np.cumsum(np.array(asr_shifts), axis=1)

                if ins_del and hyp_edits and ref_edits:
                    a = 10
                    a += 1
         
                for hyp_edit in hyp_edits:
                    correct = False
                    for ref_edit in ref_edits:
                        if equal_edits(hyp_edit, ref_edit, asr_shifts=asr_shifts):
                            if args.spoken_gec_threshold_mode == 'upper_bound':
                                if t == 0.0:
                                    correct_edits_baseline.add((i, hyp_edit.o_str, hyp_edit.c_str, hyp_edit.type))
                                else:
                                    correct_edits_ub.add((i, hyp_edit.o_str, hyp_edit.c_str, hyp_edit.type))
                            tp += 1
                            correct = True
                            ref_edits.remove(ref_edit)
                            break
                    if not correct:
                        fp += 1
            
            if correct_edits_baseline and correct_edits_ub:
                new_edits = set()
                for edit in correct_edits_ub:
                    if edit in correct_edits_baseline:
                        correct_edits_baseline.remove(edit)
                    else:
                        new_edits.add(edit)

        try:  
            fn = ref_tot - tp
            p = tp/(tp + fp)
            r = tp/(tp + fn)
            f_05 = 1.25 * (p * r) / (0.25 * p + r)
            print(f'TP {tp}:, FP:{fp}, FN:{fn}, P: {p}, R: {r}, F0.5: {f_05}')
        
        except ZeroDivisionError:
            p = p_arr[-1]
            r = r_arr[-1]
            f = f_arr[-1]


        # add metrics and thresholds to array
        p_arr.append(p)
        r_arr.append(r)
        f_arr.append(f_05)

    return p_arr, r_arr, t_arr, f_arr


def main(args, vis=False, log=True):
    speech = args.speech
    n_steps = args.n_steps

    if not speech:
        model = GecBERTModel(vocab_path=args.vocab_path,
                            model_paths=args.model_path,
                            max_len=args.max_len, min_len=args.min_len,
                            iterations=args.iteration_count,
                            min_error_probability=args.min_error_probability,
                            lowercase_tokens=args.lowercase_tokens,
                            model_name=args.transformer_model,
                            special_tokens_fix=args.special_tokens_fix,
                            log=False,
                            confidence=args.additional_confidence,
                            del_confidence=args.additional_del_confidence,
                            is_ensemble=args.is_ensemble,
                            weigths=args.weights)
    
    if speech:
        model = GECBERTModelSpeech(vocab_path=args.vocab_path,
                                model_paths=args.model_path,
                                max_len=args.max_len, min_len=args.min_len,
                                iterations=args.iteration_count,
                                min_error_probability=args.min_error_probability,
                                lowercase_tokens=args.lowercase_tokens,
                                model_name=args.transformer_model,
                                special_tokens_fix=args.special_tokens_fix,
                                log=False,
                                confidence=args.additional_confidence,
                                del_confidence=args.additional_del_confidence,
                                is_ensemble=args.is_ensemble,
                                weigths=args.weights)
    
    target_data = read_lines(args.target_file)
    source_data = read_lines(args.input_file)

    if args.spoken_gec:
        with open(args.asr_data_file, 'rb') as f:
            asr_data = pickle.load(f)
        corr_source_data = read_lines(args.corr_input_file)
    else:
        asr_data = None
        corr_source_data = None
    
    if log:
        print(f'Test data loaded, {len(source_data)} sentences')
        print('Generating prediction data')
    pred_data = get_file_pred_data(source_data, args.output_file, model, speech=speech,
                                   batch_size=args.batch_size, to_normalize=False, temp=1.0, ignore_keep=False)
    if log:
        print('Prediction data generated - thresholding')
    
    if not args.spoken_gec or (args.spoken_gec_threshold_mode == 'gec_only' and args.spoken_gec):
        conf_arr = get_conf_arr_from_pred_data(pred_data)
    elif args.spoken_gec_threshold_mode == 'asr_only' or 'upper_bound' or 'upper_bound_stc':
        conf_arr = get_conf_arr_asr(asr_data, pred_data)
    elif args.spoken_gec_threshold_mode == 'weighted_product':
        conf_arr = get_conf_arr_asr(asr_data, pred_data, alpha=args.alpha, beta=args.beta, prod=True)
    thresh_arr = calculate_thresholds(conf_arr, n_steps)
    thresh_arr = [0.0] + thresh_arr

    p, r, t, f = get_pr_conf(args, model, pred_data, source_data, target_data, t_arr=thresh_arr, 
                             model_name=args.model_name, log=True, asr_data=asr_data, corr_inp_stcs=corr_source_data)

    pr_data = zip(t, p, r, f)
    
    with open(args.output_file, 'wb') as f:
        pickle.dump(pr_data, f)


if __name__ == '__main__':
     # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help='Path to the model file.', nargs='+',
                        default=['/scratches/dialfs/alta/th624/exp-th624/trained_models/xlnet_0_gectorv2finetuned.th'])
    parser.add_argument('--vocab_path',
                        help='Path to the vocab file.',
                        default='data/output_vocabulary'  # to use pretrained models
                        )
    parser.add_argument('--input_file',
                        help='Path to the evalset file',
                        default='../data/Linguaskill/preprocessed_whisperflt/ls_test.inc')
    parser.add_argument('--output_file',
                        help='Path to the output file',
                        default='experiments/pr_data/confidence/Linguaskill_whisperflt/xlnet_ft.pkl')
    parser.add_argument('--target_file',
                        help='Path to target sentences',
                        default='../data/Linguaskill/preprocessed_whisperflt/ls_test.corr')
    parser.add_argument('--model_name',
                        type=str,
                        help='Name of model',
                        default='asr_xlnet')
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
    parser.add_argument('--transformer_model',
                        choices=['bert', 'gpt2', 'transformerxl', 'xlnet', 'distilbert', 'roberta', 'albert'
                                 'bert-large', 'roberta-large', 'xlnet-large'],
                        help='Name of the transformer model.',
                        default='xlnet')
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
                        default=0.0)
    parser.add_argument('--special_tokens_fix',
                        type=int,
                        help='Whether to fix problem with [CLS], [SEP] tokens tokenization. '
                             'For reproducing reported results it should be 0 for BERT/XLNet and 1 for RoBERTa.',
                        default=0)
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
    parser.add_argument('--n_steps',
                        help='Number of steps to threshold at',
                        type=int,
                        default=100)
    parser.add_argument('--speech',
                        help='Use model amended for speech data',
                        type=int,
                        default=1)
    
    # Arguments for spoken gec
    parser.add_argument('--spoken_gec',
                        help='Whether we are doing spoken_gec.',
                        type=int,
                        default=1)
    parser.add_argument('--corr_input_file',
                        help='Path to the correct (transcribed) input',
                        default='../data/Linguaskill/preprocessed/ls_test.inc')
    parser.add_argument('--asr_data_file',
                        help='Path to asr confidence data',
                        type=str,
                        default='/scratches/dialfs/alta/th624/exp-th624/Whisper_flt/for_gec/cem_word_confidences.pkl')
    parser.add_argument('--spoken_gec_threshold_mode',
                        help='Decides on how to threshold the asr/gec confidence scores',
                        default='upper_bound',
                        choices=['asr_only', 'gec_only', 'weighted_product', 'upper_bound', 'upper_bound_stc'])
    parser.add_argument('--alpha',
                        type=int,
                        default=1,
                        help='Exponent for asr confidence score when using weighted_product thresholding')
    parser.add_argument('--beta',
                        type=int,
                        default=1,
                        help='Exponent for gec confidence score when using weighted_product thresholding')
    parser.add_argument('--threshold_constant',
                        type=float,
                        default=0.0,
                        help='Constant to threshold other confidence score by when applying "asr_only" or "gec_only" thresholding')
    parser.add_argument('--errant_mod',
                        type=int,
                        help='Whether to use modified ERRANT evaluation for spoken GEC',
                        default=1)
    args = parser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/confidence.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    main(args)