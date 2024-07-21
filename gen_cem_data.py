from tqdm import tqdm
import argparse

from utils.preprocess_data import read_parallel_lines, align_sequences
from gector.gec_model import GecBERTModel
import numpy as np
from numpy.random import default_rng
import random

feature_length=768

def my_predict_sing_stc(stc, model, masking, masking_idxs=None):
    '''
    predict single sentence, return output sentence, probability lists and index list
    '''
    batch = [stc.split()]
    sequences = model.preprocess(batch)

    probabilities, idxs, error_probs, all_error_probs, logits_labels, embeddings = model.predict(sequences, masking, masking_idxs)
    
    pred_batch = model.postprocess_batch(batch, probabilities,
                                        idxs, error_probs)

    result_line = [" ".join(x) for x in pred_batch]

    return result_line[0] ,probabilities[0], idxs[0], all_error_probs[0], logits_labels[0], embeddings[0]

def get_single_stc_prob_edits(source_sent, target_sent, model, tot_zeros_e,tot_ones_e, masking, masking_idxs=None):
    e_0s = 0
    e_1s = 0
    
    # import pdb;pdb.set_trace()
    pred_sent, probs, idxs, _, logits_labels, all_embeddings = my_predict_sing_stc(source_sent, model, masking, masking_idxs) # make predictions
    # target_sent, probs, idxs, _, logits_labels, all_embeddings = my_predict_sing_stc(source_sent, model, masking, masking_idxs[round(num/2):])

    vocab = model.vocab
    # pick only the probabilities and embeddings at edits indices
    edits_pred = [[(i-1,i),vocab.get_token_from_index(idxs[i], namespace = 'labels')] for i in range(len(idxs)) if (idxs[i] != 0)] #  and idxs[i] != 5000
    # edits_idx = [i for i in range(len(idxs)) if idxs[i] != 0]
    # edit_logits_labels = [logits_labels[i] for i in edits_idx]

    try:
        _, edits_tgt = align_sequences(source_sent, target_sent, True)
        _, edits_pred_ = align_sequences(source_sent, pred_sent, True)
    except Exception:
        _, edits_tgt = align_sequences(source_sent, target_sent, True)
        _, edits_pred_ = align_sequences(source_sent, pred_sent, True)

    # for cme training
    labels_for_cme=[1]*len(idxs)
    mask_for_cme = [0]*len(idxs) # mask to show where is editted

    n = 0
    # if any(e[1]==vocab.get_token_from_index(5000, namespace = 'labels') for e in edits_tgt):
    #     print(f'edits_pred:{edits_pred}\n edits_tgt:{edits_tgt}\n pred_sent:{pred_sent} \n source_sent:{source_sent} \n target:{target_sent}\n\n')
    for pred_e in edits_pred:
        # print(pred_e)
        position = pred_e[0][1]
        mask_for_cme[position]=1 # ?
        if not any(my_equal_edits(e, pred_e) for e in edits_tgt):
            labels_for_cme[position] = 0
            e_0s+=1
        else:
            e_1s+=1

    tot_ones_e += e_1s
    tot_zeros_e += e_0s
    
    sentence = {
                'embedding': all_embeddings,
                # 'label_logits': logits_labels, 
                # 'e_indices':edits_idx,
                'op_tgts':labels_for_cme,
                'mask':mask_for_cme,
                'prob':probs,
                # 'e_label_logits': edit_logits_labels
                }  
    
    return sentence, tot_ones_e, tot_zeros_e


def get_single_stc_logits_only(source_sent, target_sent, model, tot_zeros_e,tot_ones_e, masking, masking_idxs=None):
    pred_sent, probs, idxs, _, logits_labels, embeddings = my_predict_sing_stc(source_sent, model, masking, masking_idxs) # make predictions
    # target_sent, probs, idxs, _, logits_labels, all_embeddings = my_predict_sing_stc(source_sent, model, masking, masking_idxs[round(num/2):])

    vocab = model.vocab
    # pick only the probabilities and embeddings at edits indices
    edits_pred = [[(i-1,i),vocab.get_token_from_index(idxs[i], namespace = 'labels')] for i in range(len(idxs)) if (idxs[i] != 0)] #  and idxs[i] != 5000
    edits_idx = [i for i in range(len(idxs)) if idxs[i] != 0]
    # edit_logits_labels = [logits_labels[i] for i in edits_idx]

    try:
        _, edits_tgt = align_sequences(source_sent, target_sent, True)
        _, edits_pred_ = align_sequences(source_sent, pred_sent, True)
    except Exception:
        _, edits_tgt = align_sequences(source_sent, target_sent, True)
        _, edits_pred_ = align_sequences(source_sent, pred_sent, True)
    
    sentence=[]

    for pred_e, idx in zip(edits_pred, edits_idx):
        # edit_logits_labels.append(logits_labels[idx])
        if not any(my_equal_edits(e, pred_e) for e in edits_tgt):
            tot_zeros_e += 1
            sentence.append({'op_tgts':0,'e_label_logits': embeddings[idx], 'prob':probs[idx], 'idx':idxs[idx]})
        else:
            tot_ones_e += 1
            # labels_for_cme.append(1)
            # sentence.append({'op_tgts':1,'e_label_logits': logits_labels[idx]})
            sentence.append({'op_tgts':1,'e_label_logits': embeddings[idx], 'prob':probs[idx],'idx':idxs[idx]})
    
    # labels_for_cme = [labels_for_cme[i] for i in edits_idx]
    # sentence = {'op_tgts':labels_for_cme,'e_label_logits': edit_logits_labels}
    
    return sentence, tot_ones_e, tot_zeros_e, pred_sent, len(edits_tgt)

def my_get_prob_edits(source_file, target_file, model, masking = None, interval=[0.0,1.0], masking_ite=1, attn=True):
    '''
    input: source file and target file path, model
    output: edits indices, all embeddings, all logits for labels, edits logits, target
    
    masking: percent of masking in embedding space, masking_idx: position of masking, optional
    '''
    source_data, target_data = read_parallel_lines(source_file, target_file)
    source_data = source_data[int(interval[0]*len(source_data)):int(interval[1]*len(source_data))]
    target_data = target_data[int(interval[0]*len(target_data)):int(interval[1]*len(target_data))]

    data = []
    tot_zeros_e = 0
    tot_ones_e = 0
    
    for source_sent, target_sent in tqdm(zip(source_data, target_data)):
        if masking:
            width_t = min(round(feature_length*masking),feature_length)
            rng=default_rng()
            numbers = rng.choice(feature_length, size=width_t*masking_ite, replace=False)
            
            for i in range(masking_ite):
                if attn:
                    sentence, tot_ones_e, tot_zeros_e=get_single_stc_prob_edits(
                        source_sent, target_sent, model, tot_zeros_e,tot_ones_e,
                        masking, numbers[i*width_t:(i+1)*width_t])
                    data.append(sentence)
                else:
                    sentence, tot_ones_e, tot_zeros_e,_,_=get_single_stc_logits_only(
                        source_sent, target_sent, model, tot_zeros_e,tot_ones_e,
                        masking, numbers[i*width_t:(i+1)*width_t])
                    data.extend(sentence)

            # sentence, tot_ones_e, tot_zeros_e=get_single_stc_prob_edits(
            #         source_sent, target_sent, model, tot_zeros_e,tot_ones_e,
            #         masking, numbers)
        else:
            if attn:
                sentence, tot_ones_e, tot_zeros_e=get_single_stc_prob_edits(
                        source_sent, target_sent, model, tot_zeros_e,tot_ones_e,
                        masking)
                data.append(sentence)
            else:
                sentence, tot_ones_e, tot_zeros_e,_,_=get_single_stc_logits_only(
                        source_sent, target_sent, model, tot_zeros_e,tot_ones_e,
                        masking)
                data.extend(sentence)
    
    print(f"e 1s:{tot_ones_e}, e 0s:{tot_zeros_e}")
    return data

def examine_masking_data(source_file, target_file, model, masking):
    source_data, target_data = read_parallel_lines(source_file, target_file)

    tot_zeros_e = 0
    tot_ones_e = 0
    
    tot_zeros_wrt_nomasking = 0
    tot_ones_wrt_nomasking = 0
    
    # source_data = source_data[:10]
    # target_data = target_data[:10]
    
    for source_sent, target_sent in tqdm(zip(source_data, target_data)):
        width_t = min(round(feature_length*masking),feature_length)
        rng=default_rng()
        numbers = rng.choice(feature_length, size=width_t, replace=False)
        
        _, _, _,pred_sent=get_single_stc_logits_only(
            source_sent, target_sent, model, 0,0,
            masking=None)

        _, tot_ones_e, tot_zeros_e,_=get_single_stc_logits_only(
            source_sent, target_sent, model, 
            tot_zeros_e,tot_ones_e,
            masking, numbers)
        _, tot_ones_wrt_nomasking, tot_zeros_wrt_nomasking,_=get_single_stc_logits_only(
            source_sent, pred_sent, model, 
            tot_zeros_wrt_nomasking,tot_ones_wrt_nomasking,
            masking, numbers)
        
    out_txt = f"\nmasking:{masking}\ne 1s:{tot_ones_e}, e 0s:{tot_zeros_e}\ncompare to no masking: e 1s:{tot_ones_wrt_nomasking}, e 0s:{tot_zeros_wrt_nomasking}"
    
    print(out_txt)
    return out_txt

def examine_unmasked_data(source_file, target_file, model, output_file = None):
    source_data, target_data = read_parallel_lines(source_file, target_file)
    sentences = []

    tot_zeros_e = 0
    tot_ones_e = 0
    tot_tgt_e = 0
    
    for source_sent, target_sent in tqdm(zip(source_data, target_data)):    
        _, tot_ones_e, tot_zeros_e,pred_sent, no_tgt_edits=get_single_stc_logits_only(
            source_sent, target_sent, model, 
            tot_zeros_e,tot_ones_e, masking = None)
        tot_tgt_e+=no_tgt_edits
        sentences.append(pred_sent)

    if output_file:
        with open(output_file, 'w') as f:
            f.write("\n".join(sentences) + '\n')
            
    output_txt = f'\nunmasked, source file:{source_file}\ne 1s:{tot_ones_e}, e 0s:{tot_zeros_e}, tgt es:{tot_tgt_e}'
    return output_txt

def predict_for_file(source_file, target_file, model, output_path, masking = None):
    source_data, target_data = read_parallel_lines(source_file, target_file)

    data = []
    for source_sent in source_data:
        
        pred_sent, probs, idxs, _, logits_labels, all_embeddings = my_predict_sing_stc(source_sent, model, masking) # make predictions
        
        data.append(pred_sent)
        # import pdb; pdb.set_trace()
    
    with open(output_path, 'w') as f:
        f.write("\n".join(data) + '\n')
        
    return data

def my_get_logits_only(source_file, target_file, model):
    '''
    input: source file and target file path, model
    output: edits indices, all embeddings, all logits for labels, edits logits, target
    '''
    source_data, target_data = read_parallel_lines(source_file, target_file) # read lines from source_file and target_file
    vocab = model.vocab


    data = []
    for source_sent, target_sent, i in tqdm(zip(source_data, target_data, range(len(source_data)))):
        pred_sent, probs, idxs, _, logits_labels, embeddings = my_predict_sing_stc(source_sent, model) # make predictions

        # pick only the probabilities and embeddings at edits indices
        edits_pred = [[(i-1,i),vocab.get_token_from_index(idxs[i], namespace = 'labels')] for i in range(len(idxs)) \
            if (idxs[i] != 0)]
        edits_idx = [i for i in range(len(idxs)) if (idxs[i] != 0)]
        
        # if len(idxs)!=len(probs):
        #     print(f'source:{source_sent} \n pred:{pred_sent} \n target:{target_sent} \n index:{i}')

        try:
            _, edits_tgt = align_sequences(source_sent, target_sent, True)
            _, edits_pred1 = align_sequences(source_sent, pred_sent, True)
        except Exception:
            _, edits_tgt = align_sequences(source_sent, target_sent, True)
            _, edits_pred1 = align_sequences(source_sent, pred_sent, True)

        # for cme training
        # if len(edits_pred) != len(edits_pred1):
        #     print(f'source:{source_sent} \n pred:{pred_sent} \n target:{target_sent} \n index:{i}')
        #     print(f'edits pred:{edits_pred} \n {edits_pred1} \n edits target:{edits_tgt}')
        sentence=[]

        for pred_e, idx in zip(edits_pred, edits_idx):
            # if idxs[idx]==1243:
            #     print(probs[idx])
            #     print(f'source:{source_sent} \n pred:{pred_sent} \n target:{target_sent} \n index:{idxs}')
                
            # edit_logits_labels.append(logits_labels[idx])
            if not any(my_equal_edits(e, pred_e) for e in edits_tgt):
                sentence.append({'op_tgts':0,'e_label_logits': embeddings[idx], 'prob':probs[idx], 'idx':idxs[idx]})
            else:
                sentence.append({'op_tgts':1,'e_label_logits': embeddings[idx], 'prob':probs[idx],'idx':idxs[idx]})
        
        # labels_for_cme = [labels_for_cme[i] for i in edits_idx]
        # sentence = {'op_tgts':labels_for_cme,'e_label_logits': edit_logits_labels}

        data.extend(sentence)

    return data

def my_equal_edits(edit1, edit2):
    return edit1[0] == edit2[0] and edit1[1] == edit2[1]

def write_to_file(data, outputpath):
    # write for CEM training data
    # import pdb;pdb.set_trace()
    np.save(outputpath,data,allow_pickle=True)
    # np.load(outputpath, allow_pickle=True).tolist()
    return


def main(args):
    # get all paths
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
                         weigths=args.weights,)
    
    print(args.masking, args.train_data_folder, args.interval)
    # out_txt = examine_unmasked_data(args.input_file, args.target_file, model, args.output_text_file)
    # with open('/home/zl437/rds/hpc-work/gector/out.txt', 'a') as f:
    #     f.write(out_txt)
    data = my_get_prob_edits(
        args.input_file, args.target_file, model, masking=args.masking, 
        interval=args.interval, masking_ite=args.masking_ite, attn=True)
    if args.train_data_folder:
        write_to_file(data, args.train_data_folder)

if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help='Path to the model file.', nargs='+',
                        default=['/home/alta/BLTSpeaking/exp-zl437/demo/model_para/gector/roberta_1_gectorv2.th']
                        )
    parser.add_argument('--vocab_path',
                        help='Path to the model file.',
                        default='/home/alta/BLTSpeaking/exp-zl437/demo/gector/data/output_vocabulary'  # to use pretrained models
                        )
    parser.add_argument('--input_file',
                        help='Path to the evalset file',
                        # default='/home/zl437/rds/hpc-work/gector/fce-data/source-test.txt'
                        default='/home/alta/BLTSpeaking/exp-zl437/demo/gector/fce-data/source-train.txt'
                        # default='/home/zl437/rds/hpc-work/gector/bea-data/nucle/source-test.txt'
                        )
    parser.add_argument('--masking', 
                        type=float,
                        help='masking percent, default: None',
                        default=None)
    parser.add_argument('--masking_ite', 
                        type=int,
                        help='masking iterations',
                        default=1)
    parser.add_argument('--interval', 
                        nargs='+', type=float,
                        help='interval of dataset',
                        default=[0.0,1.0])
    parser.add_argument('--output_text_file', 
                        help='path to save the output text file',
                        default='predict-test.txt')
    parser.add_argument('--target_file',
                        help='Path to the target file',
                        default='/home/alta/BLTSpeaking/exp-zl437/demo/gector/fce-data/target-train.txt'
                        # default='/home/zl437/rds/hpc-work/gector/fce-data/target-test.txt'
                        # default='/home/zl437/rds/hpc-work/gector/fce-data/dev/fce_target.corr'
                        # default='/home/zl437/rds/hpc-work/gector/bea-data/nucle/target-test.txt'
                        )
    parser.add_argument('--train_data_folder',
                        help='Path to the output data path',
                        # default='/home/zl437/rds/hpc-work/gector/CEM/data/fce/self_attn_dev.npy'
                        default='/home/alta/BLTSpeaking/exp-zl437/demo/gector/CEM/data/self_attn/train.npy'
                        # default='/home/zl437/rds/hpc-work/gector/CEM/data/bea_nucle/test_selfAttn.npy'
                        # default='/home/zl437/rds/hpc-work/gector/CEM/data/bea_nucle/masking_self_attn/train_3.npy'
                        )
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