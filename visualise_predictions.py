from tqdm import tqdm
import argparse
import os
import sys

from utils.preprocess_data import read_parallel_lines, align_sequences
from gector.gec_model import GecBERTModel, GECBERTModelSpeech
from utils.helpers import read_lines, normalize
import numpy as np
from numpy.random import default_rng
import random
import torch

from pr_confidence import softmax_with_temperature_batch, predict_file_with_threshold

feature_length=768

def main(args, vis=True):
     speech = args.speech
     
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
     
     else:
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

     test_data = read_lines(args.input_file)
     print(f'Test data loaded, {len(test_data)} sentences')

     for i in range(args.iteration_count):
          predictions, probs, pred_idx = predict_file_with_threshold(test_data, model, 0.1, batch_size=args.batch_size,
                                                       to_normalize=args.normalize, ignore_keep=False, speech=speech)

          print(f'Predictions Complete: {len(predictions)} sentences')
          

          if vis:
               for source, pred, stc_probs, stc_idx in zip(test_data[:10], predictions[:10], probs[:10], pred_idx[:10]):
                    print('----------')
                    print(source)
                    print(pred)
                    #seq_al = align_sequences(source, pred, True)
                    #print(seq_al)
                    print(stc_idx)
                    print(stc_probs)
                    print(f'Indexes length: {len(stc_idx)}, Probs length: {len(stc_probs)}')
                    print('----------')
          
          test_data = predictions
     

if __name__ == '__main__':
     # read parameters
     parser = argparse.ArgumentParser()
     parser.add_argument('--model_path',
                         help='Path to the model file.', nargs='+',
                         required=True)
     parser.add_argument('--vocab_path',
                         help='Path to the model file.',
                         default='data/output_vocabulary'  # to use pretrained models
                         )
     parser.add_argument('--input_file',
                         help='Path to the evalset file',
                         required=True)
     parser.add_argument('--output_file',
                         help='Path to the output file',
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
     parser.add_argument('--transformer_model',
                         choices=['bert', 'gpt2', 'transformerxl', 'xlnet', 'distilbert', 'roberta', 'albert'
                                   'bert-large', 'roberta-large', 'xlnet-large'],
                         help='Name of the transformer model.',
                         default='roberta')
     parser.add_argument('--iteration_count',
                         type=int,
                         help='The number of iterations of the model.',
                         default=3)
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
     parser.add_argument('--speech',
                         help='Use model amended for speech data',
                         type=int,
                         default=0)
     args = parser.parse_args()

     # Save the command run
     if not os.path.isdir('CMDs'):
          os.mkdir('CMDs')
     with open('CMDs/vis_predict.cmd', 'a') as f:
          f.write(' '.join(sys.argv)+'\n')

     main(args)



