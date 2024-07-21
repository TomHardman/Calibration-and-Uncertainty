import subprocess
import argparse
from align import align_no_file
from utils.helpers import read_lines


def main(args):
    model_name = args.model_name
    dataset = args.dataset
    spoken_gec = False

    if dataset == 'FCE':
        dset_inc_path = f'../data/FCE/fce-public.test.inc'
        dset_corr_path = f'../data/FCE/fce-public.test.corr'
        dset_pred_path = f'experiments/prediction_files/FCE/{model_name}.pred'
    
    elif dataset == 'Linguaskill_whisperflt':
        dset_corr_inp_path  = f'../data/Linguaskill/preprocessed/ls_test.inc' # correct speech transcriptions
        dset_inc_path = f'../data/Linguaskill/preprocessed_whisperflt/ls_test.inc' # ASR output
        dset_corr_path = f'../data/Linguaskill/preprocessed_whisperflt/ls_test.corr'
        dset_pred_path = f'experiments/prediction_files/{dataset}/{model_name}.pred'
        spoken_gec = True
    
    elif dataset.split('_')[0] == 'Linguaskill':
        dset_inc_path = f'../data/Linguaskill/preprocessed/ls_test.inc'
        dset_corr_path = f'../data/Linguaskill/preprocessed/ls_test.corr'
        dset_pred_path = f'experiments/prediction_files/{dataset}/{model_name}.pred'

    # Setup paths for errant
    pred_path = f'experiments/prediction_files/{dataset}/errant/{model_name}.pred'
    inc_path = f'experiments/prediction_files/{dataset}/errant/{model_name}.inc'
    corr_path = f'experiments/prediction_files/{dataset}/errant/{model_name}.corr'
    corr_inp_path = f'experiments/prediction_files/{dataset}/errant/{model_name}.corr_inp'
    hyp_ann_path = f'experiments/prediction_files/{dataset}/errant/{model_name}_hyp.m2'
    ref_ann_path = f'experiments/prediction_files/{dataset}/errant/{model_name}_ref.m2'

    # Setup paths for reading files
    anno_cmd_pred = ["errant_parallel", "-orig", inc_path, "-cor", pred_path, "-out", hyp_ann_path]
    if not spoken_gec:
        anno_cmd_tgt = ["errant_parallel", "-orig", inc_path, "-cor", corr_path, "-out", ref_ann_path]
    else:
        anno_cmd_tgt = ["errant_parallel", "-orig", corr_inp_path, "-cor", corr_path, "-out", ref_ann_path]
    compare_cmd = ["errant_compare", "-hyp", hyp_ann_path, "-ref", ref_ann_path]

    pred_stcs = read_lines(dset_pred_path)
    target_stcs = read_lines(dset_corr_path)
    source_stcs = read_lines(dset_inc_path)
    source_stcs_al, pred_stcs_al, target_stcs_al = align_no_file(source_stcs, pred_stcs, target_stcs)

    if spoken_gec:
        corr_inp_stcs = read_lines(dset_corr_inp_path)
        source_stcs_al, pred_stcs_al, target_stcs_al, corr_inp_stcs_al = align_no_file(source_stcs, pred_stcs, target_stcs, 
                                                                                       trans_inp_sent=corr_inp_stcs, spoken_gec=True)

    with open(pred_path, 'w') as f:  # write aligned sentences to file
        f.write("\n".join(pred_stcs_al) + '\n')
    with open(inc_path, 'w') as f:
        f.write("\n".join(source_stcs_al) + '\n')
    with open(corr_path, 'w') as f:
        f.write("\n".join(target_stcs_al) + '\n')
    
    if spoken_gec:
        with open(corr_inp_path, 'w') as f:
            f.write("\n".join(corr_inp_stcs_al) + '\n')
        
    subprocess.run(anno_cmd_tgt) # create target annotation file
    subprocess.run(anno_cmd_pred) # create prediction annotation file
    output = subprocess.check_output(compare_cmd, universal_newlines=True)
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        default = 'FCE')
                    
    parser.add_argument('--model_name',
                        default='bert')
    args = parser.parse_args()
    main(args)