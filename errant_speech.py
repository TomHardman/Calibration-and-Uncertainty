import errant
import Levenshtein
import numpy as np


def get_stc_edits(annotator, src, tgt):
    src = annotator.parse(src)
    tgt = annotator.parse(tgt)
    edits = annotator.annotate(src, tgt)
    return edits


def equal_edits(edit_hyp, edit_ref, asr_shifts):
    if all([edit_hyp.o_start + asr_shifts[0][edit_hyp.o_start] == edit_ref.o_start, 
            edit_hyp.o_end + asr_shifts[1][edit_hyp.o_end] == edit_ref.o_end, 
            edit_hyp.c_str == edit_ref.c_str,
            edit_hyp.o_str == edit_ref.o_str]):
        return True
    return False


def errant_compare_spoken(source_stcs_al, pred_stcs_al, target_stcs_al, corr_inp_stcs_al, log=True):
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
        
        for opcode, asr_start, asr_stop, trans_start, trans_stop in asr_editops:
            if opcode == 'insert':
                asr_shifts[0, asr_start] += trans_stop - trans_start
                asr_shifts[1, asr_start+1] += trans_stop - trans_start
            elif opcode == 'delete':
                asr_shifts[:, asr_stop] += - (asr_stop - asr_start)
        
        asr_shifts = np.cumsum(np.array(asr_shifts), axis=1)

                
        for hyp_edit in hyp_edits:
            correct = False
            for ref_edit in ref_edits:
                if equal_edits(hyp_edit, ref_edit, spoken=t!=0.0, asr_shifts=asr_shifts):
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
    
    fn = ref_tot - tp
    p = tp/(tp + fp)
    r = tp/(tp + fn)
    f_05 = 1.25 * (p * r) / (0.25 * p + r)
    if log:
        print(f'TP {tp}:, FP:{fp}, FN:{fn}, P: {p}, R: {r}, F0.5: {f_05}')