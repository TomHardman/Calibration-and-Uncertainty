import errant
import numpy as np
import Levenshtein


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


def errant_spoken(source_asr_stcs_al, pred_stcs_al, target_stcs_al, source_trans_stcs_al):
    '''
    Calculates TP, FP, FN, Precison, Recall and F0.5 score for evaluating spoken GEC.
    source_asr_stcs_al - list of aligned ASR output sentences
    pred_stcs_al - list of aligned predicted sentences
    target_stcs_al - list of aligned reference sentences
    source_trans_stcs_al - list of aligned manually transcribed sentences
    '''
    tp = 0
    fp = 0
    ref_tot = 0
    annotator = errant.load('en')

    for stc_asr, stc_pred, stc_tgt, stc_trans in zip(source_asr_stcs_al, pred_stcs_al, target_stcs_al, source_trans_stcs_al):
        hyp_edits = get_stc_edits(annotator, stc_asr, stc_pred)
        ref_edits = get_stc_edits(annotator, stc_trans, stc_tgt)
        ref_tot += len(ref_edits)

        # find operations to get from transcription to asr
        asr_editops = Levenshtein.opcodes(stc_asr.split(' '), stc_trans.split(' '))

        # initialise array to store required span shifts for asr stc - top row stores shift required for start of span
        # bottom row stores shift required for end of span
        asr_shifts = np.zeros((2, len(stc_asr.split(' ')) + 1))
        
        for opcode, asr_start, asr_stop, trans_start, trans_stop in asr_editops:
            if opcode == 'insert':
                asr_shifts[0, asr_start] += trans_stop - trans_start  # start of edits beginning on/after insertion shifted right by length of inserted seq
                asr_shifts[1, asr_start+1] += trans_stop - trans_start # end spans only shifted right after insertion, not on
            elif opcode == 'delete':
                asr_shifts[:, asr_stop] += -(asr_stop - asr_start)
        
        asr_shifts = np.cumsum(np.array(asr_shifts), axis=1)

        # compare edits in sentence
        for hyp_edit in hyp_edits:
            correct = False
            for ref_edit in ref_edits:
                if equal_edits(hyp_edit, ref_edit, asr_shifts):
                    tp += 1
                    correct = True
                    break
            if not correct:
                fp += 1

    fn = ref_tot - tp
    p = tp/(tp + fp)
    r = tp/(tp + fn)
    f_05 = 1.25 * (p * r) / (0.25 * p + r)
    print(f'TP {tp}:, FP:{fp}, FN:{fn}, P: {p}, R: {r}, F0.5: {f_05}')
