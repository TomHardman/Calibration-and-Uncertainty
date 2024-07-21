import subprocess

cmds1 = ['python3 pr_confidence.py --model_path ../trained_models/xlnet_0_gectorv2.th ../trained_models/bert_0_gectorv2.th ../trained_models/roberta_1_gectorv2.th ../trained_models/xlnet_0_gectorv2finetuned.th ../trained_models/bert_0_gectorv2finetuned.th ../trained_models/roberta_1_gectorv2finetuned.th --output_file experiments/pr_data/confidence/Linguaskill_whisperflt/ensemble_all_asr_only_gec0.5_mod.pkl --is_ensemble 1 --speech 1 --model_name ensemble_all1 --special_tokens_fix 0 --spoken_gec_threshold_mode asr_only --threshold_constant 0.5 --errant_mod 1',
        'python3 pr_confidence.py --model_path ../trained_models/xlnet_0_gectorv2.th ../trained_models/bert_0_gectorv2.th ../trained_models/roberta_1_gectorv2.th ../trained_models/xlnet_0_gectorv2finetuned.th ../trained_models/bert_0_gectorv2finetuned.th ../trained_models/roberta_1_gectorv2finetuned.th --output_file experiments/pr_data/confidence/Linguaskill_whisperflt/ensemble_all_gec_only_asr0.75_mod.pkl --is_ensemble 1 --speech 1 --model_name ensemble_all2 --special_tokens_fix 0 --threshold_constant 0.75 --spoken_gec_threshold_mode gec_only --errant_mod 1',
        'python3 pr_confidence.py --model_path ../trained_models/xlnet_0_gectorv2.th ../trained_models/bert_0_gectorv2.th ../trained_models/roberta_1_gectorv2.th ../trained_models/xlnet_0_gectorv2finetuned.th ../trained_models/bert_0_gectorv2finetuned.th ../trained_models/roberta_1_gectorv2finetuned.th --output_file experiments/pr_data/confidence/Linguaskill_whisperflt/ensemble_all_gec_only_asr0_mod.pkl --is_ensemble 1 --speech 1 --model_name ensemble_all3 --special_tokens_fix 0 --threshold_constant 0 --spoken_gec_threshold_mode gec_only --errant_mod 1',
]

cmds2 = [
        'python3 pr_confidence.py --model_path ../trained_models/xlnet_0_gectorv2.th ../trained_models/bert_0_gectorv2.th ../trained_models/roberta_1_gectorv2.th ../trained_models/xlnet_0_gectorv2finetuned.th ../trained_models/bert_0_gectorv2finetuned.th ../trained_models/roberta_1_gectorv2finetuned.th --output_file experiments/pr_data/confidence/Linguaskill_whisperflt/ensemble_all_prod_1_1_mod.pkl --is_ensemble 1 --speech 1 --model_name ensemble_all5 --special_tokens_fix 0 --spoken_gec_threshold_mode weighted_product --alpha 1 --beta 1 --errant_mod 1',
        'python3 pr_confidence.py --model_path ../trained_models/xlnet_0_gectorv2.th ../trained_models/bert_0_gectorv2.th ../trained_models/roberta_1_gectorv2.th ../trained_models/xlnet_0_gectorv2finetuned.th ../trained_models/bert_0_gectorv2finetuned.th ../trained_models/roberta_1_gectorv2finetuned.th --output_file experiments/pr_data/confidence/Linguaskill_whisperflt/ensemble_all_prod_2_1_mod.pkl --is_ensemble 1 --speech 1 --model_name ensemble_all6 --special_tokens_fix 0 --spoken_gec_threshold_mode weighted_product --alpha 2 --beta 1 --errant_mod 1'
]

if __name__ == '__main__':
    for cmd in cmds2:
        cmd_s = cmd.split(' ')
        subprocess.run(cmd_s)