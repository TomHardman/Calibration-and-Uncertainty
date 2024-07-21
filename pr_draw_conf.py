import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Paths to pickle files containing thresholds and scores
matplotlib.use('Agg')


def main(args, show_max=True, print_init=True, show_upper_bound=False):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    paths = args.filepaths
    labels = args.labels

    for path, label in zip(paths, labels):
        if path:
            with open(path, 'rb') as file:
                data = pickle.load(file)

            p_arr = []
            r_arr = []
            t_arr = []
            f_05_arr = []

            for t, p, r, f_05 in data:
                p_arr.append(float(p))
                r_arr.append(float(r))
                t_arr.append(float(t))
                f_05_arr.append(float(f_05))
            
            f_max = np.max(f_05_arr)
            t_max = t_arr[np.argmax(f_05_arr)]
            p_opt = p_arr[np.argmax(f_05_arr)]
            r_opt = r_arr[np.argmax(f_05_arr)]
            model_name = path.split('/')[-1].split('.')[0]

            ax[0].plot(r_arr, p_arr, label=label)
            ax[1].plot(t_arr, f_05_arr, label=label)

            if show_upper_bound:
                ax[0].hlines(0.538, xmin=0, xmax=r_arr[-1], label='Precision upper bound 0.5388', color='r')
                ax[1].hlines(0.4326, ymin=0, ymax=1, label=r'$F_{0.5}$ upper bound 0.4326', color='r')

            if show_max:
                ax[1].vlines(t_max, ymin=min(f_05_arr), ymax=max(f_05_arr), label=f'F = {f_max:.4f}, P = {p_opt:.4f}, R = {r_opt:.4f}',
                             color='r')
            
            if print_init:
                print(model_name, p_arr[0], r_arr[0], f_05_arr[0])


    ax[0].set_title('PR Curve')
    ax[0].set_xlabel('Recall')
    ax[0].set_ylabel('Precision')
    ax[0].legend()

    ax[1].set_title('F0.5 Score against Confidence Threshold')
    ax[1].set_xlabel('Threshold')
    ax[1].set_ylabel(r'$F_{0.5}$ Score')
    ax[1].legend()

    plt.savefig(args.outfile)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepaths',
                        help='Path to the model file.', nargs='+',
                        default=['/scratches/dialfs/alta/th624/exp-th624/Calibration_Uncertainty_GEC/experiments/pr_data/confidence/Linguaskill_whisperflt/ensemble_all_asr_only_gec0_mod.pkl',
                                 '/scratches/dialfs/alta/th624/exp-th624/Calibration_Uncertainty_GEC/experiments/pr_data/confidence/Linguaskill_whisperflt/ensemble_all_asr_only_gec0.5_mod.pkl',
                                 '/scratches/dialfs/alta/th624/exp-th624/Calibration_Uncertainty_GEC/experiments/pr_data/confidence/Linguaskill_whisperflt/ensemble_all_gec_only_asr0_mod.pkl',
                                 '/scratches/dialfs/alta/th624/exp-th624/Calibration_Uncertainty_GEC/experiments/pr_data/confidence/Linguaskill_whisperflt/ensemble_all_gec_only_asr0.75_mod.pkl',
                                 '/scratches/dialfs/alta/th624/exp-th624/Calibration_Uncertainty_GEC/experiments/pr_data/confidence/Linguaskill_whisperflt/ensemble_all_prod_1_1_mod.pkl',
                                 '/scratches/dialfs/alta/th624/exp-th624/Calibration_Uncertainty_GEC/experiments/pr_data/confidence/Linguaskill_whisperflt/ensemble_all_prod_2_1_mod.pkl'
                                 ])
    parser.add_argument('--labels',
                        help='Path to the model file.', nargs='+',
                        default=[r'ASR Only, $k_{gec} = 0$',
                                 r'ASR Only, $k_{gec} = 0.5$',
                                 r'GEC Only, $k_{asr} = 0$',
                                 r'GEC Only, $k_{asr} = 0.75$',
                                 r'Prod $\alpha=1$, $\beta=1$',
                                 r'Prod $\alpha=2$, $\beta=1$'])
    parser.add_argument('--outfile',
                        default='pr_whisper.png')
    args = parser.parse_args()
    main(args)