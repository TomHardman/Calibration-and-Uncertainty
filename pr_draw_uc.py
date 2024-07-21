import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse


def main(args, show_max=False, print_init=True, plot_conf=True):
    # Paths to pickle files containing thresholds and scores
    path_data = f'experiments/pr_data/uncertainty/{args.dataset}/data_uc.pkl'
    path_know = f'experiments/pr_data/uncertainty/{args.dataset}/knowledge_uc.pkl'
    path_tot = f'experiments/pr_data/uncertainty/{args.dataset}/total_uc.pkl'
    path_conf = f'experiments/pr_data/confidence/{args.dataset}/ensemble.pkl'
    matplotlib.use('Agg')

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    uc_dict = {
        'total_uc': 'Total',
        'data_uc': 'Data',
        'knowledge_uc': 'Knowledge'
                }

    for path in [path_data, path_know, path_tot, path_conf]:
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
            t_arr_norm = np.array(t_arr)/np.max(t_arr)

            if path.split('/')[2][0] == 'u':
                uc_type = path.split('/')[-1].split('.')[0]
                label = uc_dict[uc_type]
            
            else:
                label = 'Confidence'

            if label == 'Confidence' and plot_conf or label != 'Confidence':
                ax[0].plot(r_arr, p_arr, label=label)
                idx_opt = np.argmax(f_05_arr)
                if label == 'Confidence':      
                    ax[0].plot(r_arr[idx_opt:idx_opt+1], p_arr[idx_opt:idx_opt+1], 'x', color='r', label='Best Operating Point',
                               zorder=3)
                else:
                    ax[0].plot(r_arr[idx_opt:idx_opt+1], p_arr[idx_opt:idx_opt+1], 'x', color='r', zorder=3)
                ax[1].plot(t_arr_norm, f_05_arr, label=f'{label} : F_0.5 = {f_max}')

            if show_max:
                ax[1].vlines(t_max, ymin=min(f_05_arr), ymax=max(f_05_arr), label=f'F = {f_max} at t = {t_max}',
                             color='r')
            
            if print_init:
                print(label, p_arr[0], r_arr[0], f_05_arr[0], len(p_arr))


    ax[0].set_title('PR Curve')
    ax[0].set_xlabel('Recall')
    ax[0].set_ylabel('Precision')
    ax[0].legend()

    ax[1].set_title(r'F$_{0.5}$ Score against Threshold')
    ax[1].set_xlabel('Normalised Threshold')
    ax[1].set_ylabel(r'F$_{0.5}$ Score')
    ax[1].legend()

    plt.savefig(args.output_file)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        default = 'FCE')
    parser.add_argument('--output_file',
                        required = True)
    args = parser.parse_args()
    main(args)