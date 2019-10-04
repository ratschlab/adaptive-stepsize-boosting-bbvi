import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from boosting_bbvi.core.utils import eprint, debug
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")
from matplotlib.ticker import MaxNLocator
import pandas as pd
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('datapath', '/tmp', 'Path for the data')
flags.DEFINE_string('outfile', 'stdout',
                    'Path for saving the output image, stdout for show')


def analyze_run(prefix):
    all_kl = []
    for i in range(1, 11):
        fname = os.path.join('%s_%d' % (prefix, i), 'kl.csv')
        with open(fname, 'r') as f:
            kl_iter = [float(e.strip()) for e in f.readlines()] # (n_iter, )
        all_kl.append(kl_iter)
    kl_matrix = np.asarray(all_kl) # (n_seed, n_iter)
    print(prefix)
    print(np.average(kl_matrix, axis=0)) # (n_iter, ), avg over seeds
    print(np.std(kl_matrix, axis=0)) # (n_iter, ), avg over seeds
    print(np.median(kl_matrix, axis=0)) # (n_iter, ), avg over seeds


def plot_kl(df, y='kl'):
    # Integer x axis
    fix, ax = plt.subplots()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if False:
        g = sns.FacetGrid(df, col="fw_variant")
        g.map(sns.lineplot, 'fw_iter', y)
        g.add_legend()
    else:
        g = sns.lineplot(x='fw_iter', y=y, hue='fw_variant', data=df)

    plt.show()


def all_df():
    df_list = []
    for prefix in ['fixed', 'line_search', 'adafw', 'ada_afw', 'ada_pfw']:
        for i in range(1, 11):
            fname = os.path.join(FLAGS.datapath, '%s_%d' % (prefix, i), 'kl.csv')
            with open(fname, 'r') as f:
                kl_iter = [float(e.strip()) for e in f.readlines()] # (n_iter, )
            data = {
                    'fw_variant': [prefix] * 20,
                    'fw_iter': list(range(20)),
                    'kl': kl_iter,
                    'seed': [i] * 20,
                    }
            runs_df = pd.DataFrame(data)
            df_list.append(runs_df)

    all_df = pd.concat(df_list).reset_index()
    plot_kl(all_df)


def violin():
    # dictionary for better plot labels
    row_dict = {'adafw': 'adaFW', 'line_search': 'line-search'}
    runs_df = []
    for fw in ['adafw', 'line_search']:
        for d in range(1, 21):
            data_dir = os.path.join(FLAGS.datapath, "%dd" % d, fw)
            with open(os.path.join(data_dir, 'kl.csv'), 'r') as f:
                kl = [float(e.strip()) for e in f.readlines()]

            with open(os.path.join(data_dir, 'ref_kl.csv'), 'r') as f:
                ref_kl = [float(e.strip()) for e in f.readlines()]

            with open(os.path.join(data_dir, 'steps.csv'), 'r') as f:
                gamma = [float(e.strip()) for e in f.readlines()]

            n = len(kl)
            data = {'kl': kl, 'step variant': [fw] * n,
                    'dimension': [d] * n, 'gamma': gamma}
            run_df = pd.DataFrame(data)

            data_init = {'kl': ref_kl, 'step variant': ['init'] * n,
                         'dimension': [d] * n, 'gamma': gamma}
            run_df2 = pd.DataFrame(data_init)
            
            runs_df.append(run_df)
            runs_df.append(run_df2)

    all_df = pd.concat(runs_df).reset_index()
    #step_df = all_df.loc[all_df['step_variant'] != 'init']
    step_df = all_df
    chosen_df = step_df.loc[step_df['dimension'].isin([1, 2, 5, 10])]

    #init_df = all_df.loc[all_df['step_variant'] == 'init']
    #init_df = init_df.loc[init_df['dimension'].isin([1, 2, 5, 10])]


    fix, ax = plt.subplots()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    #ax = sns.violinplot(x="dimension", y="kl", hue="step_variant", data=chosen_df,
    #               palette="muted", split=False, cut=0)
    ax = sns.boxplot(x="dimension", y="kl", hue="step variant", data=chosen_df,
                palette="muted", ax=ax, hue_order=['init', 'adafw', 'line_search'])
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ['step variant', '1st component', 'adaFW', 'line-search'][1:])

    #ax = sns.violinplot(x="dimension", y="gamma", hue="step_variant", data=all_df,
    #                    palette="muted", split=True)
    #sns.lineplot(x='dimension', y='kl', hue="step_variant", data=step_df,
    #             palette="muted", ax=ax)
    #sns.lineplot(x='dimension', y='kl', hue='step_variant', data=init_df,
    #             palette='Reds', err_style=None, ax=ax)
    ax.set_yscale('symlog')
    ax.set_xlabel('Dimensionality')
    ax.set_ylabel('KL-divergence')

    if FLAGS.outfile == 'stdout':
        plt.show()
    else:
        plt.tight_layout()
        ax.get_figure().savefig(FLAGS.outfile)


def main(argv):
    #all_df()
    violin()


if __name__ == "__main__":
    app.run(main)
    #analyze_run('fixed')
    #analyze_run('line_search')
    #analyze_run('adafw')

