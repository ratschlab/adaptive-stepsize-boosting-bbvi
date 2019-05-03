"""Collect run data, parse and summarize.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
from matplotlib.ticker import MaxNLocator
import argparse

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from boosting_bbvi.core.utils import eprint, debug

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cluster", help='Is the log file from a cluster', default=False)
parser.add_argument(
    "--metric", help='metric to plot', default='roc')
parser.add_argument(
    "--n_fw_iter", help='number of fw iterations to plot', default=25, type=int)
parser.add_argument(
    "--adaptive_var",
    help='adaptive variant to process',
    nargs='+',
    choices=['adafw', 'ada_pfw', 'ada_afw'])
parser.add_argument(
    "--select_adaptive",
    default='all',
    choices=['all', 'seed_best', 'hp_best'],
    help="""method to select adaptive runs,
                all: all runs
                seed_best: for each seed, choose the best run
                hp_best: choose the best hp over all runs
                """)
parser.add_argument('--datapath', type=str, help='directory containing run data')
args = parser.parse_args()

def parse_command(cmd):
    #NOTE: Maybe just use argparse here
    """Parse running command
    
    Tokens split by = or space
    """
    tokens = cmd.split()
    if not tokens[0] == 'python' or not tokens[1].endswith('py'):
        raise ValueError('Not a valid command \n%s' % cmd)
    tokens = tokens[2:]
    result = {}
    while tokens:
        token = tokens[0]
        if token.startswith('--'):
            token = token[2:].strip()
            if '=' in token:        # --n_monte_carlo_samples=1000
                param, value = token.split('=')
                tokens = tokens[1:] # right shift by 1
            else:                   # --n_monte_carlo_samples 1000
                param = token
                value = tokens[1].strip()
                tokens = tokens[2:] # right shift by 2

            result[param] = value
        else:
            eprint("%s not parsed" % token)
            tokens = tokens[1:]

    return result


def parse_run(run):
    """Parse a run.
        Example - fixed_mvn_init_vi, ada_pfw_mvn_init_random_1_48,
        fixed_mvl_init_random_1
    """
    order_tok = ['fw_variant', 'base_dist', 'iter0', 'i', 'counter']
    idx_tok = 0
    tokens = run.split('_')
    result = {}
    while tokens:
        param = order_tok[idx_tok]
        if param == 'fw_variant' and (tokens[0] == 'ada'):
            value = "%s_%s" % (tokens[0], tokens[1])
            tokens = tokens[2:]
        elif param == 'iter0':
            assert tokens[0] == 'init', 'parsing %s failed' % run
            value = tokens[1]
            tokens = tokens[2:]
        else:
            value = tokens[0]
            tokens = tokens[1:]
        result[param] = value
        idx_tok += 1

    if 'i' not in result:
        result['i'] = 0
    if 'counter' not in result:
        result['counter'] = 0

    return result


def parse_log(run_name, path):
    """Parse log
    
    run_name: name of directory
    path: full path, assumes run.log present with first line being
        command
    cluster: if the log file is from the clusters (euler/leonhard/leomed)
        or normal stdout
    """
    if os.path.isfile(path):
        with open(path, 'r') as f:
            while args.cluster:
                if f.readline().startswith('# LSBATCH: User input'):
                    break
            cmd = f.readline()
            res = parse_command(cmd)
    else:
        res = parse_run(run_name)

    # Convert numerical values
    for key in res:
        if key in [
                'n_fw_iter', 'LMO_iter', 'n_monte_carlo_samples',
                'adafw_MAXITER'
        ]:
            res[key] = int(res[key])
        elif key in ['linit_fixed', 'exp_adafw', 'damping_adafw']:
            res[key] = float(res[key])

    if 'iter0' not in res:
        res['iter0'] = 'vi'

    # Fixed variant does not have lipschitz estimate
    if 'linit_fixed' not in res:
        res['linit_fixed'] = (0.001
                              if res['fw_variant'].startswith('ada') else None)
    if 'exp_adafw' not in res:
        res['exp_adafw'] = (2.0
                              if res['fw_variant'].startswith('ada') else None)
    if 'damping_adafw' not in res:
        res['damping_adafw'] = (0.99
                              if res['fw_variant'].startswith('ada') else None)

    if 'seed' not in res:
        res['seed'] = 0

    return res

def plot_mvl(df, y='roc', iter0_split=False):
    # Integer x axis
    fix, ax = plt.subplots()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if iter0_split:
        g = sns.FacetGrid(df, col="fw_variant", hue='iter0')
        g.map(sns.lineplot, 'fw_iter', y)
        g.add_legend()
    else:
        # show only vi iter0
        df = df.loc[df['iter0'] == 'vi']
        sns.lineplot(x='fw_iter', y=y, hue='fw_variant', data=df)



def plot_base_dist(df, y='roc'):
    # FIXME
    # Integer x axis
    fix, ax = plt.subplots()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    sns.lineplot(x='fw_iter', y=y, hue='fw_variant', style='base_dist', data=df)


def plot_adaptive(df, iter0_split=False):
    """Analyze the performance of adaptive variants.

    Args:
        base_split: keep only Laplace
        iter0_split: keep only vi
    """
    if not iter0_split: df = df.loc[df['iter0'] == 'vi']
    df = df.loc[df['fw_variant'].isin(['adafw', 'ada_afw', 'ada_pfw'])]

    # plot linit with style
    g = sns.FacetGrid(df, col="fw_variant", hue='linit_fixed')
    g.map(sns.lineplot, 'fw_iter', 'roc')
    g.add_legend()

    ## style plot gets very crowded
    #sns.lineplot(
    #    x='fw_iter', y='roc', hue='fw_variant', style='linit_fixed', data=df)

    # plot eta, tau best results heatmap
    plt.figure()
    #hp_matrix = df.groupby(['eta', 'tau'])['roc'].max().unstack()
    hp_matrix = pd.crosstab(df.eta, df.tau, values=df.roc, aggfunc='max')
    sns.heatmap(hp_matrix)

    # plot iter info
    fig, ax = plt.subplots(len(args.adaptive_var), 1)
    for i, fw_var in enumerate(args.adaptive_var):
        df_var = df.loc[df['fw_variant'] == fw_var]
        t_matrix = pd.crosstab(df_var.fw_iter, df_var.iter_type)
        #t_matrix.columns.names = ['']
        #t_matrix.reset_index(inplace=True)
        #sns.barplot(x='fw_iter', hue='iter_info', data=t_matrix)
        t_matrix.plot(kind='bar', stacked=True, title=fw_var, ax=ax[i])

    # plot lipschitz estimates
    pass


def filter_adaptive(df, y='roc'):
    """Select adaptive runs to process"""
    #TODO vectorize this
    if y not in ['roc', 'elbo']:
        raise NotImplementedError('gap and other metrics should be minimized')
    if args.select_adaptive != 'all':
        filtered = [df.loc[df['fw_variant'] == 'fixed']]
        for fw_var in args.adaptive_var:
            df_var = df.loc[df['fw_variant'] == fw_var]
            if args.select_adaptive == 'hp_best':
                # find row with max val and use it's hps
                best_idx = df_var[y].idxmax()
                df_var_filtered = df_var.loc[
                    (df_var['eta'] == df_var.loc[best_idx]['eta'])
                    & (df_var['tau'] == df_var.loc[best_idx]['tau'])
                    & (df_var['linit_fixed'] == df_var.loc[best_idx]['linit_fixed'])]
                filtered.append(df_var_filtered)
            elif args.select_adaptive == 'seed_best':
                # for each seed, find the best parameter configurations
                best_idx = df_var.groupby(['seed'])[y].idxmax()
                df_var_best_iter = df_var.loc[best_idx]
                for index, row in df_var_best_iter.iterrows():
                    df_var_filtered = df_var.loc[
                        (df_var['eta'] == row['eta'])
                        & (df_var['tau'] == row['tau'])
                        & (df_var['linit_fixed'] == row['linit_fixed'])
                        & (df_var['seed'] == row['seed'])]
                    filtered.append(df_var_filtered)
        df = pd.concat(filtered)

    #debug(df.groupby(['seed', 'fw_variant']).size())
    df.reset_index(inplace=True)

    return df

def pad_or_crop(a, n):
    """Fix len of list a to n"""
    if len(a) > n:
        a = a[:n]
    elif len(a) < n:
        a.extend([None] * (n - len(a)))
    return a


def main():
    run_names = [
        d for d in os.listdir(args.datapath)
        if os.path.isdir(os.path.join(args.datapath, d))
    ]
    run_paths = [os.path.join(args.datapath, d) for d in run_names]

    df = pd.DataFrame()
    cnt = 0
    runs_df = []
    seeds_used = set()
    for nr, dr in zip(run_names, run_paths):
        param_dict = parse_log(nr, os.path.join(dr, 'run.log'))
        if param_dict['fw_variant'] not in args.all_var:
            continue
        seeds_used.add(param_dict.get('seed', 0))

        rocs_filename = os.path.join(dr, 'roc.csv')
        with open(rocs_filename, 'r') as f:
            rocs = [float(r.strip()) for r in f.readlines()]

        # set number of iterations to plot
        #n_fw_iter = param_dict.get('n_fw_iter', len(rocs))
        n_fw_iter = args.n_fw_iter

        ll_train_filename = os.path.join(dr, 'll_train.csv')
        with open(ll_train_filename, 'r') as f:
            ll_trains = [float(r.split(',')[0]) for r in f.readlines()]

        ll_test_filename = os.path.join(dr, 'll_train.csv')
        with open(ll_test_filename, 'r') as f:
            ll_tests = [float(r.split(',')[0]) for r in f.readlines()]

        elbos_filename = os.path.join(dr, 'elbos.csv')
        with open(elbos_filename, 'r') as f:
            elbos = [float(r.strip()) for r in f.readlines()]

        gap_filename = os.path.join(dr, 'gap.csv')
        with open(gap_filename, 'r') as f:
            gaps = [float(r.strip()) for r in f.readlines()]

        iter_info_filename = os.path.join(dr, 'iter_info.txt')
        if os.path.isfile(iter_info_filename):
            with open(iter_info_filename, 'r') as f:
                iter_types = [r.strip() for r in f.readlines()]
        else:
            iter_types = ['fixed'] * n_fw_iter

        rocs = pad_or_crop(rocs, n_fw_iter)
        ll_trains = pad_or_crop(ll_trains, n_fw_iter)
        ll_tests = pad_or_crop(ll_tests, n_fw_iter)
        elbos = pad_or_crop(elbos, n_fw_iter)
        gaps = pad_or_crop(gaps, n_fw_iter)
        iter_types = pad_or_crop(iter_types, n_fw_iter)

        data = {
            'roc': rocs,
            'll_train': ll_trains,
            'll_test': ll_tests,
            'elbo': elbos,
            'gap': gaps,
            'iter_type': iter_types,
            'fw_iter': list(range(n_fw_iter)),
            'fw_variant': [param_dict['fw_variant']] * n_fw_iter,
            'seed': [param_dict['seed']] * n_fw_iter,
            'base_dist': [param_dict['base_dist']] * n_fw_iter,
            'iter0': [param_dict['iter0']] * n_fw_iter,
            'linit_fixed': [param_dict['linit_fixed']] * n_fw_iter,
            'tau': [param_dict['exp_adafw']] * n_fw_iter,
            'eta': [param_dict['damping_adafw']] * n_fw_iter
        }
        run_df = pd.DataFrame(data)
        runs_df.append(run_df)

        cnt += 1
        if cnt % 500 == 0: debug(cnt, " runs processed")


    debug('seeds used ', list(seeds_used))
    df = pd.concat(runs_df)
    # mvl vs mvn is a modelling choice
    df = df.loc[df['base_dist'] == 'mvl']
    df.reset_index(inplace=True)
    #df = filter_adaptive(df, args.metric)

    ### Make plot ###
    #plot_base_dist(df, args.metric)
    plot_mvl(df, args.metric)
    #plot_mvl(df, args.metric, iter0_split=True)
    #plot_adaptive(df)
    plt.show()
    pass


def info():
    run_names = [
        d for d in os.listdir(args.datapath)
        if os.path.isdir(os.path.join(args.datapath, d))
    ]
    run_paths = [os.path.join(args.datapath, d) for d in run_names]

    def get_n_lines(filepath):
        if not os.path.isfile(filepath): return 0
        with open(filepath, 'r') as f:
            return len(list(f.readlines()))

    fixed_n, adafw_n, ada_afw_n, ada_pfw_n  = [], [], [], []
    for dn, dr in zip(run_names, run_paths):
        if dn.startswith('fixed'):
            fixed_n.append(get_n_lines(os.path.join(dr, 'roc.csv')))
        elif dn.startswith('adafw'):
            adafw_n.append(get_n_lines(os.path.join(dr, 'roc.csv')))
        elif dn.startswith('ada_afw'):
            ada_afw_n.append(get_n_lines(os.path.join(dr, 'roc.csv')))
        elif dn.startswith('ada_pfw'):
            ada_pfw_n.append(get_n_lines(os.path.join(dr, 'roc.csv')))

    fixed_n = np.asarray(fixed_n)
    print('fixed: median run length %d, number of runs %d' %
          (np.median(fixed_n), len(fixed_n)))
    adafw_n = np.asarray(adafw_n)
    print('adafw: median run length %d, number of runs %d' %
          (np.median(adafw_n), len(adafw_n)))
    ada_afw_n = np.asarray(ada_afw_n)
    print('ada_afw: median run length %d, number of runs %d' %
          (np.median(ada_afw_n), len(ada_afw_n)))
    ada_pfw_n = np.asarray(ada_pfw_n)
    print('ada_pfw: median run length %d, number of runs %d' %
          (np.median(ada_pfw_n), len(ada_pfw_n)))


if __name__ == "__main__":
    args.all_var = args.adaptive_var + ['fixed']
    #info()
    main()
