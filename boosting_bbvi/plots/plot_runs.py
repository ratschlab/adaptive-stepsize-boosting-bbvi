"""Collect run data, parse and summarize.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
from matplotlib.ticker import MaxNLocator


import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from boosting_bbvi.core.utils import eprint, debug

def parse_command(cmd):
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
    
    path: full path, assumes run.log present with first line being
        command
    """
    if os.path.isfile(path):
        cmd = open(path, 'r').readline()
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


    return res

def plot_mvl(df, y='roc', base_split=False, iter0_split=False):
    if not base_split: df = df.loc[df['base_dist'] == 'mvl']

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
    mvl_df = df.loc[df['base_dist'] == 'mvl']
    mvn_df = df.loc[df['base_dist'] == 'mvn']
    fix, ax = plt.subplots(1, 2)
    sns.lineplot(x='fw_iter', y=y, hue='fw_variant', data=mvl_df, ax=ax[0])
    sns.lineplot(x='fw_iter', y=y, hue='fw_variant', data=mvn_df, ax=ax[1])


def plot_adaptive(df, base_split=False, iter0_split=False):
    """Analyze the performance of adaptive variants.

    Args:
        base_split: keep only Laplace
        iter0_split: keep only vi
    """
    if not base_split: df = df.loc[df['base_dist'] == 'mvl']
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
    fig, ax = plt.subplots(3, 1)
    for i, fw_var in enumerate(['adafw', 'ada_pfw', 'ada_afw']):
        df_var = df.loc[df['fw_variant'] == fw_var]
        t_matrix = pd.crosstab(df_var.fw_iter, df_var.iter_type)
        #t_matrix.columns.names = ['']
        #t_matrix.reset_index(inplace=True)
        #sns.barplot(x='fw_iter', hue='iter_info', data=t_matrix)
        t_matrix.plot(kind='bar', stacked=True, title=fw_var, ax=ax[i])

    # plot lipschitz estimates
    pass


def main(datapath):
    run_names = [
        d for d in os.listdir(datapath)
        if os.path.isdir(os.path.join(datapath, d))
    ]
    run_paths = [os.path.join(datapath, d) for d in run_names]

    df = pd.DataFrame()
    for nr, dr in zip(run_names, run_paths):
        param_dict = parse_log(nr, os.path.join(dr, 'run.log'))

        rocs_filename = os.path.join(dr, 'roc.csv')
        with open(rocs_filename, 'r') as f:
            rocs = [float(r.strip()) for r in f.readlines()]

        # set number of iterations to plot
        n_fw_iter = param_dict.get('n_fw_iter', len(rocs))

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

        n_fw_iter = max([
            len(rocs),
            len(ll_trains),
            len(ll_tests),
            len(elbos),
            len(gaps),
            len(iter_types), n_fw_iter
        ])

        # Padding to make all lists of equal length
        if len(rocs) < n_fw_iter:
            rocs.extend([None] * (n_fw_iter - len(rocs)))
        if len(ll_trains) < n_fw_iter:
            ll_trains.extend([None] * (n_fw_iter - len(ll_trains)))
        if len(ll_tests) < n_fw_iter:
            ll_tests.extend([None] * (n_fw_iter - len(ll_tests)))
        if len(elbos) < n_fw_iter:
            elbos.extend([None] * (n_fw_iter - len(elbos)))
        if len(gaps) < n_fw_iter:
            gaps.extend([None] * (n_fw_iter - len(gaps)))
        if len(iter_types) < n_fw_iter:
            iter_types.extend([None] * (n_fw_iter - len(iter_types)))

        data = {
            'roc': rocs,
            'll_train': ll_trains,
            'll_test': ll_tests,
            'elbo': elbos,
            'gap': gaps,
            'iter_type': iter_types,
            'fw_iter': list(range(n_fw_iter)),
            'fw_variant': [param_dict['fw_variant']] * n_fw_iter,
            'base_dist': [param_dict['base_dist']] * n_fw_iter,
            'iter0': [param_dict['iter0']] * n_fw_iter,
            'linit_fixed': [param_dict['linit_fixed']] * n_fw_iter,
            'tau': [param_dict['exp_adafw']] * n_fw_iter,
            'eta': [param_dict['damping_adafw']] * n_fw_iter
        }
        run_df = pd.DataFrame(data)
        df = df.append(run_df, ignore_index=True)

    ### Make plot ###
    plot_mvl(df, 'roc')
    plot_mvl(df, 'roc', iter0_split=True)
    #plot_mvl(df, 'elbo')
    #plot_adaptive(df)
    plt.show()
    pass


def info(datapath):
    run_names = [
        d for d in os.listdir(datapath)
        if os.path.isdir(os.path.join(datapath, d))
    ]
    run_paths = [os.path.join(datapath, d) for d in run_names]

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
    #info(sys.argv[1])
    main(sys.argv[1])
