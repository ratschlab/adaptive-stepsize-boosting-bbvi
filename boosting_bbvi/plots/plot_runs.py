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
        if param == 'fw_variant' and tokens[0].startswith('ada'):
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

    return res

def plot_mvl(df, y='roc', iter0_split=True):
    mvl_df = df.loc[df['base_dist'] == 'mvl']
    fix, ax = plt.subplots()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if not iter0_split:
        # show only vi iter0
        mvl_df = mvl_df.loc[mvl_df['iter0'] == 'vi']

    if iter0_split:
        sns.lineplot(
            x='fw_iter', y=y, hue='fw_variant', style='iter0', data=mvl_df)
    else:
        sns.lineplot(
            x='fw_iter', y=y, hue='fw_variant', data=mvl_df)


def plot_base_dist(df, y='roc'):
    mvl_df = df.loc[df['base_dist'] == 'mvl']
    mvn_df = df.loc[df['base_dist'] == 'mvn']
    fix, ax = plt.subplots(1, 2)
    sns.lineplot(x='fw_iter', y=y, hue='fw_variant', data=mvl_df, ax=ax[0])
    sns.lineplot(x='fw_iter', y=y, hue='fw_variant', data=mvn_df, ax=ax[1])


def plot_adaptive():
    # plot linit with style
    # plot eta, tau best results heatmap
    # plot iter info
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

        n_fw_iter = param_dict.get('n_fw_iter', len(rocs))
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

        data = {
            'roc': rocs,
            'll_train': ll_trains,
            'll_test': ll_tests,
            'elbo': elbos,
            'gap': gaps,
            'fw_iter': list(range(n_fw_iter)),
            'fw_variant': [param_dict['fw_variant']] * n_fw_iter,
            'base_dist': [param_dict['base_dist']] * n_fw_iter,
            'iter0': [param_dict['iter0']] * n_fw_iter
        }
        run_df = pd.DataFrame(data)

        df = df.append(run_df, ignore_index=True)

    ### Make plot ###
    #plot_mvl(df, 'roc', True)
    #plot_mvl(df, 'elbo', False)
    #plot_mvl(df, 'gap', False)
    plt.show()
    pass


if __name__ == "__main__":
    main(sys.argv[1])
