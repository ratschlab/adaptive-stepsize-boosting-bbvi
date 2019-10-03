"""Collect run data, parse and summarize.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
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
    choices=['adafw', 'ada_pfw', 'ada_afw', 'line_search', 'none'])
parser.add_argument("--exp", default='info', choices=['info', 'blr', 'bmf'])
parser.add_argument(
    "--select_run",
    default='all',
    choices=['all', 'hp_max', 'hp_mean', 'hp_median', 'run_mean', 'run_median'],
    help="""method to select adaptive runs,
                all: all runs
                hp_max: choose the hp which gave the overall best metric
                hp_mean: hp which gave overall best mean over all seeds
                seed_best: for each seed, choose the best hp
                run_mean: select run with best mean metric
                run_median: select run with best median metric
                """)
parser.add_argument(
    '--datapath', type=str, help='directory containing run data')
parser.add_argument(
    '--outfile',
    type=str,
    default='stdout',
    help='path to store plots, stdout to show')
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
            # FIXME only works with --restore , --norestore
            # not with --restore0 --restore true etc.
            elif token in ["restore", "norestore"]: # boolean
                param = token
                value = (token == "restore")
                tokens = tokens[1:]
            else:                   # --n_monte_carlo_samples 1000
                param = token
                value = tokens[1].strip()
                tokens = tokens[2:] # right shift by 2

            result[param] = value
        else:
            eprint("%s not parsed" % token)
            tokens = tokens[1:]
            return None

    return result


def parse_run(run):
    """Parse a run.
        Example - fixed, ada_pfw_1_48,
        fixed_4, adafw_0_4
    """
    order_tok = ['fw_variant', 'i', 'counter']
    idx_tok = 0
    tokens = run.split('_')
    result = {}
    while tokens:
        param = order_tok[idx_tok]
        if param == 'fw_variant' and (tokens[0] == 'ada' or
                                      tokens[0] == 'line'):
            value = "%s_%s" % (tokens[0], tokens[1])
            tokens = tokens[2:]
        else:
            value = tokens[0]
            tokens = tokens[1:]
        result[param] = value
        idx_tok += 1

    if 'i' not in result: result['i'] = 0
    if 'counter' not in result: result['counter'] = 0
    result['i'] = int(result['i'])
    result['counter'] = int(result['counter'])
    return result


def parse_log(run_name, path):
    """Parse log
    
    run_name: name of directory
    path: full path, assumes run.log present with first line being
        command
    """
    if os.path.isfile(path):
        with open(path, 'r') as f:
            while args.cluster:
                if f.readline().startswith('# LSBATCH: User input'):
                    break
            cmd = f.readline()
            res = parse_command(cmd)
            if res is None:
                eprint(path, 'not parsed')
                raise ValueError

    res_name = parse_run(run_name)
    res['counter'] = res_name['counter'] # hp counter

    # Convert numerical values
    for key in res:
        if key in [
                'n_fw_iter', 'LMO_iter', 'n_monte_carlo_samples',
                'adafw_MAXITER'
        ]:
            res[key] = int(res[key])
        elif key in ['linit_fixed', 'exp_adafw', 'damping_adafw']:
            res[key] = float(res[key])

    # Fixed variant does not have lipschitz estimate
    if 'linit_fixed' not in res:
        # linit_fixed is also used as initial step size in line search
        # NOTE bad coding practice
        res['linit_fixed'] = (0.001
                              if (res['fw_variant'].startswith('ada') or
                                  res['fw_variant'] == 'line_search') else None)
    if 'exp_adafw' not in res:
        res['exp_adafw'] = (2.0
                              if res['fw_variant'].startswith('ada') else None)
    if 'damping_adafw' not in res:
        res['damping_adafw'] = (0.99
                              if res['fw_variant'].startswith('ada') else None)

    if 'seed' not in res:
        res['seed'] = 0
    return res

def plot_iteration(df, y='elbo', fw_split='split', ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    # Integer x axis
    #ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    #df.rename(columns={'fw_variant': 'step variant'}, inplace=True)

    if fw_split == 'split':
        #g = sns.FacetGrid(df, col="fw_variant")
        #debug(df['linit_fixed'].unique())
        #debug(df.groupby(['counter'], as_index=False).agg({ 'linit_fixed': ['first']}))
        g = sns.FacetGrid(df, col="fw_variant")
        g.map(sns.lineplot, 'fw_iter', y, 'counter', ci=None)
        g.add_legend()
    elif fw_split == 'range':
        # raw matplotlib
        df_group = df.groupby(
            ['fw_variant', 'fw_iter'], as_index=False).agg({
                y: ['mean', 'std', 'median'],
            })
        df_group.columns = ['fw_variant', 'fw_iter', '%s_mean' % y, "%s_std" % y, "%s_median" % y]
        x = np.arange(1, args.n_fw_iter + 1)

        line_search_mean = df_group.loc[df_group['fw_variant'] == 'line_search']["%s_mean" % y].values
        line_search_std = df_group.loc[df_group['fw_variant'] == 'line_search']["%s_std" % y].values
        ax.plot(x, line_search_mean, label='line-search', color='#4878D0')
        ax.fill_between(x, line_search_mean - line_search_std, line_search_mean + line_search_std, alpha=0.2, facecolor='#4878D0')

        fixed_mean = df_group.loc[df_group['fw_variant'] == 'fixed']["%s_mean" % y].values
        fixed_std = df_group.loc[df_group['fw_variant'] == 'fixed']["%s_std" % y].values
        ax.plot(x, fixed_mean, label='Fixed', color='#EE854A')
        ax.fill_between(x, fixed_mean - fixed_std, fixed_mean + fixed_std, alpha=0.2, facecolor='#EE854A')

        adafw_mean = df_group.loc[df_group['fw_variant'] == 'adafw']["%s_mean" % y].values
        adafw_std = df_group.loc[df_group['fw_variant'] == 'adafw']["%s_std" % y].values
        ax.plot(x, adafw_mean, label='AdaFW', color='#6ACC64')
        ax.fill_between(x, adafw_mean - adafw_std, adafw_mean + adafw_std, alpha=0.2, facecolor='#6ACC64')

        ada_afw_mean = df_group.loc[df_group['fw_variant'] == 'ada_afw']["%s_mean" % y].values
        ada_afw_std = df_group.loc[df_group['fw_variant'] == 'ada_afw']["%s_std" % y].values
        ax.plot(x, ada_afw_mean, label='AdaAFW', color='#D65F5F')
        ax.fill_between(x, ada_afw_mean - ada_afw_std, ada_afw_mean + ada_afw_std, alpha=0.2, facecolor='#D65F5F')

        ada_pfw_mean = df_group.loc[df_group['fw_variant'] == 'ada_pfw']["%s_mean" % y].values
        ada_pfw_std = df_group.loc[df_group['fw_variant'] == 'ada_pfw']["%s_std" % y].values
        ax.plot(x, ada_pfw_mean, label='AdaPFW', color='#956CB4')
        ax.fill_between(x, ada_pfw_mean - ada_pfw_std, ada_pfw_mean + ada_pfw_std, alpha=0.2, facecolor='#956CB4')

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
        ax.set(ylabel='ELBO', xlabel='iteration')
    else:
        g = sns.lineplot(x='fw_iter', y=y, hue='fw_variant', data=df, ax=ax)
        #g = sns.lineplot(x='fw_iter', y=y, hue='fw_variant', data=df, ax=ax, estimator=np.median)
        #g = sns.lineplot(x='ctime', y=y, hue='fw_variant', data=df, ax=ax)
        handles, _ = g.get_legend_handles_labels()
        #row_dict = {'adafw': 'adaFW', 'line_search': 'line-search', 'ada_afw': 'adaAFW',
        #        'ada_pfw': 'adaPFW', 'fixed': 'fixed', 'fw_variant': 'step variant'}

        #plt.legend(loc='lower left')
        ax.set(ylabel='ELBO', xlabel='iteration')
        #ax.set_yscale('symlog')
        #ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02, 1., 0.102),
        #        mode="expand", borderaxespad=0., ncol=2)
        g.legend(handles[1:], ['step variant', 'adaAFW', 'adaPFW', 'adaFW', 'fixed', 'line-search'][1:],
                loc='lower left', bbox_to_anchor=(0, 1.02, 1., 0.102),
                mode="expand", borderaxespad=0., ncol=2)

    if args.outfile == 'stdout':
        plt.show()
    else:
        plt.tight_layout()
        ax.get_figure().savefig(args.outfile)


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


def make_table(df, metric):
    """Compute metrics."""
    result = df.groupby(['fw_variant'], as_index=False).agg(
            {metric: ['mean', 'std', 'median', 'max']})
    pd.set_option('precision',4)
    print(result, '\n')


def make_iter_table(df, metric, return_latex=False):
    """Compute metrics and select best iteration."""

    if metric in ['roc', 'll_train', 'll_test', 'elbo']:
        best = 'max'
        maximize = True
    elif metric in ['mse_train', 'mse_test']:
        best = 'min'
        maximize = False
    else: raise NotImplementedError('metric %s not supported ' % metric)

    # aggregate over random seeds
    if args.exp == 'bmf':
        df_group = df.groupby(
            ['fw_variant', 'counter', 'fw_iter'], as_index=False).agg({
                metric: ['mean', 'std', best, 'median'],
                #'mse_train': ['mean', 'std'], 'll_train': ['mean', 'std'],
                #'mse_train': ['mean', 'std'], 'mse_test': ['mean', 'std'],
                'mse_test': ['mean', 'std', 'median'], 'll_train': ['mean', 'std', 'median'],
            })
        df_group.columns = [
            'fw_variant', 'counter', 'best_fw_iter', '%s_mean' % metric,
            '%s_std' % metric, '%s_%s' % (metric, best), '%s_median' % metric,
            #'mse_train_mean', 'mse_train_std', 'll_train_mean', 'll_train_std'
            #'mse_train_mean', 'mse_train_std', 'mse_test_mean', 'mse_test_std'
            'mse_test_mean', 'mse_test_std', 'mse_test_median', 'll_train_mean', 'll_train_std', 'll_train_median'
        ]
    elif args.exp == 'blr':
        df_group = df.groupby(
            ['fw_variant', 'counter', 'fw_iter'], as_index=False).agg({
                metric: ['mean', 'std', best, 'median'],
                #'ll_test': ['mean', 'std'], 'll_train': ['mean', 'std'],
                'll_test': ['mean', 'std', 'median'], 'roc': ['mean', 'std', 'median'],
            })
        df_group.columns = [
            'fw_variant', 'counter', 'best_fw_iter', '%s_mean' % metric,
            '%s_std' % metric, '%s_%s' % (metric, best), '%s_median' % metric,
            #'ll_test_mean', 'll_test_std', 'll_train_mean', 'll_train_std'
            'll_test_mean', 'll_test_std', 'll_test_median', 'roc_mean', 'roc_std', 'roc_median'
        ]
    else:
        raise NotImplementedError

    # metric to compute best iteration
    # ex - roc for blr, mse_test for bmf
    # median or mean for best iteration
    if args.select_run == 'run_mean': eval_metric = '%s_mean' % metric
    elif args.select_run == 'run_median': eval_metric = '%s_median' % metric
    else: raise NotImplementedError('%s not supported' % args.select_run)

    # find best iteration per hp
    #if maximize:
    #res_hp = df_group.groupby(['fw_variant', 'counter'], sort=False)[eval_metric].max()
    #else:
    #res_hp = df_group.groupby(['fw_variant', 'counter'], sort=False)[eval_metric].min()
    #debug(res_hp)

    # find best hp, iteration per fw_variant
    if maximize:
        best_counter_idx = df_group.groupby(['fw_variant'], sort=False)[eval_metric].idxmax()
    else:
        best_counter_idx = df_group.groupby(['fw_variant'], sort=False)[eval_metric].idxmin()

    res = df_group.loc[best_counter_idx]
    debug(res, '\n')
    # for every fw_variant, filter and use only the best hp
    fw_vars = res['fw_variant'].tolist()
    best_hp = res['counter'].tolist()
    filtered_runs = []
    for fw_var, best_counter in zip(fw_vars, best_hp):
        debug(fw_var, ' counter ', best_counter)
        run_filtered = df.loc[(df['counter'] == best_counter) &
                              (df['fw_variant'] == fw_var)]
        filtered_runs.append(run_filtered)
    df_best = pd.concat(filtered_runs).reset_index()

    if return_latex:
        time_res = df_best.groupby(
            ['fw_variant'], as_index=False).agg({
                'time': ['mean', 'std', 'median']
            })
        time_res.columns = ['fw_variant', 'time_mean', 'time_std', 'time_median']
        # keeping index same to can add columns later
        res.reset_index(inplace=True)
        time_res.reset_index(inplace=True)
        # res and time_res should've same order of fw variants
        assert res['fw_variant'].tolist() == time_res['fw_variant'].tolist()
        res['time_mean'] = time_res['time_mean']
        res['time_std'] = time_res['time_std']
        res['time_median'] = time_res['time_median']
        if args.exp == 'blr':
            #metrics_to_print = [metric, 'll_train', 'time']
            metrics_to_print = [metric, 'roc', 'time']
        elif args.exp == 'bmf':
            #metrics_to_print = [metric, 'mse_train', 'll_train']
            metrics_to_print = [metric, 'mse_test', 'll_train']
        else:
            raise NotImplementedError
        print('fw_variant &', " &".join(metrics_to_print), end="\\\\\n")
        print("\hline")
        for _, row in res.iterrows():
            print(row['fw_variant'], end=" ")
            for m in metrics_to_print:
                if m == metric:
                    print("& %.3f" % (row["%s_median" % m]),
                        end="")
                else:
                    print("& %.3f $\pm$ %.3e" % (row["%s_mean" % m], row["%s_std" % m]),
                        end=" ")
            print('\\\\')

    return res, df_best


def filter_runs(df_all, df_best, metric='roc'):
    """Filter runs to process based on run selection strategy.
    
    Using df_best, it finds the parameters to use depending on the
    selection strategy. Assuming each counter refers to one particular
    parameter configuration, grouping by counter is much better than
    by param set {mu, eta, linit} for adaptive and {binit} for line search
    """
    # group by variant,parameters
    #if metric in ['roc', 'elbo', 'll_test', 'll_train']: # maximize
    #    idx = df_best.groupby(['fw_variant', 'counter'], sort=False)[metric].transform(max) == df_best[metric]
    #    print(df_best[idx])
    #elif metric in ['mse_train', 'mse_test', 'kl']: # minimize
    #    raise NotImplementedError()

    if args.select_run != 'all':
        filtered_all = [df_all.loc[df_all['fw_variant'] == 'fixed']]
        filtered_best = [df_best.loc[df_best['fw_variant'] == 'fixed']]
        for fw_var in args.adaptive_var:
            df_all_var = df_all.loc[df_all['fw_variant'] == fw_var]
            df_best_var = df_best.loc[df_best['fw_variant'] == fw_var]
            res = df_best_var.groupby(
                ['counter'], as_index=False).agg({
                    metric: ['mean', 'max', 'median']
                })
            res.columns = [
                'counter',
                '%s_mean' % metric,
                '%s_max' % metric,
                '%s_median' % metric
            ]
            if args.select_run == 'hp_max':
                best_idx = res['%s_max' % metric].idxmax()
            elif args.select_run == 'hp_mean':
                best_idx = res['%s_mean' % metric].idxmax()
            elif args.select_run == 'hp_median':
                best_idx = res['%s_median' % metric].idxmax()
            else:
                raise NotImplementedError('%s not implemented' % args.select_run)
            best_counter = res.loc[best_idx]['counter']
            debug(res.shape, ' param configs best is ', best_counter)
            all_var_filtered = df_all_var.loc[df_all_var['counter'] == best_counter]
            best_var_filtered = df_best_var.loc[df_best_var['counter'] == best_counter]
            filtered_all.append(all_var_filtered)
            filtered_best.append(best_var_filtered)

        df_all = pd.concat(filtered_all).reset_index()
        df_best = pd.concat(filtered_best).reset_index()
    return df_all, df_best


def pad_or_crop(a, n):
    """Fix len of list a to n"""
    if len(a) > n:
        a = a[:n]
    elif len(a) < n:
        a.extend([None] * (n - len(a)))
    return a


def blr():
    run_names = [
        d for d in os.listdir(args.datapath)
        if os.path.isdir(os.path.join(args.datapath, d))
    ]
    run_paths = [os.path.join(args.datapath, d) for d in run_names]

    df = pd.DataFrame()
    cnt = 0
    # runs_df contains all iterations
    runs_df = []
    # best runs contains only the best iteration for every run
    #NOTE FIXME selecting the best iteration early on can result in
    # outlier estimates. Select best iteration based on the one which
    # gives best mean/median over random seeds (not hp)
    best_runs = []
    seeds_used = set()
    for nr, dr in zip(run_names, run_paths):
        param_dict = parse_log(nr, os.path.join(dr, 'run.log'))
        if param_dict['fw_variant'] not in args.all_var:
            continue
        seeds_used.add(param_dict['seed'])

        rocs_filename = os.path.join(dr, 'roc.csv')
        with open(rocs_filename, 'r') as f:
            rocs = [float(r.strip()) for r in f.readlines()]

        # set number of iterations to plot
        #n_fw_iter = param_dict.get('n_fw_iter', len(rocs))
        n_fw_iter = args.n_fw_iter

        ll_train_filename = os.path.join(dr, 'll_train.csv')
        with open(ll_train_filename, 'r') as f:
            ll_trains = [float(r.split(',')[0]) for r in f.readlines()]

        ll_test_filename = os.path.join(dr, 'll_test.csv')
        with open(ll_test_filename, 'r') as f:
            ll_tests = [float(r.split(',')[0]) for r in f.readlines()]

        elbos_filename = os.path.join(dr, 'elbos.csv')
        def get_elbo(token):
            # elbo is sometimes elbo or elbo_t,KLqploss
            token = token.strip()
            if ',' in token: token = token.split(',')[0]
            return float(token)
        with open(elbos_filename, 'r') as f:
            elbos = [get_elbo(e) for e in f.readlines()]

        gap_filename = os.path.join(dr, 'gap.csv')
        with open(gap_filename, 'r') as f:
            gaps = [float(r.strip()) for r in f.readlines()]

        times_filename = os.path.join(dr, 'times.csv')
        with open(times_filename, 'r') as f:
            # NOTE: time[0] is always 0.
            times = [float(r.strip()) for r in f.readlines()]

        ctimes = np.cumsum(times)

        iter_info_filename = os.path.join(dr, 'iter_info.txt')
        if os.path.isfile(iter_info_filename):
            with open(iter_info_filename, 'r') as f:
                iter_types = [r.strip() for r in f.readlines()]
        else:
            # fixed or line_search
            iter_types = [param_dict['fw_variant']] * n_fw_iter

        rocs = pad_or_crop(rocs, n_fw_iter)
        ll_trains = pad_or_crop(ll_trains, n_fw_iter)
        ll_tests = pad_or_crop(ll_tests, n_fw_iter)
        elbos = pad_or_crop(elbos, n_fw_iter)
        gaps = pad_or_crop(gaps, n_fw_iter)
        iter_types = pad_or_crop(iter_types, n_fw_iter)
        times = pad_or_crop(times, n_fw_iter)
        ctimes = pad_or_crop(times, n_fw_iter)

        data = {
            'roc': rocs,
            'll_train': ll_trains,
            'll_test': ll_tests,
            'elbo': elbos,
            'gap': gaps,
            'iter_type': iter_types,
            'time': times,
            'ctime': ctimes,
            'fw_iter': list(range(n_fw_iter)),
            'fw_variant': [param_dict['fw_variant']] * n_fw_iter,
            'seed': [param_dict['seed']] * n_fw_iter,
            'linit_fixed': [param_dict['linit_fixed']] * n_fw_iter,
            'tau': [param_dict['exp_adafw']] * n_fw_iter,
            'eta': [param_dict['damping_adafw']] * n_fw_iter,
            'counter': [param_dict['counter']] * n_fw_iter,
        }
        run_df = pd.DataFrame(data)
        runs_df.append(run_df)
        # If we choose the best iteration in a FW training algorithms
        # with high variance might get the best results even after
        # performing well overall
        best_run = run_df.loc[run_df['roc'].idxmax()]
        #best_run = run_df.tail(1)
        best_runs.append(best_run)

        cnt += 1
        if cnt % 500 == 0: debug(cnt, " runs processed")

    debug('seeds used ', list(seeds_used))
    all_run_df = pd.concat(runs_df)
    all_run_df.reset_index(inplace=True)
    debug(all_run_df.shape)
    # index and fw_iter have same values
    del all_run_df['index']
    # if best_runs is df then concat, if pd.Series then DataFrame()
    best_run_df = pd.DataFrame(best_runs, columns=best_runs[0].index)
    #best_run_df = pd.concat(best_runs)
    best_run_df.reset_index(inplace=True)
    del best_run_df['index']

    # NOTE: best runs from each individual iteration are too noisy
    # compute best iter after aggregating over all seeds
    #res, best_run_df = make_iter_table(all_run_df, 'roc', True)
    res, best_run_df = make_iter_table(all_run_df, 'll_train', True)
    debug('shape of runs', all_run_df.shape, best_run_df.shape)

    # get filtered runs here
    #all_run_df, best_run_df = filter_runs(all_run_df, best_run_df, 'roc')

    ### Make plot ###
    #plot_iteration(best_run_df, 'elbo')
    plot_iteration(all_run_df, 'elbo', fw_split='range')
    #make_table(best_run_df, 'roc')
    #make_table(all_run_df, 'time')
    #make_table(best_run_df, 'time')
    #make_table(best_run_df, 'll_train')
    #make_table(best_run_df, 'll_test')


def bmf():
    """Plot Bayesian Matrix Factorization data"""
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
        logfile = os.path.join(dr, 'run.log')
        # did the job finish?
        if not os.path.isfile(logfile): continue
        param_dict = parse_log(nr, logfile)
        if param_dict['fw_variant'] not in args.all_var: continue
        seeds_used.add(param_dict.get('seed', 0))

        mse_test_filename = os.path.join(dr, 'mse_test.csv')
        with open(mse_test_filename, 'r') as f:
            mse_tests = [float(r.strip()) for r in f.readlines()]

        n_fw_iter = args.n_fw_iter

        mse_train_filename = os.path.join(dr, 'mse_train.csv')
        with open(mse_train_filename, 'r') as f:
            mse_trains = [float(r.strip()) for r in f.readlines()]

        ll_train_filename = os.path.join(dr, 'll_train.csv')
        with open(ll_train_filename, 'r') as f:
            ll_trains = [float(r.split(',')[0]) for r in f.readlines()]

        ll_test_filename = os.path.join(dr, 'll_test.csv')
        with open(ll_test_filename, 'r') as f:
            ll_tests = [float(r.split(',')[0]) for r in f.readlines()]

        elbos_filename = os.path.join(dr, 'elbos.csv')
        def get_elbo(token):
            # elbo is sometimes elbo or elbo_t,KLqploss
            token = token.strip()
            if ',' in token: token = token.split(',')[0]
            return float(token)
        with open(elbos_filename, 'r') as f:
            elbos = [get_elbo(e) for e in f.readlines()]

        gap_filename = os.path.join(dr, 'gap.csv')
        with open(gap_filename, 'r') as f:
            gaps = [float(r.strip()) for r in f.readlines()]

        iter_info_filename = os.path.join(dr, 'iter_info.txt')
        if os.path.isfile(iter_info_filename):
            with open(iter_info_filename, 'r') as f:
                iter_types = [r.strip() for r in f.readlines()]
        else:
            iter_types = [param_dict['fw_variant']] * n_fw_iter

        times_filename = os.path.join(dr, 'times.csv')
        with open(times_filename, 'r') as f:
            # NOTE: time[0] is always 0.
            times = [float(r.strip()) for r in f.readlines()]

        gamma_filename = os.path.join(dr, 'steps.csv')
        with open(gamma_filename, 'r') as f:
            gammas = [float(r.strip()) for r in f.readlines()]

        mse_tests = pad_or_crop(mse_tests, n_fw_iter)
        mse_trains = pad_or_crop(mse_trains, n_fw_iter)
        ll_trains = pad_or_crop(ll_trains, n_fw_iter)
        ll_tests = pad_or_crop(ll_tests, n_fw_iter)
        elbos = pad_or_crop(elbos, n_fw_iter)
        gaps = pad_or_crop(gaps, n_fw_iter)
        iter_types = pad_or_crop(iter_types, n_fw_iter)
        times = pad_or_crop(times, n_fw_iter)

        data = {
            'mse_test': mse_tests,
            'mse_train': mse_trains,
            'll_train': ll_trains,
            'll_test': ll_tests,
            'elbo': elbos,
            'gap': gaps,
            'time': times,
            'iter_type': iter_types,
            'fw_iter': list(range(n_fw_iter)),
            'fw_variant': [param_dict['fw_variant']] * n_fw_iter,
            'seed': [param_dict['seed']] * n_fw_iter,
            'linit_fixed': [param_dict['linit_fixed']] * n_fw_iter,
            'tau': [param_dict['exp_adafw']] * n_fw_iter,
            'eta': [param_dict['damping_adafw']] * n_fw_iter,
            'counter': [param_dict['counter']] * n_fw_iter
        }
        run_df = pd.DataFrame(data)
        runs_df.append(run_df)

        cnt += 1
        if cnt % 500 == 0: debug(cnt, " cnt processed")

    df = pd.concat(runs_df).reset_index()
    del df['index']

    res, best_run_df = make_iter_table(df, 'mse_train', True)
    debug('shape of runs', df.shape, best_run_df.shape)

    ### Make plot
    #plot_iteration(best_run_df, 'elbo')
    #plot_iteration(best_run_df, 'mse_test')


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

    fixed_n, line_search_n, adafw_n, ada_afw_n, ada_pfw_n  = [], [], [], [], []
    for dn, dr in zip(run_names, run_paths):
        if dn.startswith('fixed'):
            fixed_n.append(
                get_n_lines(os.path.join(dr, '{}.csv'.format(args.metric))))
        if dn.startswith('line_search'):
            line_search_n.append(
                get_n_lines(os.path.join(dr, '{}.csv'.format(args.metric))))
        elif dn.startswith('adafw'):
            adafw_n.append(
                get_n_lines(os.path.join(dr, '{}.csv'.format(args.metric))))
        elif dn.startswith('ada_afw'):
            ada_afw_n.append(
                get_n_lines(os.path.join(dr, '{}.csv'.format(args.metric))))
        elif dn.startswith('ada_pfw'):
            ada_pfw_n.append(
                get_n_lines(os.path.join(dr, '{}.csv'.format(args.metric))))

    print('fixed: number of runs %d' % (len(fixed_n)))
    if fixed_n:
        fixed_n = np.asarray(fixed_n)
        print('fixed: median run length %d, min %d' % (np.median(fixed_n),
                                                       np.min(fixed_n)))

    print('line search: number of runs %d' % (len(line_search_n)))
    if line_search_n:
        line_search_n = np.asarray(line_search_n)
        print('line_search: median run length %d, min %d' %
              (np.median(line_search_n), np.min(line_search_n)))

    print('adafw: number of runs %d' % (len(adafw_n)))
    if adafw_n:
        adafw_n = np.asarray(adafw_n)
        print('adafw: median run length %d, min %d' % (np.median(adafw_n),
                                                       np.min(adafw_n)))

    print('ada_afw: number of runs %d' % (len(ada_afw_n)))
    if ada_afw_n:
        ada_afw_n = np.asarray(ada_afw_n)
        print('ada_afw: median run length %d' % (np.median(ada_afw_n)))

    print('ada_pfw: number of runs %d' % (len(ada_pfw_n)))
    if ada_pfw_n:
        ada_pfw_n = np.asarray(ada_pfw_n)
        print('ada_pfw: median run length %d' % (np.median(ada_pfw_n)))


if __name__ == "__main__":
    args.all_var = args.adaptive_var + ['fixed']
    pd.set_option('precision', 3)
    if args.exp == 'info':
        info()
    elif args.exp == 'blr':
        blr()
    elif args.exp == 'bmf':
        bmf()
