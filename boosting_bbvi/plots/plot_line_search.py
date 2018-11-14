"""Plot gradients and gamma for line search.

Usage:
    python plot_line_search.py \
        --outdir=stdout \
        --title=test run \
        --runs=path/line_search_samples_10.npy,path/line_search_samples_35.npy
"""

from __future__ import print_function

import os
import sys
import numpy as np
import scipy.stats as stats
import plot_utils as utils
from plot_utils import eprint, debug, plot_hline

from absl import app
from absl import flags

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FLAGS = flags.FLAGS
flags.DEFINE_string('outdir', '/tmp/',
                    'Output file to store plot into, set to stdout to '
                    'only show the plots and not save them')
flags.DEFINE_string('outfile', 'mixtures.png', 'name of the plot file')
flags.DEFINE_string('title', 'my awesome figure', '')
flags.DEFINE_enum('metric', 'gamma', ['E_s', 'E_q', 'gamma'], 'metric to plot')
flags.DEFINE_string('extra', '', 'extra info on each plot, depends on metric')
flags.DEFINE_list('runs', [], 'comma separated list of run files to load')

flags.DEFINE_list('labels', [], 'list of labels for the plots')
flags.DEFINE_list('styles', ['+', '-', '.', '-'], 'styles for each plot')

def load_dataframe_for_run(runfile, offset=0):
    """Load a run and create a DataFrame from metrics.
    
    Args:
        runfile: path to load the run from
        offset: start iteration for plotting
    Returns:
        pandas DataFrame
    """
    data = np.load(runfile)
    # (n_line_search_iter, n_line_search_samples, 1)
    e_s = np.array([d['E_s'] for d in data])[offset:]
    e_q = np.array([d['E_q'] for d in data])[offset:]
    # (n_line_search_iter)
    gamma = np.array([d['gamma'] for d in data])[offset:]
    n_line_search_samples = e_s.shape[1]
    debug('line search samples %d' % n_line_search_samples)
    n_line_search_iter = e_s.shape[0]
    # (n_line_search_iter)
    iter_nos = np.arange(offset, offset + n_line_search_iter)

    # construct flattened columns for dataframe
    e_s_flat = e_s.flatten()
    e_q_flat = e_q.flatten()
    gamma_flat = np.repeat(gamma, n_line_search_samples)
    iters_flat = np.repeat(iter_nos, n_line_search_samples)
    line_search_samples = np.repeat(n_line_search_samples,
            n_line_search_samples * n_line_search_iter)
    return pd.DataFrame({
        'E_s': e_s_flat,
        'E_q': e_q_flat,
        'gamma': gamma_flat,
        'iterations': iters_flat,
        'n_samples': line_search_samples
    })


def main():
    # first iteration of E_q has very high variance
    offset = 1 if FLAGS.metric == 'E_q' else 0
    df = pd.concat(
        [load_dataframe_for_run(runfile, offset) for runfile in FLAGS.runs])

    sns.set_style("darkgrid")
    # NOTE: lineplot is messing up the hue numbers
    if FLAGS.metric == 'gamma':
        ax = sns.catplot(
            x="iterations",
            y=FLAGS.metric,
            hue='n_samples',
            kind='point',
            data=df)
        try:
            true_gamma = float(FLAGS.extra)
            ax.map_dataframe(
                plot_hline,
                y=true_gamma,
                marker='x',
                linewidth=4.0,
                c='black')
        except ValueError:
            pass
    else:
        ax = sns.catplot(
            x="iterations",
            y=FLAGS.metric,
            hue="n_samples",
            kind="point",
            data=df)
        #ax = sns.lineplot(
        #    x="iterations",
        #    y=FLAGS.metric,
        #    hue="n_samples",
        #    style="n_samples",
        #    data=df)

    if FLAGS.outdir == 'stdout':
        plt.show()
    else:
        outname = os.path.join(os.path.expanduser(FLAGS.outdir), FLAGS.outfile)
        ax.savefig(outname)


if __name__ == "__main__":
    FLAGS(sys.argv)
    main()
