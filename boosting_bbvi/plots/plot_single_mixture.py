"""
Plot multiple mixture models

Usage:
    python plots/plot_single_mixture.py \
            --outdir=/tmp \
            --title=single mixture \
            --target=/path/target_dist.npz \
            --qt=/path/qt_iter25.npz,/path/qt_latest.npz \
            --labels=iter25,latest
"""

from __future__ import print_function

import os
import sys
import numpy as np
import scipy.stats as stats

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from boosting_bbvi.core.utils import eprint, debug

from absl import app
from absl import flags

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['lines.color'] = 'blue'

import tensorflow as tf
from edward.models import Categorical, MultivariateNormalDiag, Normal, Mixture

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from boosting_bbvi.core.infinite_mixture import InfiniteMixtureScipy

FLAGS = flags.FLAGS
flags.DEFINE_string('outdir', '/tmp/',
                    'Output file to store plot into, set to stdout to '
                    'only show the plots and not save them')
flags.DEFINE_string('outfile', 'mixtures.png', 'name of the plot file')
flags.DEFINE_string('title', 'results', '')
flags.DEFINE_string('ylabel', 'y', '')
flags.DEFINE_string('xlabel', 'x', '')
flags.DEFINE_string('target', None, 'path to target.npz')
flags.mark_flag_as_required('target')
flags.DEFINE_list('qt', [], 'comma-separated list,of,qts to visualize')
flags.DEFINE_list('labels', [], 'list of labels to be associated with the qts')
flags.DEFINE_list('styles', [], 'styles for each plot')
flags.DEFINE_boolean('widegrid', False, 'range for the x-axis')
flags.DEFINE_boolean('grid2d', False, '3D plot')
flags.DEFINE_boolean('bars', False,
                     'plot bar chart (loc, weight) for each component')


def deserialize_mixture_from_file(filename):
    qt_deserialized = np.load(filename)
    locs = qt_deserialized['locs'].astype(np.float32)
    scale_diags = qt_deserialized['scale_diags'].astype(np.float32)
    weights = qt_deserialized['weights'].astype(np.float32)

    q_comps = [
        MultivariateNormalDiag(
            loc=loc,
            scale_diag=scale_diag)
        for loc, scale_diag in zip(locs, scale_diags)
    ]
    cat = Categorical(probs=tf.convert_to_tensor(weights))
    q_latest = Mixture(cat=cat, components=q_comps)
    return q_latest


def deserialize_target_from_file(filename):
    qt_deserialized = np.load(filename)
    mus = qt_deserialized['mus'].astype(np.float32)
    stds = qt_deserialized['stds'].astype(np.float32)
    pi = qt_deserialized['pi'].astype(np.float32)

    cat = Categorical(probs=tf.convert_to_tensor(pi[0]))
    target_comps = [
        MultivariateNormalDiag(
            loc=tf.convert_to_tensor(mus[i]),
            scale_diag=tf.convert_to_tensor(stds[i]))
        for i in range(len(mus))
    ]
    return Mixture(cat=cat, components=target_comps)


def main(argv):
    del argv

    x = deserialize_target_from_file(FLAGS.target)

    if FLAGS.widegrid:
        grid = np.arange(-25, 25, 0.1).astype(np.float32)
    else:
        grid = np.arange(-4, 4, 0.1).astype(np.float32)

    if FLAGS.grid2d:
        # 2D grid
        grid = np.arange(-2, 2, 0.1).astype(np.float32)
        gridx, gridy = np.meshgrid(grid, grid)
        grid = np.vstack((gridx.flatten(), gridy.flatten())).T

    if FLAGS.labels:
        labels = FLAGS.labels
    else:
        labels = ['approximation'] * len(FLAGS.qt)

    if FLAGS.styles:
        styles = FLAGS.styles
    else:
        styles = ['+', 'x', '.', '-']
        colors = ['Greens', 'Reds']

    sess = tf.Session()
    if FLAGS.grid2d:
        fig = plt.figure()
        ax = fig.add_subplot(211)
    else:
        fig, ax = plt.subplots()
        grid = np.expand_dims(grid, 1)  # package dims for tf
    with sess.as_default():
        xprobs = x.log_prob(grid)
        xprobs = tf.exp(xprobs).eval()
        if FLAGS.grid2d:
            ax.pcolormesh(
                gridx, gridy, xprobs.reshape(gridx.shape), cmap='Blues')
        else:
            ax.plot(grid, xprobs, label='target', linewidth=2.0)

        if len(FLAGS.qt) == 0:
            eprint(
                "provide some qts to the `--qt` option if you would like to "
                "plot them"
            )

        for i, (qt_filename, label) in enumerate(zip(FLAGS.qt, labels)):
            debug("visualizing %s" % qt_filename)
            qt = deserialize_mixture_from_file(qt_filename)
            qtprobs = tf.exp(qt.log_prob(grid))
            qtprobs = qtprobs.eval()
            if FLAGS.grid2d:
                ax2 = fig.add_subplot(212)
                ax2.pcolormesh(
                    gridx, gridy, qtprobs.reshape(gridx.shape), cmap='Greens')
            else:
                ax.plot(
                    grid,
                    qtprobs,
                    styles[i % len(styles)],
                    label=label,
                    linewidth=2.0)

        if len(FLAGS.qt) == 1 and FLAGS.bars:
            locs = [comp.loc.eval() for comp in qt.components]
            ax.plot(locs, [0] * len(locs), '+')

            weights = qt.cat.probs.eval()
            for i in range(len(locs)):
                ax.bar(locs[i], weights[i], .05)

    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xlabel(FLAGS.xlabel)
    ax.set_ylabel(FLAGS.ylabel)
    fig.suptitle(FLAGS.title)
    if not FLAGS.grid2d:
        legend = plt.legend(
            loc='upper right', prop={'size': 15}, bbox_to_anchor=(1.08, 1))
    if FLAGS.outdir == 'stdout':
        plt.show()
    else:
        fig.tight_layout()
        outname = os.path.join(os.path.expanduser(FLAGS.outdir), FLAGS.outfile)
        fig.savefig(outname, bbox_extra_artists=(legend,), bbox_inches='tight')
        print('saved to ', outname)


if __name__ == "__main__":
    app.run(main)
