"""
Plot components of the mixture model

Usage:
    python plot_mixture_comps.py \
            --outdir=stdout \
            --qt=/path/qt_iter25.npz \
            --label=iter25
"""

from __future__ import print_function

import os
import sys
import numpy as np
import scipy.stats as stats

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from boosting_bbvi.core.utils import eprint, debug
from boosting_bbvi.core.infinite_mixture import InfiniteMixtureScipy

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


FLAGS = flags.FLAGS
flags.DEFINE_string('outdir', '/tmp/',
                    'Output file to store plot into, set to stdout to '
                    'only show the plots and not save them')
flags.DEFINE_string('outfile', 'mixtures.png', 'name of the plot file')
flags.DEFINE_string('title', 'results', '')
flags.DEFINE_string('ylabel', 'y', '')
flags.DEFINE_string('xlabel', 'x', '')
flags.DEFINE_string('qt', "", 'iteration to visualize')
flags.DEFINE_string('label', '', 'label to be associated with the qt')
flags.DEFINE_boolean('widegrid', False, 'range for the x-axis')
flags.DEFINE_boolean('grid2d', False, '3D plot')

def get_mixture_params_from_file(filename):
    qt_deserialized = np.load(filename)
    locs = qt_deserialized['locs'].astype(np.float32)
    scale = qt_deserialized['scale_diags'].astype(np.float32)
    weights = qt_deserialized['weights'].astype(np.float32)
    return {'locs': locs, 'scale_diags': scale, 'weights': weights}

def plot_normal_mix(pis, mus, sigmas, ax, label=''):
    """Plots the mixture of Normal models to axis=ax"""
    grid = np.arange(-4, 4, 0.1).astype(np.float32)
    final = np.zeros_like(grid)
    if FLAGS.grid2d:
        raise NotImplementedError('2d not implemented yet...')
    for i, (weight_mix, mu_mix, sigma_mix) in enumerate(zip(pis, mus, sigmas)):
        temp = stats.norm.pdf(grid, mu_mix, sigma_mix)
        # Do not multiply by weights and show unweighted distribution
        #temp *= weight_mix
        final = final + weight_mix * temp
        ax.plot(grid, temp, label='{}'.format(i))
    ax.plot(grid, final, '+', label='Mixture')
    ax.legend(fontsize=13)


def main(argv):
    del argv
    if FLAGS.grid2d:
        raise NotImplementedError('Only 1D Normal supported...')

    if FLAGS.qt == "":
        eprint(
            "provide some qt to the `--qt` option if you would like to "
            "plot"
        )

    if FLAGS.label:
        label = FLAGS.label
    else:
        qt_file = os.path.splitext(FLAGS.qt)[0]
        label = qt_file[qt_file.find('qt_') + len('qt_'):]

    plt.figure(1)
    debug("visualizing %s" % os.path.basename(FLAGS.qt))
    mixture_params = get_mixture_params_from_file(FLAGS.qt)
    plot_normal_mix(mixture_params['weights'], mixture_params['locs'],
                    mixture_params['scale_diags'], plt, label)

    plt.figure(2)
    w = mixture_params['weights']
    plt.bar(np.arange(len(w)), w, color='b', label=label)

    if FLAGS.outdir == 'stdout':
        plt.show()
    else:
        fig.tight_layout()
        outname = os.path.join(os.path.expanduser(FLAGS.outdir), FLAGS.outfile)
        fig.savefig(outname, bbox_extra_artists=(legend,), bbox_inches='tight')
        print('saved to ', outname)


if __name__ == "__main__":
    app.run(main)
