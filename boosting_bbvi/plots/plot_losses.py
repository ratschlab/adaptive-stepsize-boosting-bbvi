"""
Plots metrics elbos, relbos, kl and times

Usage:
    python plot_losses.py \
            --elbos_files=/path/elbos.csv,/path2/elbos.csv \
            --relbos_files=/path/relbos.csv,/path2/relbos.csv \
            --kl_files=/path/kl.csv \
            --mse_files=/path/mse_test.csv,/path2/mse_test.csv \
            --times_files=/path/times.csv,/path2/times.csv \
            --labels=variant1,variant2
"""

import os
import sys

import numpy as np
import plot_utils as utils

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import seaborn as sns
plt.style.use('ggplot')

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from boosting_bbvi.core.utils import eprint, debug
from boosting_bbvi.core.infinite_mixture import InfiniteMixtureScipy

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_list('labels', ["1", "2", "3", "4"], 'Labels of file sets')

flags.DEFINE_string('xlabel', 'iterations', '')
flags.DEFINE_list('elbos_files', [], '')
flags.DEFINE_list('kl_files', [], '')
flags.DEFINE_list('relbos_files', [], '')
flags.DEFINE_list('mse_files', [], '')
flags.DEFINE_list('times_files', [], '')
flags.DEFINE_float('smoothingWt', 0., 'smoothness for the plot')
flags.DEFINE_integer('start', 0, 'starting iteration')

def resmoothDataset(x, alpha=0.6):
    """Apply linear filter
    """
    # From tensorboard:
    # Explanation:
    # https://github.com/tensorflow/tensorboard/blob/f801ebf1f9fbfe2baee1ddd65714d0bccc640fb1/tensorboard/plugins/scalar/vz_line_chart/vz-line-chart.ts#L55
    # Implementation:
    # https://github.com/tensorflow/tensorboard/blob/f801ebf1f9fbfe2baee1ddd65714d0bccc640fb1/tensorboard/plugins/scalar/vz_line_chart/vz-line-chart.ts#L704
    #x = x[FLAGS.start:]
    last = x[0]
    x_smoothed = []
    for e in x:
        s = last * alpha + (1 - alpha) * e
        x_smoothed.append(s)
        last = s
    return x_smoothed


def plot_elbos(ax):
    ax.set_yscale("symlog")
    for i, fname in enumerate(FLAGS.elbos_files):
        with open(fname, 'r') as f:
            elbos = [float(e.split(',')[0]) for e in f.readlines()]
            elbos = resmoothDataset(elbos, alpha=FLAGS.smoothingWt)
            print('Maximum elbo for %s is %.3f at iter %d' % (FLAGS.labels[i],
                                                              max(elbos),
                                                              np.argmax(elbos)))
        ax.plot(elbos, label=FLAGS.labels[i])
        ax.set_xlim(xmin=FLAGS.start)
        #plt.semilogy(elbos, label=FLAGS.labels[i])
    plt.ylabel('elbo')
    #plt.legend(loc='lower right')


def plot_relbos(ax):
    ax.set_yscale("symlog")
    for i, fname in enumerate(FLAGS.relbos_files):
        with open(fname, 'r') as f:
            relbos = list(map(float, f.readlines()))
        plt.plot(relbos, label=FLAGS.labels[i])
        #plt.semilogy(relbos, label=FLAGS.labels[i])
    plt.ylabel('relbo')
    plt.legend(loc='lower right')


def plot_kl(ax):
    for i, fname in enumerate(FLAGS.kl_files):
        with open(fname, 'r') as f:
            kl = list(map(float, f.readlines()))
            kl = resmoothDataset(kl, alpha=FLAGS.smoothingWt)
        ax.plot(kl, label=FLAGS.labels[i])
    plt.ylabel('kl')
    #plt.legend(loc='lower right')


def plot_mse(ax):
    ax.set_yscale("log")
    for i, fname in enumerate(FLAGS.mse_files):
        with open(fname, 'r') as f:
            mse = list(map(float, f.readlines()))
            mse = resmoothDataset(mse, alpha=FLAGS.smoothingWt)
            print('Minimum mse for %s is %.3f at iter %d' % (FLAGS.labels[i],
                                                             min(mse),
                                                             np.argmin(mse)))
        ax.plot(mse, label=FLAGS.labels[i])
        ax.set_xlim(xmin=FLAGS.start)
    plt.ylabel('mse')


def plot_times(ax):
    for i, fname in enumerate(FLAGS.times_files):
        with open(fname, 'r') as f:
            times = list(map(float, f.readlines()))
        plt.plot(times, label=FLAGS.labels[i])
    plt.ylabel('time')
    plt.legend(loc='lower right')

def main(argv):
    del argv
    fig, axes = plt.subplots(nrows=1, ncols=2)
    ax = plt.subplot(121)
    plot_elbos(ax)
    ax = plt.subplot(122)
    plot_mse(ax)
    #ax = plt.subplot(122)
    #plot_kl(ax)
    #ax = plt.subplot(223)
    #plot_relbos(ax)
    #ax = plt.subplot(224)
    #plot_times(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center')
    plt.show()

if __name__ == "__main__":
    app.run(main)
