"""
Plots metrics elbos, relbos, kl and times

Usage:
    python plot_losses.py \
            --elbos_files=/path/elbos.csv,/path2/elbos.csv \
            --relbos_files=/path/relbos.csv,/path2/relbos.csv \
            --kl_files=/path/relbos.csv \
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
flags.DEFINE_list('times_files', [], '')

def plot_elbos(ax):
    ax.set_yscale("symlog")
    for i, fname in enumerate(FLAGS.elbos_files):
        with open(fname, 'r') as f:
            elbos = [float(e.split(',')[0]) for e in f.readlines()]
        plt.plot(elbos, label=FLAGS.labels[i])
        #plt.semilogy(elbos, label=FLAGS.labels[i])
    plt.ylabel('elbo')
    plt.legend(loc='lower right')


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
        plt.plot(kl, label=FLAGS.labels[i])
    plt.ylabel('kl')
    plt.legend(loc='lower right')


def plot_times(ax):
    for i, fname in enumerate(FLAGS.times_files):
        with open(fname, 'r') as f:
            times = list(map(float, f.readlines()))
        plt.plot(times, label=FLAGS.labels[i])
    plt.ylabel('time')
    plt.legend(loc='lower right')

def main(argv):
    del argv
    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax = plt.subplot(221)
    plot_elbos(ax)
    ax = plt.subplot(222)
    plot_relbos(ax)
    ax = plt.subplot(223)
    plot_kl(ax)
    ax = plt.subplot(224)
    plot_times(ax)
    plt.show()

if __name__ == "__main__":
    app.run(main)
