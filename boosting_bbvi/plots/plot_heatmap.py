"""Plot heatmap of eta and tau hyperparameters vs best KL divergence.

Assumes directories have corresponding csv files
as well as meta.info files containing hyperparameter
values written as

tau=1.01, eta=0.99, linit=0.001, n_monte_carlo_samples=1000, x=y ..

Usage:
    python plot_heatmap.py \
            --dirlist=/path1,/path2, \
            --metric=kl \
"""


import os
import sys

import numpy as np
import plot_utils as utils

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from boosting_bbvi.core.utils import eprint, debug
from boosting_bbvi.core.infinite_mixture import InfiniteMixtureScipy

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('metric', 'kl', 'metric to plot')
flags.DEFINE_list('dirlist', [], 'directory list')

def get_best_metric(filename):
    "Combine metric among iterations"
    if FLAGS.metric != 'kl':
        raise NotImplementedError('metric %s not supported, only kl supported' %
                                  (FLAGS.metric))
    return min([float(e.strip()) for e in open(filename, 'r').readlines()])

def parse(hp_string):
    "Get hyperparameter values from string"
    metric_dict = {}
    metrics = hp_string.split(',')
    for metric in metrics:
        metric_name, metric_val = metric.strip().split('=')
        metric_val = float(metric_val)
        metric_dict[metric_name] = metric_val
    return metric_dict

def main(argv):
    # NOTE: keep values monotonic
    tau_list = [1.01, 1.1, 1.5, 2.0]
    eta_list = [0.1, 0.01, 0.5, 0.99]
    if FLAGS.metric != 'kl':
        raise NotImplementedError('metric %s not supported, only kl supported' %
                                  (FLAGS.metric))
    val_matrix = np.full((len(tau_list), len(eta_list)), np.inf)
    for folder in FLAGS.dirlist:
        # sending string as some folders may not have meta.info
        hyper_params = parse(
            open(os.path.join(folder, "meta.info"), 'r').readline())
        x = tau_list.index(hyper_params['tau'])
        y = eta_list.index(hyper_params['eta'])
        val_matrix[x, y] = get_best_metric(os.path.join(folder, "kl.csv"))

    debug(val_matrix)
    # val_matrix = np.random.rand(4, 4)

    fig, ax = plt.subplots()
    im = ax.imshow(val_matrix, cmap='magma_r')
    ax.set_xticks(np.arange(len(tau_list)))
    ax.set_yticks(np.arange(len(eta_list)))
    ax.set_xticklabels(tau_list)
    ax.set_yticklabels(eta_list)
    plt.setp(ax.get_xticklabels(), fontsize=16)
    plt.setp(ax.get_yticklabels(), fontsize=16)

    #for i in range(len(tau_list)):
    #    for j in range(len(eta_list)):
    #        text = ax.text(j, i, val_matrix[i, j],
    #                    ha="center", va="center", color="w")

    ax.set_title("%s value for different hp configurations" % (FLAGS.metric))
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    app.run(main)
