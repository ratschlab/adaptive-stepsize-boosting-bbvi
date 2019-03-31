"""Utilities for relbo and optimization."""
import logging
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import kl_divergence
from edward.models import (Categorical, Dirichlet, Empirical, InverseGamma,
                           MultivariateNormalDiag, Normal, ParamMixture,
                           Mixture, Laplace, VectorLaplaceDiag)
import edward as ed
from colorama import Fore
from colorama import Style

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from boosting_bbvi.core.mvn import mvn
from boosting_bbvi.core.lpl import lpl

def eprint(*args, **kwargs):
    print(Fore.RED, *args, Style.RESET_ALL, file=sys.stderr, **kwargs)


def debug(*args, **kwargs):
    print(Fore.YELLOW, *args, Style.RESET_ALL, file=sys.stderr, **kwargs)


def decay_linear(value):
    return 1. / value


def decay_log(value):
    return 1. / np.log(value + 1)


def decay_squared(value):
    return 1. / (value**2)


logger = logging.getLogger('Bbbvi')
logger.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)
def get_logger():
    return logger

def update_weights(weights, gamma, iter):
    if iter == 0:
        weights.append(gamma)
    else:
        weights = [(1. - gamma) * w for w in weights]
        weights.append(gamma)

    return weights

def get_mixture(weights, components):
    """Build a mixture model with given weights and components.
    
    Args:
        weights: list or np.array
        components: list ed.distribution
    Returns:
        constructed mixture
    """
    assert len(weights) == len(
        components), 'Weights size %d not same as components size %d' % (
            len(weights), len(components))
    assert math.isclose(
        1., sum(weights), rel_tol=1e-5), "Weights not normalized"

    if len(weights) == 1: return components[0] # not a mixture
    return Mixture(
        cat=Categorical(probs=tf.convert_to_tensor(weights)),
        components=components)


def construct_multivariatenormaldiag(dims, iter, name=''):
    return construct_base('mvnormal', dims, iter, name, multivariate=True)


def construct_normal(dims, iter, name=''):
    return construct_base('normal', dims, iter, name)


def construct_laplace(dims, iter, name=''):
    return construct_base('laplace', dims, iter, name)


def base_loc_scale(dist_name, loc, scale, **kwargs):
    """Get base distribution from location-scale family.
    
    Args:
        dist_name: name of the distribution
        loc: mean
        scale: variance scale
        kwargs: other information for constructing the distribution
    Returns:
        Instance of the distribution
    """
    base_dict = {
        'normal': Normal,
        'laplace': Laplace,
        'mvnormal': MultivariateNormalDiag,
        'mvlaplace': VectorLaplaceDiag,
        'mvn': mvn,
        'mvl': lpl,
    }
    if dist_name in ['mvl']:
        eprint('mvn and lpl dont have multivariate log_prob()')
        raise NotImplementedError
    Base = base_dict[dist_name]

    # Handle MultivariateNormalDiag
    is_vector = kwargs.pop('multivariate', False)
    if is_vector:
        return Base(loc=loc, scale_diag=scale)

    return Base(loc=loc, scale=scale)

def construct_base(dist_name, dims, iter, name='', **kwargs):
    """Construct base distribution for Variational Approximation.
    
    Args:
        dist_name: name of the distribution
        dims: dimensionality of the distribution
        iter: iteration of the algorithm (for naming)
        name: name of variable
        kwargs: other information for constructing the distribution
    Returns:
        An instance of the distribution with Gaussian initialization
        for mean and variances"""
    # TODO(sauravshekhar) check if np.random.normal() matters and remove o/w
    loc = tf.get_variable(
        name + "_loc%d" % iter,
        initializer=tf.random_normal(dims) + np.random.normal())
    scale = tf.nn.softplus(
        tf.get_variable(
            name + "_scale%d" % iter,
            initializer=tf.random_normal(dims) + np.random.normal()))
    return base_loc_scale(dist_name, loc, scale, **kwargs)


def compute_relbo(s, qt, p, l):
    # Assumes being called under a tensorflow session
    s_samples = s.sample(500)
    relbo = tf.reduce_sum(p.log_prob(s_samples)) \
            - l * tf.reduce_sum(s.log_prob(s_samples)) \
            - tf.reduce_sum(qt.log_prob(s_samples))
    return relbo.eval()


def append_to_file(path, value):
    """Append value to the file at path."""
    with open(path, 'a') as f:
        f.write(str(value))
        f.write('\n')


def block_diagonal(matrices, dtype=tf.float32):
    r"""Constructs block-diagonal matrices from a list of batched 2D tensors.

  Args:
    matrices: A list of Tensors with shape [..., N_i, M_i] (i.e. a list of
      matrices with the same batch dimension).
    dtype: Data type to use. The Tensors in `matrices` must match this dtype.
  Returns:
    A matrix with the input matrices stacked along its main diagonal, having
    shape [..., \sum_i N_i, \sum_i M_i].

  """
    matrices = [tf.convert_to_tensor(matrix, dtype=dtype) for matrix in matrices]
    blocked_rows = tf.Dimension(0)
    blocked_cols = tf.Dimension(0)
    batch_shape = tf.TensorShape(None)
    for matrix in matrices:
        full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
        batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
        blocked_rows += full_matrix_shape[-2]
        blocked_cols += full_matrix_shape[-1]
    ret_columns_list = []
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        ret_columns_list.append(matrix_shape[-1])
    ret_columns = tf.add_n(ret_columns_list)
    row_blocks = []
    current_column = 0
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        row_before_length = current_column
        current_column += matrix_shape[-1]
        row_after_length = ret_columns - current_column
        row_blocks.append(tf.pad(
            tensor=matrix,
            paddings=tf.concat(
                [tf.zeros([tf.rank(matrix) - 1, 2], dtype=tf.int32),
                 [(row_before_length, row_after_length)]],
                axis=0)))
    blocked = tf.concat(row_blocks, -2)
    blocked.set_shape(batch_shape.concatenate((blocked_rows, blocked_cols)))
    return blocked
