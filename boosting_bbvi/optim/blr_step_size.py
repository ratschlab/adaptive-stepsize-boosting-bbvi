"""FW Step size selection methods for Bayesian Logistic Regression."""
import sys, os
import numpy as np
import copy
import tensorflow as tf
from tensorflow.contrib.distributions import kl_divergence
import edward as ed
from edward.models import (Categorical, Dirichlet, Empirical, InverseGamma,
                           MultivariateNormalDiag, Normal, ParamMixture,
                           Mixture)
from scipy.misc import logsumexp as logsumexp

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from boosting_bbvi.optim.utils import divergence, grad_elbo, argmax_grad_dotp, elbo
from boosting_bbvi.core.utils import eprint, debug
import boosting_bbvi.core.utils as coreutils
logger = coreutils.get_logger()

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer(
    'n_monte_carlo_samples', 1000,
    'Number of samples for approximating gradient')
flags.DEFINE_integer(
    'n_line_search_iter', 10,
    'Number of iterations line search gradient descent')
flags.DEFINE_enum('linit', 'fixed',
                  ['fixed', 'lipschitz_v2', 'lipschitz_v1'],
                  'Initialization methods versions for lipschitz constant')
flags.DEFINE_float('linit_fixed', 0.001,
                   'Fixed initial estimate of Lipschitz constant')
flags.DEFINE_integer('adafw_MAXITER', 10,
                     'Maximum iterations of adaptive fw L estimation')
flags.DEFINE_float('damping_adafw', 0.99, 'Damping constant')
flags.DEFINE_float('exp_adafw', 2.0, 'Multiplicative factor in L')
flags.DEFINE_enum(
    'distance_metric', 'dotproduct', ['dotproduct', 'kl', 'constant'],
    'Metric to use for distance norm between probability distrbutions')

MIN_GAMMA = 0.01

def fixed(weights, params, q_t, mu_s, cov_s, s_t, p, k, gap=None):
    """Fixed step size.
    
    Args:
        weights: [k] weights of the mixture components of q_t
        params: dictionary containing mixture parameters ('loc', 'scale')
        q_t: current solution iterate
        mu_s: [dim], mean of LMO solution s
        cov_s: [dim], cov matrix for LMO solution s
        s_t: Current atom & LMO Solution s
        p: Joint distribution p(z, x)
        k: iteration number of Frank-Wolfe
        gap: Duality-Gap (if already computed)
    Returns:
        dictionary containing gamma, new weights and new components
        and duality gap of current iterate
    """
    gamma = 2. / (k + 2.)
    new_weights = [(1. - gamma) * w for w in weights]
    new_weights.append(gamma)
    new_params = copy.copy(params)
    new_params.append({'loc': mu_s, 'scale': cov_s})

    # Compute Frank-Wolfe gap
    if gap is None:
        N_samples = FLAGS.n_monte_carlo_samples
        sample_q = q_t.sample([N_samples])
        sample_s = s_t.sample([N_samples])
        step_s = tf.reduce_mean(grad_elbo(q_t, p, sample_s)).eval()
        step_q = tf.reduce_mean(grad_elbo(q_t, p, sample_q)).eval()
        gap = step_q - step_s
    if gap < 0: logger.warning("Frank-Wolfe duality gap is negative")

    return {
        'gamma': gamma,
        'weights': new_weights,
        'params': new_params,
        'gap': gap,
        'step_type': 'fixed'
    }


def adafw_linit(q_0, p):
    """Initialization of L estimate for
    Adaptive Frank Wolfe algorithm.
    
    Args:
        q_0: Initial iterate
        p: Joint to compute objective function
    Returns:
        L initialized value, float
    """
    if FLAGS.linit != 'fixed':
        raise NotImplementedError('Initialization only fixed for now')
    return FLAGS.linit_fixed


def adaptive_fw(weights, params, q_t, mu_s, cov_s, s_t, p, k, l_prev,
                gap=None):
    """Adaptive Frank-Wolfe algorithm.
    
    Sets step size as suggested in Algorithm 1 of
    https://arxiv.org/pdf/1806.05123.pdf

    Args:
        weights: [k], weights of the mixture components of q_t
        params: list containing dictionary of mixture params ('mu', 'scale')
        q_t: current mixture iterate q_t
        mu_s: [dim], mean for LMO solution s
        cov_s: [dim], cov matrix for LMO solution s
        s_t: Current atom & LMO Solution s
        p: edward.model, target distribution p
        k: iteration number of Frank-Wolfe
        l_prev: previous lipschitz estimate
        gap: Duality-Gap (if already computed)
    Returns:
        a dictionary containing gamma, new weights, new parameters
        lipschitz estimate, duality gap of current iterate
        and step information
    """

    # FIXME
    is_vector = FLAGS.base_dist in ['mvnormal', 'mvlaplace']

    d_t_norm = divergence(s_t, q_t, metric=FLAGS.distance_metric).eval()
    logger.info('\ndistance norm is %.3e' % d_t_norm)

    N_samples = FLAGS.n_monte_carlo_samples
    if gap is None:
        # create and sample from $s_t, q_t$
        sample_q = q_t.sample([N_samples])
        sample_s = s_t.sample([N_samples])
        step_s = tf.reduce_mean(grad_elbo(q_t, p, sample_s)).eval()
        step_q = tf.reduce_mean(grad_elbo(q_t, p, sample_q)).eval()
        gap = step_q - step_s
    logger.info('duality gap %.3e' % gap)
    if gap < 0:
        logger.warning("Duality gap is negative returning fixed step")
        return fixed(weights, params, q_t, mu_s, cov_s, s_t, p, k, gap)

    gamma = 2. / (k + 2.)
    tau = FLAGS.exp_adafw
    eta = FLAGS.damping_adafw
    # NOTE: this is from v1 of the paper, new version
    # replaces multiplicative eta with divisor eta
    pow_tau = 1.0
    i, l_t = 0, l_prev
    # Objective in this case is -ELBO
    f_t = -elbo(q_t, p, N_samples, return_std=False)
    debug('f(q_t) = %.3e' % (f_t))
    # return intial estimate if gap is -ve
    while gamma >= MIN_GAMMA and i < FLAGS.adafw_MAXITER:
        # compute $L_t$ and $\gamma_t$
        l_t = pow_tau * eta * l_prev
        gamma = min(gap / (l_t * d_t_norm), 1.0)
        d_1 = - gamma * gap
        d_2 = gamma * gamma * l_t * d_t_norm / 2.
        debug('linear d1 = %.3e, quad d2 = %.3e' % (d_1, d_2))
        quad_bound_rhs = f_t + d_1 + d_2

        # $w_{t + 1} = [(1 - \gamma)w_t, \gamma]$
        # Handling the case of gamma = 1.0
        # separately, weights might not get exactly 0 because
        # of precision issues. 0 wt components should be removed
        if gamma != 1.0:
            new_weights = copy.copy(weights)
            new_weights = [(1. - gamma) * w for w in new_weights]
            new_weights.append(gamma)
            new_params = copy.copy(params)
            new_params.append({'loc': mu_s, 'scale': cov_s})
            new_components = [
                coreutils.base_loc_scale(
                    FLAGS.base_dist,
                    c['loc'],
                    c['scale'],
                    multivariate=is_vector) for c in new_params
            ]
        else:
            new_weights = [1.]
            new_params = [{'loc': mu_s, 'scale': cov_s}]
            new_components = [s_t]

        qt_new = coreutils.get_mixture(new_weights, new_components)
        quad_bound_lhs = -elbo(qt_new, p, N_samples, return_std=False)
        logger.info('lt = %.3e, gamma = %.3f, f_(qt_new) = %.3e, '
                    'linear extrapolated = %.3e' % (l_t, gamma, quad_bound_lhs,
                                                    quad_bound_rhs))
        if quad_bound_lhs <= quad_bound_rhs:
            # Adaptive loop succeeded
            return {
                'gamma': gamma,
                'l_estimate': l_t,
                'weights': new_weights,
                'params': new_params,
                'gap': gap,
                'step_type': 'adaptive'
            }
        pow_tau *= tau
        i += 1

    # gamma below MIN_GAMMA
    logger.warning("gamma below threshold value, returning fixed step")
    return fixed(weights, params, q_t, mu_s, cov_s, s_t, p, k, gap)


def adaptive_pfw(weights, params, q_t, mu_s, cov_s, s_t, p, k, l_prev):
    """Adaptive pairwise variant.
    
    Args:
        weights: [k], weights of the mixture components of q_t
        params: list containing dictionary of mixture params ('mu', 'scale')
        q_t: current mixture iterate q_t
        mu_s: [dim], mean for LMO solution s
        cov_s: [dim], cov matrix for LMO solution s
        s_t: Current atom & LMO Solution s
        p: edward.model, target distribution p
        k: iteration number of Frank-Wolfe
        l_prev: previous lipschitz estimate
    Returns:
        a dictionary containing gamma, new weights, new parameters
        lipschitz estimate, duality gap of current iterate
        and step information
    """

    # FIXME
    is_vector = FLAGS.base_dist in ['mvnormal', 'mvlaplace']

    d_t_norm = divergence(s_t, q_t, metric=FLAGS.distance_metric).eval()
    logger.info('\ndistance norm is %.3e' % d_t_norm)

    # Find v_t
    qcomps = q_t.components
    index_v_t, step_v_t = argmax_grad_dotp(p, q_t, qcomps,
                                           FLAGS.n_monte_carlo_samples)
    v_t = qcomps[index_v_t]

    # Pairwise gap
    N_samples = FLAGS.n_monte_carlo_samples
    sample_s = s_t.sample([N_samples])
    step_s = tf.reduce_mean(grad_elbo(q_t, p, sample_s)).eval()
    gap_pw = step_v_t - step_s
    logger.info('Pairwise gap %.3e' % gap_pw)
    if gap_pw <= 0:
        logger.warning('Pairwise gap <= 0, returning fixed step')
        return fixed(weights, params, q_t, mu_s, cov_s, s_t, p, k, gap_pw)
    gap = gap_pw

    MAX_GAMMA = weights[index_v_t]

    gamma = 2. / (k + 2.)
    tau = FLAGS.exp_adafw
    eta = FLAGS.damping_adafw
    pow_tau = 1.0
    i, l_t = 0, l_prev
    f_t = -elbo(q_t, p, N_samples, return_std=False)
    debug('f(q_t) = %.3e' % f_t)
    is_drop_step = False
    while gamma >= MIN_GAMMA and i < FLAGS.adafw_MAXITER:
        # compute L_t and gamma_t
        l_t = pow_tau * eta * l_prev
        gamma = min(gap / (l_t * d_t_norm), MAX_GAMMA)

        d_1 = -gamma * gap
        d_2 = gamma * gamma * l_t * d_t_norm / 2.
        debug('linear d1 = %.5f, quad d2 = %.5f' % (d_1, d_2))
        quad_bound_rhs = f_t  + d_1 + d_2

        # construct q_{t + 1}
        # handle the case of gamma = MAX_GAMMA separately
        new_weights = copy.copy(weights)
        new_weights.append(gamma)
        new_params = copy.copy(params)
        new_params.append({'loc': mu_s, 'scale': cov_s})
        if gamma != MAX_GAMMA:
            new_weights[index_v_t] -= gamma
            is_drop_step = False
        else:
            # hardcoding to 0
            del new_weights[index_v_t]
            del new_params[index_v_t]
            is_drop_step = True

        new_components = [
            coreutils.base_loc_scale(
                FLAGS.base_dist, c['loc'], c['scale'], multivariate=is_vector)
            for c in new_params
        ]

        qt_new = coreutils.get_mixture(new_weights, new_components)
        quad_bound_lhs = -elbo(qt_new, p, N_samples, return_std=False)
        logger.info('lt = %.3e, gamma = %.3f, f_(qt_new) = %.3e, '
                    'linear extrapolated = %.3e' % (l_t, gamma, quad_bound_lhs,
                                                    quad_bound_rhs))
        if quad_bound_lhs <= quad_bound_rhs:
            # Adaptive loop succeeded
            return {
                'gamma': gamma,
                'l_estimate': l_t,
                'weights': new_weights,
                'params': new_params,
                'gap': gap,
                'step_type': 'drop' if is_drop_step else 'adaptive'
            }
        pow_tau *= tau
        i += 1

    # gamma below MIN_GAMMA
    logger.warning("gamma below threshold value, returning fixed step")
    return fixed(weights, params, q_t, mu_s, cov_s, s_t, p, k, gap)


def adaptive_afw(weights, params, q_t, mu_s, cov_s, s_t, p, k, l_prev):
    """Adaptive Away Steps algorithm.

    Args:
        weights: [k], weights of the mixture components of q_t
        params: list containing dictionary of mixture params ('mu', 'scale')
        q_t: current mixture iterate q_t
        mu_s: [dim], mean for LMO solution s
        cov_s: [dim], cov matrix for LMO solution s
        s_t: Current atom & LMO Solution s
        p: edward.model, target distribution p
        k: iteration number of Frank-Wolfe
        l_prev: previous lipschitz estimate
    Returns:
        a dictionary containing gamma, new weights, new parameters
        lipschitz estimate, duality gap of current iterate
        and step information
    """
    # FIXME
    is_vector = FLAGS.base_dist in ['mvnormal', 'mvlaplace']

    d_t_norm = divergence(s_t, q_t, metric=FLAGS.distance_metric).eval()
    logger.info('\ndistance norm is %.3e' % d_t_norm)

    # Find v_t
    qcomps = q_t.components
    index_v_t, step_v_t = argmax_grad_dotp(p, q_t, qcomps,
                                           FLAGS.n_monte_carlo_samples)
    v_t = qcomps[index_v_t]

    # Frank-Wolfe gap
    N_samples = FLAGS.n_monte_carlo_samples
    sample_q = q_t.sample([N_samples])
    sample_s = s_t.sample([N_samples])
    step_s = tf.reduce_mean(grad_elbo(q_t, p, sample_s)).eval()
    step_q = tf.reduce_mean(grad_elbo(q_t, p, sample_q)).eval()
    gap_fw = step_q - step_s
    if gap_fw < 0: logger.warning("Frank-Wolfe duality gap is negative")
    # Away gap
    gap_a = step_v_t - step_q
    if gap_a < 0: eprint('Away gap < 0!!!')
    logger.info('fw gap %.3e, away gap %.3e' % (gap_fw, gap_a))

    # FIXME(sauravshekhar): In case of one component w will be 1.0
    # fix FW direction in that case as w / (1 - w) will cause issues
    if (gap_fw >= gap_a) or (len(params) == 1):
        # FW direction, proceeds exactly as adafw
        logger.info('Proceeding in FW direction ')
        return adaptive_fw(weights, params, q_t, mu_s, cov_s, s_t, p, k,
                           l_prev, gap_fw)

    # Away direction
    logger.info('Proceeding in Away direction ')
    adaptive_step_type = 'away'
    gap = gap_a
    if weights[index_v_t] < 1.0:
        MAX_GAMMA = weights[index_v_t] / (1.0 - weights[index_v_t])
    else:
        MAX_GAMMA = 100. # Large value when t = 1

    gamma = 2. / (k + 2.)
    tau = FLAGS.exp_adafw
    eta = FLAGS.damping_adafw
    pow_tau = 1.0
    i, l_t = 0, l_prev
    f_t =  -elbo(q_t, p, N_samples, return_std=False)
    debug('f(q_t) = %.5f' % (f_t))
    is_drop_step = False
    while gamma >= MIN_GAMMA and i < FLAGS.adafw_MAXITER:
        # compute $L_t$ and $\gamma_t$
        l_t = pow_tau * eta * l_prev
        # NOTE: Handle extreme values of gamma carefully
        gamma = min(gap / (l_t * d_t_norm), MAX_GAMMA)

        d_1 = - gamma * gap
        d_2 = gamma * gamma * l_t * d_t_norm / 2.
        debug('linear d1 = %.5f, quad d2 = %.5f' % (d_1, d_2))
        quad_bound_rhs = f_t  + d_1 + d_2

        # construct $q_{t + 1}$
        new_weights = copy.copy(weights)
        new_params = copy.copy(params)
        if gamma == MAX_GAMMA:
            # drop v_t
            is_drop_step = True
            del new_weights[index_v_t]
            new_weights = [(1. + gamma) * w for w in new_weights]
            del new_params[index_v_t]
        else:
            is_drop_step = False
            new_weights = [(1. + gamma) * w for w in new_weights]
            new_weights[index_v_t] -= gamma


        new_components = [
            coreutils.base_loc_scale(
                FLAGS.base_dist, c['loc'], c['scale'], multivariate=is_vector)
            for c in new_params
        ]

        qt_new = coreutils.get_mixture(new_weights, new_components)
        quad_bound_lhs = -elbo(qt_new, p, N_samples, return_std=False)
        logger.info('lt = %.3e, gamma = %.3f, f_(qt_new) = %.3e, '
                    'linear extrapolated = %.3e' % (l_t, gamma, quad_bound_lhs,
                                                    quad_bound_rhs))
        if quad_bound_lhs <= quad_bound_rhs:
            return {
                'gamma': gamma,
                'l_estimate': l_t,
                'weights': new_weights,
                'params': new_params,
                'gap': gap,
                'step_type': "drop" if is_drop_step else "away"
            }
        pow_tau *= tau
        i += 1

    # gamma below MIN_GAMMA
    logger.warning("gamma below threshold value, returning fixed step")
    return fixed(weights, params, q_t, mu_s, cov_s, s_t, p, k, gap)


def line_search_dkl(weights, params, q_t, mu_s, cov_s, s_t, p, k, gap=None):
    """Performs line search for the best step size gamma.
    
    Uses gradient ascent to find gamma that minimizes
    ELBO(q_t + gamma (s - q_t) || p)
    
    Args:
        weights: [k], weights of mixture components of q_t
        params: list containing dictionary of mixture params ('mu', 'scale')
        q_t: current mixture iterate q_t
        mu_s: [dim], mean for LMO Solution s
        cov_s: [dim], cov matrix for LMO solution s
        s_t: Current atom & LMO Solution s
        p: edward.model, target distribution p
        k: iteration number of Frank-Wolfe
    Returns:
        a dictionary containing gamma, new weights, new parameters
        lipschitz estimate, duality gap of current iterate
        and step information
    """
    # FIXME
    is_vector = FLAGS.base_dist in ['mvnormal', 'mvlaplace']

    N_samples = FLAGS.n_monte_carlo_samples
    # sample from $q_t$ and s
    sample_q = q_t.sample([N_samples])
    sample_s = s_t.sample([N_samples])

    if gap is None:
        # create and sample from $s_t, q_t$
        sample_q = q_t.sample([N_samples])
        sample_s = s_t.sample([N_samples])
        step_s = tf.reduce_mean(grad_elbo(q_t, p, sample_s)).eval()
        step_q = tf.reduce_mean(grad_elbo(q_t, p, sample_q)).eval()
        gap = step_q - step_s
    logger.info('duality gap %.3e' % gap)
    if gap < 0:
        logger.warning("Duality gap is negative.")

    # initialize $\gamma$
    gamma = 2. / (k + 2.)
    n_steps = FLAGS.n_line_search_iter
    prog_bar = ed.util.Progbar(n_steps)
    # storing gradients for analysis
    grad_gamma = []
    for it in range(n_steps):
        print("line_search iter %d, %.5f" % (it, gamma))
        new_weights = copy.copy(weights)
        new_weights = [(1. - gamma) * w for w in new_weights]
        new_weights.append(gamma)
        new_params = copy.copy(params)
        new_params.append({'loc': mu_s, 'scale': cov_s})
        new_components = [
            coreutils.base_loc_scale(
                FLAGS.base_dist,
                c['loc'],
                c['scale'],
                multivariate=is_vector) for c in new_params
        ]
        qt_new = coreutils.get_mixture(new_weights, new_components)
        step_s = tf.reduce_mean(grad_elbo(qt_new, p, sample_s)).eval()
        step_q = tf.reduce_mean(grad_elbo(qt_new, p, sample_q)).eval()
        # Gradient descent step size decreasing as $\frac{1}{it + 1}$
        gamma = gamma - 0.1 * (step_s - step_q) / (it + 1.)
        gap = step_q - step_s
        # Projecting it back to [0, 1]
        if gamma >= 1 or gamma <= 0:
            gamma = max(min(gamma, 1.), 0.)
            break
    return {
        'gamma': gamma,
        'n_samples': N_samples,
        'weights': new_weights,
        'params': new_params,
        'step_type': 'line_search'
    }
