"""Frank-Wolfe Optimization for Boosting BBVI.

Contains different step size selection methods.
"""
import sys, os
import numpy as np
import random
import copy
import scipy.stats as stats
import tensorflow as tf
from tensorflow.contrib.distributions import kl_divergence
import edward as ed
from edward.models import (Categorical, Dirichlet, Empirical, InverseGamma,
                           MultivariateNormalDiag, Normal, ParamMixture,
                           Mixture)
from scipy.misc import logsumexp as logsumexp

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from boosting_bbvi.core.infinite_mixture import InfiniteMixtureScipy
from boosting_bbvi.core.opt_utils import setup_outdir, divergence, softmax
from boosting_bbvi.core.opt_utils import grad_kl
from boosting_bbvi.core.utils import eprint, debug
import boosting_bbvi.core.utils as utils
logger = utils.get_logger()

flags = tf.app.flags
FLAGS = flags.FLAGS

# TODO(sauravshekhar) add entire FW Optimization process here.
# moving code from scripts/mixture_model_relbo.py
flags.DEFINE_integer(
    'n_monte_carlo_samples', 10,
    'Number of samples for approximating gradient')
flags.DEFINE_integer(
    'n_line_search_iter', 10,
    'Number of iterations line search gradient descent')
flags.DEFINE_enum('linit', 'fixed',
                  ['fixed', 'lipschitz_v2', 'lipschitz_v1'],
                  'Initialization methods versions for lipschitz constant')
flags.DEFINE_float('linit_fixed', 0.001,
                   'Fixed initial estimate of Lipschitz constant')
flags.DEFINE_integer('adafw_MAXITER', 32,
                     'Maximum iterations of adaptive fw L estimation')
flags.DEFINE_float('damping_adafw', 0.99, 'Damping constant')
flags.DEFINE_float('exp_adafw', 2.0, 'Multiplicative factor in L')

def adafw_linit(q_0, p):
    """Initialization of L estimate for Adaptive
    Frank Wolfe algorithm. Given in v2 of the
    paper https://arxiv.org/pdf/1806.05123.pdf

    Args:
        q_0: initial iterate
        p: target distribution
    Returns:
        L initialized value, float
    """
    if FLAGS.linit == 'fixed':
        return FLAGS.linit_fixed
    elif FLAGS.linit != 'lipschitz_v2':
        raise NotImplementedError('v1 not implemented')

    # larger sample size for more accuracy
    N_samples = FLAGS.n_monte_carlo_samples * 5
    theta = q_0.sample([N_samples])
    # grad_q0 = grad_kl(q_0, p, theta).eval()
    log_q0 = q_0.log_prob(theta).eval()
    log_p = p.log_prob(theta).eval()
    grad_q0 = log_q0 - log_p
    prob_q0 = q_0.prob(theta).eval()

    def get_diff(L):
        h = -1.*grad_q0 / L
        # q_0 + h is not a valid probability distribution so values
        # can get negative. Performing clipping before taking log
        t0 = np.clip(prob_q0 + h, 1e-5, None)
        #eprint('t0 range is [%.5f..%.5f] mean: %.5f +- %.5f' % (np.min(t0),
        #                                                        np.max(t0),
        #                                                        np.mean(t0),
        #                                                        np.std(t0)))
        t1 = np.log(t0)
        t2 = np.mean(t1 - log_q0)
        t3 = t1 - log_p
        t4 = (h * t3) / prob_q0
        t5 = np.mean(t4)
        return t2 - t5

    L_init_estimate = FLAGS.linit_fixed
    while get_diff(L_init_estimate) > 0.:
        debug('L estimate diff is %.5f for L %.2f' %
              (get_diff(L_init_estimate), L_init_estimate))
        L_init_estimate *= 10.
    debug('L estimate diff is %.5f for L %.2f' %
            (get_diff(L_init_estimate), L_init_estimate))
    logger.info('initial Lipschitz estimate is %.5f\n' % L_init_estimate)
    return L_init_estimate

# TODO(sauravshekhar) fix arguments
def adaptive_fw(**kwargs):
    """Adaptive Frank-Wolfe algorithm.
    
    Sets step size as suggested in Algorithm 1 of
    https://arxiv.org/pdf/1806.05123.pdf
    """
    fw_iter = kwargs['fw_iter']
    st_tf = kwargs['s_t'] # LMO solution
    mu_s = kwargs['mu_s']
    cov_s = kwargs['cov_s']
    qt_tf = kwargs['q_t'] # current iterate
    locs = kwargs['locs']
    diags = kwargs['diags']
    p = kwargs['p'] # target dist
    weights = kwargs['weights']
    l_prev = kwargs['l_prev']

    # Set $q_{t+1}$'s params
    new_locs = copy.copy(locs)
    new_diags = copy.copy(diags)
    new_locs.append(mu_s)
    new_diags.append(cov_s)

    d_t_norm = divergence(st_tf, qt_tf, metric='dotproduct').eval()
    logger.info('distance norm is %.5f' % d_t_norm)

    if 'step_size_update' in kwargs:
        update_rule = kwargs['step_size_update']
    else:
        update_rule = 'adaptive'

    N_samples = FLAGS.n_monte_carlo_samples
    # create and sample from $s_t, q_t$
    sample_q = qt_tf.sample([N_samples])
    s_t = MultivariateNormalDiag(loc=mu_s, scale_diag=cov_s)
    sample_s = s_t.sample([N_samples])
    step_s = tf.reduce_mean(grad_kl(qt_tf, p, sample_s)).eval()
    step_q = tf.reduce_mean(grad_kl(qt_tf, p, sample_q)).eval()
    gap = step_q - step_s
    logger.info('duality gap %.5f' % gap)
    # FIXME this assertion is failing
    # Removing this condition will lead to NaN values in log probs in
    # further iterations.
    # assert gap >= 0, eprint("Duality gap is negative...")
    if gap < 0: logger.warning("Duality gap is negative returning fixed step")

    gamma = 2. / (fw_iter + 2.)
    # default values in the paper
    tau = FLAGS.exp_adafw
    eta = FLAGS.damping_adafw
    pow_tau = 1.0
    i, l_t = 0, l_prev
    f_t =  kl_divergence(qt_tf, p, allow_nan_stats=False).eval()
    # return intial estimate if gap is -ve
    while gap >= 0:
        # compute $L_t$ and $\gamma_t$
        l_t = pow_tau * eta * l_prev
        if update_rule == 'adaptive':
            gamma = min(gap / (l_t * d_t_norm), 1.0)
        else:
            raise NotImplementedError('other updates not added, consult '
                    'demyanov et al 1970, fabian 2018 etc for other options.')
        d_1 = - gamma * gap
        d_2 = gamma * gamma * l_t * d_t_norm / 2.
        quad_bound_rhs = f_t  + d_1 + d_2

        # $w_{t + 1} = [(1 - \gamma)w_t, \gamma]$
        new_weights = copy.copy(weights)
        new_weights = [(1. - gamma) * w for w in new_weights]
        new_weights.append(gamma)
        qt_new = Mixture(
            cat=Categorical(probs=tf.convert_to_tensor(new_weights)),
            components=[
                MultivariateNormalDiag(loc=loc, scale_diag=diag)
                for loc, diag in zip(new_locs, new_diags)
            ])
        quad_bound_lhs = kl_divergence(qt_new, p, allow_nan_stats=False).eval()
        logger.info('lt = %.5f, gamma = %.3f, f_(qt_new) = %.5f, '
                    'linear extrapolated = %.5f' % (l_t, gamma, quad_bound_lhs,
                                                    quad_bound_rhs))
        if quad_bound_lhs <= quad_bound_rhs:
            break
        i += 1
        if i > FLAGS.adafw_MAXITER:
            # estimate not good
            gamma = 2. / (fw_iter + 2.)
            l_t = l_prev
            break
        pow_tau *= tau

    if 'return_l' in kwargs and kwargs['return_l']:
        return {'gamma': gamma, 'l_estimate': l_t, 'gap': gap}
    return gamma


def line_search_dkl(weights, locs, diags, mu_s, cov_s, p, k, outdir):
    """Performs line search for the best step size gamma.
    
    Uses gradient ascent to find gamma that minimizes
    KL(q_t + gamma (s - q_t) || p)
    
    Args:
        weights: [k], weights of mixture components of q_t
        locs: [k x dim], means of mixture components of q_t
        diags: [k x dim], deviations of mixture components of q_t
        mu_s: [dim], mean for LMO Solution s
        cov_s: [dim], cov matrix for LMO solution s
        p: edward.model, target distribution p
        k: iteration number of Frank-Wolfe
        outdir: directory to put output
    Returns:
       Computed gamma
    """
    N_samples = FLAGS.n_monte_carlo_samples
    # Create current iter $q_t$
    qt = Mixture(
        cat=Categorical(probs=tf.convert_to_tensor(weights)),
        components=[
            MultivariateNormalDiag(loc=loc, scale_diag=diag)
            for loc, diag in zip(locs, diags)
        ])
    # sample from $q_t$ and s
    sample_q = qt.sample([N_samples])
    s_t = MultivariateNormalDiag(loc=mu_s, scale_diag=cov_s)
    sample_s = s_t.sample([N_samples])
    # set $q_{t+1}$'s parameters
    new_locs = copy.copy(locs)
    new_diags = copy.copy(diags)
    new_locs.append(mu_s)
    new_diags.append(cov_s)
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
        qt_new = Mixture(
            cat=Categorical(probs=tf.convert_to_tensor(new_weights)),
            components=[
                MultivariateNormalDiag(loc=loc, scale_diag=diag)
                for loc, diag in zip(new_locs, new_diags)
            ])
        rez_s = grad_kl(qt_new, p, sample_s).eval()
        rez_q = grad_kl(qt_new, p, sample_q).eval()
        grad_gamma.append({'E_s': rez_s, 'E_q': rez_q, 'gamma': gamma})
        # Gradient descent step size decreasing as $\frac{1}{it + 1}$
        gamma = gamma - 0.1 * (np.mean(rez_s) - np.mean(rez_q)) / (it + 1.)
        # Projecting it back to [0, 1]
        if gamma >= 1 or gamma <= 0:
            gamma = max(min(gamma, 1.), 0.)
            break
    goutdir = setup_outdir(os.path.join(outdir, 'gradients'))
    g_outfile = os.path.join(goutdir,
                             'line_search_samples_%d.npy.%d' % (N_samples, k))
    logger.info('saving line search data to, %s' % g_outfile)
    np.save(open(g_outfile, 'wb'), grad_gamma)
    return gamma


def fully_corrective(q, p):
    """Fully corrective FW strategy"""
    comps = q.components
    n_comps = len(comps)
    # randomly initialize, rather than take q.cat as the initialization
    weights = np.random.random(n_comps).astype(np.float32)
    weights /= np.sum(weights)
    n_samples = 1000
    samples = [comp.sample(n_samples) for comp in comps]
    S = []
    for j in range(n_comps):
        for i in range(n_comps):
            comp_log_prob = tf.squeeze(comps[i].log_prob(samples[j]))
            S.append(comp_log_prob)
    S = tf.transpose(tf.reshape(tf.stack(S), [n_comps, n_comps, n_samples]))
    S = S.eval()

    p_log_probs = [tf.squeeze(p.log_prob(i)).eval() for i in samples]

    T = 1000000
    for t in range(T):
        grad = np.zeros(n_comps)
        for i in range(n_comps):
            part = np.zeros([n_samples, n_comps])
            for j in range(n_comps):
                probs = S[:, j, i]
                part[:, j] = np.log(weights[j] + 1e-10) + probs
            part = logsumexp(part, axis=1)
            diff = part - p_log_probs[i]
            grad[i] = np.mean(diff, axis=0)
        min_i = np.argmin(grad)
        corner = np.zeros(weights.shape)
        corner[min_i] = 1
        if t % 1000 == 0:
            print("grad", grad)
        duality_gap = -np.dot(grad, (corner - weights))
        if t % 1000 == 0:
            print("duality_gap", duality_gap)
        if duality_gap < 1e-6:
            return weights
        weights += 2. / (t + 2.) * (corner - weights)
        if t % 1000 == 0:
            print("weights", weights, t)
    print("weights", weights, t)
    return weights
