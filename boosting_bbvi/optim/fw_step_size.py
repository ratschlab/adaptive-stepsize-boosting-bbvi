"""Frank-Wolfe step size selection methods.
"""
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
from boosting_bbvi.optim.utils import divergence, grad_kl, argmax_grad_dotp
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
flags.DEFINE_integer('adafw_MAXITER', 32,
                     'Maximum iterations of adaptive fw L estimation')
flags.DEFINE_float('damping_adafw', 0.99, 'Damping constant')
flags.DEFINE_float('exp_adafw', 2.0, 'Multiplicative factor in L')
flags.DEFINE_enum(
    'distance_metric', 'dotproduct', ['dotproduct', 'kl', 'constant'],
    'Metric to use for distance norm between probability distrbutions')

MIN_GAMMA = 0.001

# TODO(sauravshekhar) The initialization process suggested in
# the paper seems like a heuristic and is complicated in the
# case of probability distributions. FIXME later
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

    logger.warning('AdaFW initializer might not be correct')
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


def adaptive_pfw(weights, comps, locs, diags, q_t, mu_s, cov_s, s_t, p,
                 k, l_prev):
    """
        Adaptive pairwise variant.
    Args:
        same as fixed
    """
    d_t_norm = divergence(s_t, q_t, metric=FLAGS.distance_metric).eval()
    logger.info('distance norm is %.5f' % d_t_norm)

    # Find v_t
    qcomps = q_t.components
    index_v_t, step_v_t = argmax_grad_dotp(p, q_t, qcomps,
                                           FLAGS.n_monte_carlo_samples)
    v_t = qcomps[index_v_t]

    # Pairwise gap
    sample_s = s_t.sample([FLAGS.n_monte_carlo_samples])
    step_s = tf.reduce_mean(grad_kl(q_t, p, sample_s)).eval()
    gap_pw = step_v_t - step_s
    if gap_pw < 0: eprint("Pairwise gap is negative")

    def default_fixed_step(fail_type='fixed'):
        # adaptive failed, return to fixed
        gamma = 2. / (k + 2.)
        new_comps = copy.copy(comps)
        new_comps.append({'loc': mu_s, 'scale_diag': cov_s})
        new_weights = [(1. - gamma) * w for w in weights]
        new_weights.append(gamma)
        return {
            'gamma': 2. / (k + 2.),
            'l_estimate': l_prev,
            'weights': new_weights,
            'comps': new_comps,
            'gap': gap_pw,
            'step_type': fail_type
        }

    logger.info('Pairwise gap %.5f' % gap_pw)

    # Set $q_{t+1}$'s params
    new_locs = copy.copy(locs)
    new_diags = copy.copy(diags)
    new_locs.append(mu_s)
    new_diags.append(cov_s)
    gap = gap_pw
    if gap <= 0:
        return default_fixed_step()
    gamma_max = weights[index_v_t]
    step_type = 'adaptive'

    tau = FLAGS.exp_adafw
    eta = FLAGS.damping_adafw
    pow_tau = 1.0
    i, l_t = 0, l_prev
    f_t =  kl_divergence(q_t, p, allow_nan_stats=False).eval()
    drop_step = False
    debug('f(q_t) = %.5f' % (f_t))
    gamma = 2. / (k + 2)
    while gamma >= MIN_GAMMA and i < FLAGS.adafw_MAXITER:
        # compute $L_t$ and $\gamma_t$
        l_t = pow_tau * eta * l_prev
        gamma = min(gap / (l_t * d_t_norm), gamma_max)

        d_1 = - gamma * gap
        d_2 = gamma * gamma * l_t * d_t_norm / 2.
        debug('linear d1 = %.5f, quad d2 = %.5f' % (d_1, d_2))
        quad_bound_rhs = f_t  + d_1 + d_2

        # construct $q_{t + 1}$
        new_weights = copy.copy(weights)
        new_weights.append(gamma)
        if gamma == gamma_max:
            # hardcoding to 0 for precision issues
            new_weights[index_v_t] = 0
            drop_step = True
        else:
            new_weights[index_v_t] -= gamma
            drop_step = False

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
            new_comps = copy.copy(comps)
            new_comps.append({'loc': mu_s, 'scale_diag': cov_s})
            if drop_step:
                del new_comps[index_v_t]
                del new_weights[index_v_t]
                logger.info("...drop step")
                step_type = 'drop'
            return {
                'gamma': gamma,
                'l_estimate': l_t,
                'weights': new_weights,
                'comps': new_comps,
                'gap': gap,
                'step_type': step_type
            }
        pow_tau *= tau
        i += 1
    
    # gamma below MIN_GAMMA
    logger.warning("gamma below threshold value, returning fixed step")
    return default_fixed_step("fixed_adaptive_MAXITER")


def adaptive_afw(weights, comps, locs, diags, q_t, mu_s, cov_s, s_t, p,
                 k, l_prev):
    """
        Away steps variant
    Args:
        same as fixed
    """
    d_t_norm = divergence(s_t, q_t, metric=FLAGS.distance_metric).eval()
    logger.info('distance norm is %.5f' % d_t_norm)

    # Find v_t
    qcomps = q_t.components
    index_v_t, step_v_t = argmax_grad_dotp(p, q_t, qcomps,
                                           FLAGS.n_monte_carlo_samples)
    v_t = qcomps[index_v_t]

    # Frank-Wolfe gap
    sample_q = q_t.sample([FLAGS.n_monte_carlo_samples])
    sample_s = s_t.sample([FLAGS.n_monte_carlo_samples])
    step_s = tf.reduce_mean(grad_kl(q_t, p, sample_s)).eval()
    step_q = tf.reduce_mean(grad_kl(q_t, p, sample_q)).eval()
    gap_fw = step_q - step_s
    if gap_fw < 0: logger.warning("Frank-Wolfe duality gap is negative")
    # Away gap
    gap_a = step_v_t - step_q
    if gap_a < 0: eprint('Away gap < 0!!!')
    logger.info('fw gap %.5f, away gap %.5f' % (gap_fw, gap_a))

    # Set $q_{t+1}$'s params
    new_locs = copy.copy(locs)
    new_diags = copy.copy(diags)
    # FIXME(sauravshekhar): In case of one component w will be 1.0
    # fix FW direction in that case as w / (1 - w) will cause issues
    if (gap_fw >= gap_a) or (len(comps) == 1):
        # FW direction, proceeds exactly as adafw
        logger.info('Proceeding in FW direction ')
        adaptive_step_type = 'fw'
        gap = gap_fw
        new_locs.append(mu_s)
        new_diags.append(cov_s)
        gamma_max = 1.0
    else:
        # Away direction
        logger.info('Proceeding in Away direction ')
        adaptive_step_type = 'away'
        gap = gap_a
        if weights[index_v_t] < 1.0:
            gamma_max = weights[index_v_t] / (1.0 - weights[index_v_t])
        else:
            gamma_max = 100. # Large value when t = 1

    def default_fixed_step(fail_type='fixed'):
        # adaptive failed, return to fixed
        gamma = 2. / (k + 2.)
        new_comps = copy.copy(comps)
        new_comps.append({'loc': mu_s, 'scale_diag': cov_s})
        new_weights = [(1. - gamma) * w for w in weights]
        new_weights.append(gamma)
        return {
            'gamma': 2. / (k + 2.),
            'l_estimate': l_prev,
            'weights': new_weights,
            'comps': new_comps,
            'gap': gap,
            'step_type': fail_type
        }
    
    if gap <= 0:
        return default_fixed_step()

    tau = FLAGS.exp_adafw
    eta = FLAGS.damping_adafw
    pow_tau = 1.0
    i, l_t = 0, l_prev
    f_t =  kl_divergence(q_t, p, allow_nan_stats=False).eval()
    debug('f(q_t) = %.5f' % (f_t))
    gamma = 2. / (k + 2)
    is_drop_step = False
    while gamma >= MIN_GAMMA and i < FLAGS.adafw_MAXITER:
        # compute $L_t$ and $\gamma_t$
        l_t = pow_tau * eta * l_prev
        # NOTE: Handle extreme values of gamma carefully
        gamma = min(gap / (l_t * d_t_norm), gamma_max)

        d_1 = - gamma * gap
        d_2 = gamma * gamma * l_t * d_t_norm / 2.
        debug('linear d1 = %.5f, quad d2 = %.5f' % (d_1, d_2))
        quad_bound_rhs = f_t  + d_1 + d_2

        # construct $q_{t + 1}$
        if adaptive_step_type == 'fw':
            if gamma == gamma_max:
                # gamma = 1.0, q_{t + 1} = s_t
                new_comps = [{'loc': mu_s, 'scale_diag': cov_s}]
                new_weights = [1.]
                qt_new = MultivariateNormalDiag(loc=mu_s, scale_diag=cov_s)
            else:
                new_comps = copy.copy(comps)
                new_comps.append({'loc': mu_s, 'scale_diag': cov_s})
                new_weights = copy.copy(weights)
                new_weights = [(1. - gamma) * w for w in new_weights]
                new_weights.append(gamma)
                qt_new = Mixture(
                    cat=Categorical(probs=tf.convert_to_tensor(new_weights)),
                    components=[
                        MultivariateNormalDiag(loc=loc, scale_diag=diag)
                        for loc, diag in zip(new_locs, new_diags)
                    ])
        elif adaptive_step_type == 'away':
            new_weights = copy.copy(weights)
            new_comps = copy.copy(comps)
            if gamma == gamma_max:
                # drop v_t
                is_drop_step = True
                logger.info('...drop step')
                del new_weights[index_v_t]
                new_weights = [(1. + gamma) * w for w in new_weights]
                del new_comps[index_v_t]
                # NOTE: recompute locs and diags after dropping v_t
                drop_locs = [c['loc'] for c in new_comps]
                drop_diags = [c['scale_diag'] for c in new_comps]
                qt_new = Mixture(
                    cat=Categorical(probs=tf.convert_to_tensor(new_weights)),
                    components=[
                        MultivariateNormalDiag(loc=loc, scale_diag=diag)
                        for loc, diag in zip(drop_locs, drop_diags)
                    ])
            else:
                is_drop_step = False
                new_weights = [(1. + gamma) * w for w in new_weights]
                new_weights[index_v_t] -= gamma
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
            step_type = "adaptive"
            if adaptive_step_type == "away": step_type = "away"
            if is_drop_step: step_type = "drop"
            return {
                'gamma': gamma,
                'l_estimate': l_t,
                'weights': new_weights,
                'comps': new_comps,
                'gap': gap,
                'step_type': step_type
            }
        pow_tau *= tau
        i += 1

    # adaptive loop failed, return fixed step size
    logger.warning("gamma below threshold value, returning fixed step")
    return default_fixed_step()


def adaptive_fw(weights, locs, diags, q_t, mu_s, cov_s, s_t, p, k, l_prev,
                return_gamma=False):
    """Adaptive Frank-Wolfe algorithm.
    
    Sets step size as suggested in Algorithm 1 of
    https://arxiv.org/pdf/1806.05123.pdf

    Args:
        weights: [k], weights of the mixture components of q_t
        locs: [k x dim], means of mixture components of q_t
        diags: [k x dim], std deviations of mixture components of q_t
        q_t: current mixture iterate q_t
        mu_s: [dim], mean for LMO solution s
        cov_s: [dim], cov matrix for LMO solution s
        s_t: Current atom & LMO Solution s
        p: edward.model, target distribution p
        k: iteration number of Frank-Wolfe
        l_prev: previous lipschitz estimate
        return_gamma: only return the value of gamma
    Returns:
        If return_gamma is True, only the computed value of gamma
        is returned. Else returns a dictionary containing gamma, 
        lipschitz estimate, duality gap and step information
    """

    # Set $q_{t+1}$'s params
    new_locs = copy.copy(locs)
    new_diags = copy.copy(diags)
    new_locs.append(mu_s)
    new_diags.append(cov_s)

    d_t_norm = divergence(s_t, q_t, metric=FLAGS.distance_metric).eval()
    logger.info('distance norm is %.5f' % d_t_norm)

    N_samples = FLAGS.n_monte_carlo_samples
    # create and sample from $s_t, q_t$
    sample_q = q_t.sample([N_samples])
    sample_s = s_t.sample([N_samples])
    step_s = tf.reduce_mean(grad_kl(q_t, p, sample_s)).eval()
    step_q = tf.reduce_mean(grad_kl(q_t, p, sample_q)).eval()
    gap = step_q - step_s
    logger.info('duality gap %.5f' % gap)
    if gap < 0: logger.warning("Duality gap is negative returning 0 step")

    #gamma = 2. / (k + 2.)
    gamma = 0.
    tau = FLAGS.exp_adafw
    eta = FLAGS.damping_adafw
    # did the adaptive loop suceed or not
    step_type = "fixed"
    # NOTE: this is from v1 of the paper, new version
    # replaces multiplicative tau with divisor eta
    pow_tau = 1.0
    i, l_t = 0, l_prev
    f_t =  kl_divergence(q_t, p, allow_nan_stats=False).eval()
    debug('f(q_t) = %.5f' % (f_t))
    # return intial estimate if gap is -ve
    while gap >= 0:
        # compute $L_t$ and $\gamma_t$
        l_t = pow_tau * eta * l_prev
        gamma = min(gap / (l_t * d_t_norm), 1.0)
        d_1 = - gamma * gap
        d_2 = gamma * gamma * l_t * d_t_norm / 2.
        debug('linear d1 = %.5f, quad d2 = %.5f' % (d_1, d_2))
        quad_bound_rhs = f_t  + d_1 + d_2

        # $w_{t + 1} = [(1 - \gamma)w_t, \gamma]$
        # TODO(sauravshekhar): Handle the case of gamma = 1.0
        # separately, weights might not get exactly 0 because
        # of precision issues. 0 wt components should be removed
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
            step_type = "adaptive"
            break
        pow_tau *= tau
        i += 1
        #if i > FLAGS.adafw_MAXITER or gamma < MIN_GAMMA:
        if i > FLAGS.adafw_MAXITER:
            # estimate not good
            #gamma = 2. / (k + 2.)
            gamma = 0.
            l_t = l_prev
            step_type = "fixed_adaptive_MAXITER"
            break

    if return_gamma: return gamma
    return {
        'gamma': gamma,
        'l_estimate': l_t,
        'gap': gap,
        'step_type': step_type
    }


def line_search_dkl(weights, locs, diags, q_t, mu_s, cov_s, s_t, p, k,
                    return_gamma=False):
    """Performs line search for the best step size gamma.
    
    Uses gradient ascent to find gamma that minimizes
    KL(q_t + gamma (s - q_t) || p)
    
    Args:
        weights: [k], weights of mixture components of q_t
        locs: [k x dim], means of mixture components of q_t
        diags: [k x dim], deviations of mixture components of q_t
        q_t: current mixture iterate q_t
        mu_s: [dim], mean for LMO Solution s
        cov_s: [dim], cov matrix for LMO solution s
        s_t: Current atom & LMO Solution s
        p: edward.model, target distribution p
        k: iteration number of Frank-Wolfe
        return_gamma: only return the value of gamma
    Returns:
        If return_gamma is True, only the computed value of
        gamma is returned. Else along with gradient data
        is returned in a dict
    """
    N_samples = FLAGS.n_monte_carlo_samples
    # sample from $q_t$ and s
    sample_q = q_t.sample([N_samples])
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
        gamma_prime = gamma - 0.1 * (np.mean(rez_s) - np.mean(rez_q)) / (it + 1.)
        # Projecting it back to [0, 1]
        if gamma_prime >= 1 or gamma_prime <= 0:
            gamma_prime = max(min(gamma_prime, 1.), 0.)

        if np.abs(gamma - gamma_prime) < 1e-6:
            gamma = gamma_prime
            break

        gamma = gamma_prime

    if return_gamma: return gamma
    return {'gamma': gamma, 'n_samples': N_samples, 'grad_gamma': grad_gamma}


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
