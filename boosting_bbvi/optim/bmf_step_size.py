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
import boosting_bbvi.core.elbo as elboModel
import boosting_bbvi.core.utils as coreutils
logger = coreutils.get_logger()

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer(
    'n_monte_carlo_samples', 1000,
    'Number of samples for approximating gradient')
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
    'distance_metric', 'kl', ['dotproduct', 'kl', 'constant'],
    'Metric to use for distance norm between probability distrbutions')

MIN_GAMMA = 0.01

#step_result = opt.fixed(weights, qUVt_components, qUV_prev,
#                        loc_s, scale_s, sUV, UV, data, t)
#elbo_loss = elboModel.KLqp({UV: qUV_new}, data={R: R_true, I: I_train})
#res_update = elbo_loss.run()
def fixed(weights, params, q_t, mu_s, cov_s, s_t, pz, data, k, gap=None):
    """Fixed step size.
    
    Args:
        weights: [k] weights of the mixture components of q_t
        params: dictionary containing mixture parameters ('loc', 'scale')
        q_t: current solution iterate
        mu_s: [dim], mean of LMO solution s
        cov_s: [dim], cov matrix for LMO solution s
        s_t: Current atom & LMO Solution s
        pz: latent variable distribution (UV)
        data: training data
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

    return {
        'gamma': gamma,
        'weights': new_weights,
        'params': new_params,
        'step_type': 'fixed'
    }


