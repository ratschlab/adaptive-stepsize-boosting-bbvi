# TODO(sauravshekhar) not working correctly, line search is terminating only
# after 1 iteration even though n_line_search_iter is not 1. Maybe projection
# is going wrong.
"""
Run Black Box relbo

Example usage

python scripts/mixture_model_relbo.py \
        --relbo_reg 1.0 \
        --relbo_anneal linear \
        --fw_variant fixed \
        --outdir=${TD}/2d \
        --n_fw_iter=10 \
        --LMO_iter=20 \
        --exp mixture_2d

"""

import os
import sys
import numpy as np
import time

import tensorflow as tf

from edward.models import (Categorical, Dirichlet, Empirical, InverseGamma,
                           MultivariateNormalDiag, Normal, ParamMixture,
                           Mixture)

import edward as ed

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import boosting_bbvi.core.relbo as relbo
import boosting_bbvi.core.utils as utils
from boosting_bbvi.core.opt_utils import elbo, setup_outdir
import boosting_bbvi.core.opt as opt
from boosting_bbvi.core.utils import eprint, debug
logger = utils.get_logger()


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('outdir', '/tmp',
                    'directory to store all the results, models, plots, etc.')
flags.DEFINE_integer('seed', 0, 'The random seed to use for everything.')
flags.DEFINE_integer('n_fw_iter', 100, '')
flags.DEFINE_integer('LMO_iter', 1000, '')
flags.DEFINE_string('exp', 'mixture',
                    'select from [mixture, s_and_s (aka spike and slab), many]')
flags.DEFINE_enum('metric', 'gamma', ['E_s', 'E_q', 'gamma'], 'metric to plot')
flags.DEFINE_enum(
    'fw_variant', 'fixed', ['fixed', 'line_search', 'fc', 'adafw'],
    '[fixed (default), line_search, fc] The Frank-Wolfe variant to use.')
# flags.DEFINE_string('decay', 'log',
# '[linear, log (default), squared] The decay rate to use for Lambda.')

ed.set_seed(FLAGS.seed)
np.random.seed(FLAGS.seed)

# what we are trying to fit
if FLAGS.exp == 'mixture':
    pi = np.array([[0.4, 0.6]]).astype(np.float32)
    mus = [[1.], [-1.]]
    stds = [[.6], [.6]]
elif FLAGS.exp == 'balanced':
    pi = np.array([[0.5, 0.5]]).astype(np.float32)
    mus = [[1.], [-1.]]
    stds = [[.6], [.6]]
elif FLAGS.exp == 's_and_s':
    pi = np.array([[0.4, 0.6]]).astype(np.float32)
    mus = [[1.], [-1.]]
    stds = [[.1], [10.]]
elif FLAGS.exp == 'many':
    mus = np.array([[5.0], [10.0], [20.0], [-2]]).astype(np.float32)
    stds = np.array([[2], [2], [1], [1]]).astype(np.float32)
    pi = np.array([[1.0 / 3, 1.0 / 4, 1.0 / 4, 1.0 / 6]]).astype(np.float32)
elif FLAGS.exp == 'mixture_2d':
    pi = np.array([[0.4, 0.6]]).astype(np.float32)
    mus = [[1.], [-1.]]
    stds = [[.6], [.6]]
    # create mus and stds for all dimensions, copying for now
    # TODO(sauravshekhar) create asymmetrical mv gaussians
    mus = np.tile(mus, [1, 2])
    stds = np.tile(stds, [1, 2])
else:
    raise KeyError("undefined experiment")

# global settings
N = 500


def build_toy_dataset(N, D=1):
    x = np.zeros((N, D), dtype=np.float32)
    ks = np.zeros(N, dtype=np.int)
    for n in range(N):
        k = np.argmax(np.random.multinomial(1, pi[0]))
        x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))
        ks[n] = k
    return x, ks


def construct_multivariatenormaldiag(dims, iter, name='', sample_shape=N):
    #loc = tf.get_variable(name + "_loc%d" % iter, dims)
    loc = tf.get_variable(
        name + "_loc%d" % iter, initializer=tf.random_normal(dims))
    #scale = tf.nn.softplus(tf.get_variable(name + "_scale%d" % iter, dims))
    scale = tf.nn.softplus(
        tf.get_variable(
            name + "_scale%d" % iter, initializer=tf.random_normal(dims)))
    mvn = MultivariateNormalDiag(
        loc=loc, scale_diag=scale, sample_shape=sample_shape)
    return mvn


def construct_normal(dims, iter, name=''):
    loc = tf.get_variable(
        name + "_loc%d" % iter,
        initializer=tf.random_normal(dims) + np.random.normal())
    scale = tf.get_variable(
        name + "_scale%d" % iter, initializer=tf.random_normal(dims))
    return Normal(loc=loc, scale=tf.nn.softplus(scale))


def main(argv):
    del argv

    if FLAGS.exp.endswith('2d'):
        x_train, components = build_toy_dataset(N, D=2)
    else:
        x_train, components = build_toy_dataset(N)
    n_examples, n_features = x_train.shape

    # save the target
    outdir = setup_outdir(FLAGS.outdir)
    np.savez(os.path.join(outdir, 'target_dist.npz'), pi=pi, mus=mus, stds=stds)

    weights, comps = [], []
    elbos = []
    relbo_vals = []
    times = []
    lipschitz_estimates = [1.0] # TODO compute as suggested in paper
    # p is the target distribution (mu, std)
    # comps are the component atoms of boosting
    # weights are weights given every iter over comps
    # comps and weights make the current boosted iterate
    for iter in range(FLAGS.n_fw_iter):
        g = tf.Graph()
        with g.as_default():
            tf.set_random_seed(FLAGS.seed)
            sess = tf.InteractiveSession()
            with sess.as_default():
                # build model
                pcomps = [
                    MultivariateNormalDiag(
                        loc=tf.convert_to_tensor(mus[i], dtype=tf.float32),
                        scale_diag=tf.convert_to_tensor(
                            stds[i], dtype=tf.float32))
                    for i in range(len(mus))
                ]
                p = Mixture(
                    cat=Categorical(probs=tf.convert_to_tensor(pi[0])),
                    components=pcomps)

                # s is the solution to LMO. It is initialized randomly
                s = construct_normal([n_features], iter, 's')
                if iter > 0:
                    # current iterate (solution until now)
                    qtx = Mixture(
                        cat=Categorical(probs=tf.convert_to_tensor(weights)),
                        components=[MultivariateNormalDiag(**c) for c in comps])
                    fw_iterates = {p: qtx}
                else:
                    fw_iterates = {}

                sess.run(tf.global_variables_initializer())

                total_time = 0
                start_inference_time = time.time()
                # Run inference on relbo to solve LMO problem
                # NOTE: KLqp has a side effect, it is modifying s
                inference = relbo.KLqp(
                    {
                        p: s
                    }, fw_iterates=fw_iterates, fw_iter=iter)
                inference.run(n_iter=FLAGS.LMO_iter)
                # s now contains solution to LMO
                end_inference_time = time.time()

                total_time += end_inference_time - start_inference_time

                if iter > 0:
                    relbo_vals.append(-utils.compute_relbo(
                        s, fw_iterates[p], p, np.log(iter + 1)))

                # compute step size to update the next iterate
                if iter == 0:
                    gamma = 1.
                elif FLAGS.fw_variant == 'fixed':
                    gamma = 2. / (iter + 2.)
                elif FLAGS.fw_variant == 'line_search':
                    logger.warning('Line search might not be working correctly')
                    start_line_search_time = time.time()
                    gamma = opt.line_search_dkl(
                        weights, [c['loc'] for c in comps],
                        [c['scale_diag'] for c in comps], s.loc.eval(),
                        s.stddev().eval(), p, iter, FLAGS.outdir)
                    end_line_search_time = time.time()
                    total_time += end_line_search_time - start_line_search_time
                elif FLAGS.fw_variant == 'fc':
                    gamma = 2. / (iter + 2.)
                elif FLAGS.fw_variant == 'adafw':
                    logger.warning('AdaFW might not be correct')
                    adaptive_iter = opt.adaptive_fw(
                        fw_iter=iter,
                        p=p,
                        weights=weights,
                        l_prev=lipschitz_estimates[-1],
                        s_t=s,
                        mu_s=s.loc.eval(),
                        cov_s=s.stddev().eval(),
                        q_t=qtx,
                        comps=comps,
                        locs=[c['loc'] for c in comps],
                        diags=[c['scale_diag'] for c in comps],
                        return_l=True)
                    gamma = adaptive_iter['gamma']
                    lipschitz_estimates.append(adaptive_iter['l_estimate'])

                comps.append({
                    'loc': s.mean().eval(),
                    'scale_diag': s.stddev().eval()
                })
                weights = utils.update_weights(weights, gamma, iter)

                #print("weights", weights)
                #print("comps", [c['loc'] for c in comps])
                #print("scale_diags", [c['scale_diag'] for c in comps])

                q_latest = Mixture(
                    cat=Categorical(probs=tf.convert_to_tensor(weights)),
                    components=[MultivariateNormalDiag(**c) for c in comps])

                if FLAGS.fw_variant == "fc":
                    start_fc_time = time.time()
                    weights = opt.fully_corrective(q_latest, p)
                    weights = list(weights)
                    for i in reversed(range(len(weights))):
                        w = weights[i]
                        if w == 0:
                            del weights[i]
                            del comps[i]
                    weights = np.array(weights)
                    end_fc_time = time.time()
                    total_time += end_fc_time - start_fc_time

                q_latest = Mixture(
                    cat=Categorical(probs=tf.convert_to_tensor(weights)),
                    components=[MultivariateNormalDiag(**c) for c in comps])

                elbos.append(elbo(q_latest, p))

                outdir = setup_outdir(FLAGS.outdir)

                print("total time", total_time)
                times.append(float(total_time))
                utils.save_times(os.path.join(outdir, 'times.csv'), times)

                elbos_filename = os.path.join(outdir, 'elbos.csv')
                logger.info("iter, %d, elbo, %.2f +/- %.2f" % (iter,
                                                               *elbos[-1]))
                np.savetxt(elbos_filename, elbos, delimiter=',')
                logger.info("saving elbos to, %s" % elbos_filename)

                relbos_filename = os.path.join(outdir, 'relbos.csv')
                np.savetxt(relbos_filename, relbo_vals, delimiter=',')
                logger.info("saving relbo values to, %s" % relbos_filename)

                for_serialization = {
                    'locs': np.array([c['loc'] for c in comps]),
                    'scale_diags': np.array([c['scale_diag'] for c in comps])
                }
                qt_outfile = os.path.join(outdir, 'qt_iter%d.npz' % iter)
                np.savez(qt_outfile, weights=weights, **for_serialization)
                np.savez(
                    os.path.join(outdir, 'qt_latest.npz'),
                    weights=weights,
                    **for_serialization)
                logger.info("saving qt to, %s" % qt_outfile)
        tf.reset_default_graph()


if __name__ == "__main__":
    tf.app.run()
