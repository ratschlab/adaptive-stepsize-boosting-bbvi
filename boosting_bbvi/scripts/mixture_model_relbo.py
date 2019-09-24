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
import tensorflow as tf
import edward as ed
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import boosting_bbvi.core.relbo as relbo
import boosting_bbvi.core.utils as coreutils
import boosting_bbvi.optim.fw as optim
from boosting_bbvi.core.utils import eprint, debug
logger = coreutils.get_logger()


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('outdir', '/tmp',
                    'directory to store all the results, models, plots, etc.')
flags.DEFINE_integer('seed', 0, 'The random seed to use for everything.')
flags.DEFINE_string('exp', 'mixture',
                    'select from [mixture, s_and_s (aka spike and slab), many]')
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
    n_features = 2
    mus = np.tile(mus, [1, n_features]).astype(np.float32)
    stds = np.tile(stds, [1, n_features]).astype(np.float32)
elif FLAGS.exp == 'mixture_nd':
    p = np.random.dirichlet([1, 4, 10, 5])
    pi = p[np.newaxis, :].astype(np.float32)
    n_features = 10
    mus = np.random.rand(4, n_features).astype(np.float32)*3 - 1.
    stds = np.random.rand(4, n_features).astype(np.float32)
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


def main(argv):
    del argv

    # create dataset
    #if FLAGS.exp.endswith('2d'):
    #    x_train, components = build_toy_dataset(N, D=2)
    #else:
    #    x_train, components = build_toy_dataset(N)

    #n_examples, n_features = x_train.shape

    outdir = FLAGS.outdir
    if '~' in outdir: outdir = os.path.expanduser(outdir)
    os.makedirs(outdir, exist_ok=True)

    # NOTE: in this case the joint model p(x, z) is the same 
    # as the target posterior p(z | x). Save the target
    np.savez(
        os.path.join(outdir, 'target_dist.npz'), pi=pi, mus=mus, stds=stds)
    
    # Run frank-wolfe
    boosted_bbvi = optim.FWOptimizer()
    boosted_bbvi.run(outdir, pi, mus, stds, n_features)


if __name__ == "__main__":
    tf.app.run()
