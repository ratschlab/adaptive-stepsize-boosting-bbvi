"""Thin wrapper over Laplace to be used as a Multivariate Laplace."""

import tensorflow as tf
from edward.models import Laplace


class lpl(Laplace):
    def _batch_shape_tensor(self):
        return tf.constant([], dtype=dtypes.int32)

    def _batch_shape(self):
        return tf.TensorShape([])

    def _event_shape_tensor(self):
        return tf.shape(self.loc)

    def _event_shape(self):
        return self._loc.get_shape()

    def _log_prob(self, value):
        # value can be (dim, ) or (N, dim)
        dim_log_prob = super(lpl, self)._log_prob(value)  # (dim, ) or (N, dim)
        dim_log_prob_t = tf.transpose(dim_log_prob)  # (dim, ) or (dim, N)
        sum_log_prob = tf.reduce_sum(dim_log_prob_t, axis=0)  # () or (N, )
        return sum_log_prob

    def _sample_n(self, n, seed=None):
        new_shape = tf.concat([[n], self.event_shape_tensor()], 0)
        sample = tf.distributions.Laplace(
            loc=self.loc, scale=self.scale).sample(n)
        return tf.cast(sample, self.dtype)
