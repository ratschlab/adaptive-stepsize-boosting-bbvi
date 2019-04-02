"""Thin wrapper over Normal to be used as a Multivariate Normal."""

import tensorflow as tf
from edward.models import Normal


class mvn(Normal):
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
        dim_log_prob = super(mvn, self)._log_prob(value)  # (dim, ) or (N, dim)
        dim_log_prob_t = tf.transpose(dim_log_prob)  # (dim, ) or (dim, N)
        sum_log_prob = tf.reduce_sum(dim_log_prob_t, axis=0)  # () or (N, )
        return sum_log_prob

    def _sample_n(self, n, seed=None):
        new_shape = tf.concat([[n], self.event_shape_tensor()], 0)
        sample = tf.random_normal(
            new_shape,
            seed=seed,
            dtype=self.loc.dtype,
            mean=self.loc,
            stddev=self.scale)
        return tf.cast(sample, self.dtype)
