"""Thin wrapper over edward.models.Normal to simulate a multivariate Normal.
"""
import tensorflow as tf
from edward.models import Normal

# FIXME not working correctly, batch_shape and event_shape have been
# set as that of a MultiVariateNormalDiag, but _sample_n is still
# causing issues in shapes
class MVNormal(Normal):
    def __init__(self, *args, **kwargs):
        super(MVNormal, self).__init__(*args, **kwargs)

    def _batch_shape_tensor(self):
        return tf.constant([], dtype=dtypes.int32)

    def _batch_shape(self):
        return tf.TensorShape([])

    def _event_shape_tensor(self):
        return tf.shape(self.loc)

    def _event_shape(self):
        return self._loc.get_shape()

    def _log_prob(self, value):
        """Log probability density/mass function
        
        log probability of Normal will be of univariate distributions
        on individual dimensions.

        Args:
            value: single sample of dimensionality (D, ) or multiple samples
                of dimensionality (N, D)
        """
        dim_log_prob = super(MVNormal, self)._log_prob(value)
        # reduce_sum(dim_log_prob, axis=1) won't work
        # in the case of a single sample of shape (D, )
        dim_log_prob_t = tf.transpose(dim_log_prob)
        sum_log_prob = tf.reduce_sum(dim_log_prob_t, axis=0) # (N, ) or ()
        return sum_log_prob


MVNormal._sample_n = Normal._sample_n
