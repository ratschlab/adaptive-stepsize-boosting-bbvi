"""Thin wrapper over edward.models.Laplace to simulate a multivariate laplace.
"""
import tensorflow as tf
from edward.models import Laplace

# FIXME not working correctly, batch_shape and event_shape have been
# set as that of a MultiVariateNormalDiag, but _sample_n is still
# causing issues in shapes

# TODO NOTE FIXME See if VectorLaplace is a correct generalization
# for edward.models.Laplace
class MVLaplace(Laplace):
    def __init__(self, *args, **kwargs):
        super(MVLaplace, self).__init__(*args, **kwargs)

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
        
        log probability of Laplace will be of univariate distributions
        on individual dimensions.
        """
        dim_log_prob = super(MVNormal, self).log_prob(value)
        return tf.reduce_sum(dim_log_prob, axis=1)

    #def _sample_n(self, n, seed=None):
    #    super(MVNormal, self)._sample_n(n, seed)
