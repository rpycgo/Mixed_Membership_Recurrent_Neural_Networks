from ...config.config import model_config

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, LSTM


class BiasVectorsBlock(Layer):
    def __init__(self, config=model_config, **kwargs):
        super(BiasVectorsBlock, self).__init__(**kwargs)
        self.config = config

    def _cov(self, inputs, axis=-1):
        inputs = inputs - tf.reduce_mean(inputs, axis=axis, keepdims=True)
        fact = tf.cast(inputs.shape[axis] - 1, tf.float32)

        if axis == -1 or axis == 2:
            cov = (inputs @ tf.math.conj(tf.transpose(inputs, perm=(0, 2, 1)))) / fact
        elif axis == 1:
            cov = (tf.transpose(inputs, perm=(0, 2, 1)) @ tf.math.conj(inputs)) / fact

        return tf.reduce_mean(cov, axis=0)

    def call(self, inputs, training=None):
        mean = tf.math.reduce_mean(inputs, axis=[0, 1])
        cov = self._cov(inputs, axis=1)

        return tf.constant(np.random.multivariate_normal(mean, cov, size=self.config.batch_size, check_valid='warn', tol=1e-8))
