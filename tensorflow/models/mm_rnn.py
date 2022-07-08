from ...config.config import model_config
from ..layers.rnn import RNN

import tensorflow as tf
from tensorflow.keras.layers import 
from tensorflow.keras.models import Model


class MMRNN(Model):
    def __init__(self, config=model_config, **kwargs):
        super(self, MMRNN).__init__()
        self.config = config

        self.rnn = RNN(name='rnn')

    def call(self, inputs):
        sigma = self.rnn(inputs)
        
        mean = tf.math.reduce_mean(sigma, axis=[0, -1])
        std = tf.stack([tf.math.reduce_std(sigma[:, t, :]) for t in range(inputs.shape[1])])

        return tf.compat.v1.distributions.Normal(loc=mean, scale=std).sample()
