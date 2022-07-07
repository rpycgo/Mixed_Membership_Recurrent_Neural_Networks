from ...config.config import model_config
from ..layers.bias_vectors import BiasVectorsBlock

import tensorflow as tf
from tensorflow.keras.layers import Layer, LSTM, Dense, RepeatVector


class RNN(Layer):
    def __init__(self, config=model_config, **kwargs):
        super(RNN, self).__init__(**kwargs)
        self.config = config
                        
        self.bias_vectors = BiasVectorsBlock(name='group_bias_vector')
        self.rho = Dense(units=1, activation='sigmoid')

        self.function = eval(f'tf.nn.{config.function}')

    def build(self, input_shape):
        self.rnn = LSTM(units=input_shape[-1], return_sequences=True)
        self.repeat_vector = RepeatVector(input_shape[1])

    def call(self, inputs, training=None):
        rnn_output = self.rnn(inputs)
        _group_level_bias_vectors = self.bias_vectors(inputs)
        group_level_bias_vectors = self.repeat_vector(_group_level_bias_vectors)
        rho = self.rho(inputs)
        sigma = rho*rnn_output + (1-rho)*group_level_bias_vectors

        return self.function(sigma, axis=-1)
