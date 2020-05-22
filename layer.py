import tensorflow as tf
import numpy as np
from tensorflow import keras
from scipy.linalg import block_diag


class DenseLayer(keras.layers.Layer):
    def __init__(self, units, activation='relu', kernel_initializer='he_normal', kernel_regularizer_l=0.001, **kwargs):
        super(DenseLayer, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = kernel_initializer
        if kernel_regularizer_l:
            self.kernel_regularizer = keras.regularizers.l2(kernel_regularizer_l)
        else:
            self.kernel_regularizer = None
        self.dense_layer = keras.layers.Dense(units=self.units,
                                              kernel_initializer=self.kernel_initializer,
                                              bias_initializer=keras.initializers.Constant(value=0.1),
                                              kernel_regularizer=self.kernel_regularizer)
        self.bn_layer = keras.layers.BatchNormalization(renorm=True)
        self.act_layer = keras.layers.Activation(activation=activation)

    def call(self, inputs, training=True):
        # if training is not None:
        #    self.trainable = training
        #    self.dense_layer.trainable = training
        #    self.bn_layer.trainable = training
        output = self.dense_layer(inputs)
        output = self.bn_layer(output, training=training)
        output = self.act_layer(output)

        return output

    def get_config(self):
        config = super(DenseLayer, self).get_config()
        config.update({'units': self.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DenseLayerWithMask(keras.layers.Layer):
    def __init__(self, num_of_splits, units_per_split, activation='relu', kernel_initializer='he_normal',
                 kernel_regularizer_l=0.001, bn_flag=True, **kwargs):
        super(DenseLayerWithMask, self).__init__()
        self.num_of_splits = num_of_splits
        self.units_per_split = units_per_split
        self.kernel_regularizer_l = kernel_regularizer_l
        self.kernel_initializer = kernel_initializer
        self.bn_flag = bn_flag
        if kernel_regularizer_l:
            self.kernel_regularizer = keras.regularizers.l2(kernel_regularizer_l)
        else:
            self.kernel_regularizer = None
        if self.bn_flag:
            self.bn_layer = keras.layers.BatchNormalization()
        self.mask = tf.constant(block_diag(
                *[np.ones(shape=[int(input_shape[-1] // self.num_of_splits), self.units_per_split]) for _ in
                  range(self.num_of_splits)]), dtype=tf.float32, name='mask')
        self.act_layer = keras.layers.Activation(activation=activation)

    def build(self, input_shape):
        self.weight = self.add_weight(name='weight',shape=(input_shape[-1], self.units_per_split * self.num_of_splits),
                                      initializer=self.kernel_initializer,
                                      trainable=True)

        self.bias = self.add_weight(name='bias', shape=(self.units_per_split * self.num_of_splits,),
                                    initializer=keras.initializers.Constant(value=0.1),
                                    regularizer=self.kernel_regularizer,
                                    trainable=True)

    def call(self, inputs, training=True, **kwargs):
        if self.kernel_regularizer_l is not None:
            self.add_loss(self.kernel_regularizer_l*tf.norm(self.weight * self.mask, ord=2))
        output = tf.matmul(inputs, self.weight * self.mask) + self.bias
        if self.bn_flag:
            output = self.bn_layer(output, training=training)
        output = self.act_layer(output)

        return output


    def get_config(self):
        config = super(DenseLayerWithMask, self).get_config()
        config.update({'num_of_splits': self.num_of_splits})
        config.update({'units_per_split': self.units_per_split})

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DropOutLayer(keras.layers.Layer):
    def __init__(self, units, activation='relu', kernel_initializer='he_normal', kernel_regularizer_l=0.001, **kwargs):
        super(DropOutLayer, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = kernel_initializer
        if kernel_regularizer_l:
            self.kernel_regularizer = keras.regularizers.l2(kernel_regularizer_l)
        else:
            self.kernel_regularizer = None
        self.dense_layer = keras.layers.Dense(units=self.units,
                                              kernel_initializer=self.kernel_initializer,
                                              bias_initializer=keras.initializers.Constant(value=0.1),
                                              kernel_regularizer=self.kernel_regularizer)
        self.do_layer = keras.layers.Dropout()
        self.act_layer = keras.layers.Activation(activation=activation)

    def call(self, inputs, training=True):
        # if training is not None:
        #    self.trainable = training
        #    self.dense_layer.trainable = training
        #    self.bn_layer.trainable = training
        output = self.dense_layer(inputs)
        output = self.do_layer(output, training=training)
        output = self.act_layer(output)

        return output

    def get_config(self):
        config = super(DropOutLayer, self).get_config()
        config.update({'units': self.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class LayerNormLayer(keras.layers.Layer):
    def __init__(self, units, activation='relu', kernel_initializer='he_normal', kernel_regularizer_l=0.001, **kwargs):
        super(LayerNormLayer, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = kernel_initializer
        if kernel_regularizer_l:
            self.kernel_regularizer = keras.regularizers.l2(kernel_regularizer_l)
        else:
            self.kernel_regularizer = None
        self.dense_layer = keras.layers.Dense(units=self.units,
                                              kernel_initializer=self.kernel_initializer,
                                              bias_initializer=keras.initializers.Constant(value=0.1),
                                              kernel_regularizer=self.kernel_regularizer)
        self.ln_layer = keras.layers.LayerNormalization()
        self.act_layer = keras.layers.Activation(activation=activation)

    def call(self, inputs, training=True):
        # if training is not None:
        #    self.trainable = training
        #    self.dense_layer.trainable = training
        #    self.bn_layer.trainable = training
        output = self.dense_layer(inputs)
        output = self.ln_layer(output)
        output = self.act_layer(output)

        return output

    def get_config(self):
        config = super(LayerNormLayer, self).get_config()
        config.update({'units': self.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SamplingLayer(keras.layers.Layer):

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        eps = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * eps
