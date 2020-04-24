import tensorflow as tf
from tensorflow import keras


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
                                              kernel_regularizer=self.kernel_regularizer)
        self.bn_layer = keras.layers.BatchNormalization()
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
