import utils
import model_config
from layer import *


class EncoderBlock(keras.Model):
    def __init__(self, latent_dim, architecture, act_fn='relu', output_act_fn=model_config.encoder_output_act_fn,
                 name='encoder',
                 stochastic_flag=False,
                 kernel_regularizer_l=0.001, **kwargs):
        super(EncoderBlock, self).__init__(name=name, **kwargs)
        self.intermediate_layers = []
        self.latent_dim = latent_dim
        self.architecture = architecture
        self.stochastic_flag = stochastic_flag
        for dim in architecture:
            self.intermediate_layers.append(
                DenseLayer(units=dim, activation=act_fn, kernel_regularizer_l=kernel_regularizer_l))
        self.output_layer = keras.layers.Dense(latent_dim, kernel_initializer='he_normal',
                                               bias_initializer=keras.initializers.Constant(value=0.1),
                                               activation=output_act_fn)
        if self.stochastic_flag:
            self.extra_output_layer = keras.layers.Dense(latent_dim, kernel_initializer='he_normal',
                                                         bias_initializer=keras.initializers.Constant(value=0.1),
                                                         activation=keras.activations.relu)

    def __repr__(self):
        if self.stochastic_flag:
            return 'stochastic_' + utils.list_to_repr(self.architecture) + repr(self.latent_dim)
        return utils.list_to_repr(self.architecture) + repr(self.latent_dim)

    def call(self, inputs, training=True):
        for layer in self.intermediate_layers:
            inputs = layer(inputs, training=training)
        # if training is not None:
        #    self.output_layer.trainable = training
        latent_code = self.output_layer(inputs)
        latent_code = tf.nn.l2_normalize(latent_code, axis=1)
        if self.stochastic_flag:
            # if training is not None:
            #    self.extra_output_layer.trainable = training
            extra_latent_code = self.extra_output_layer(inputs)
            # latent_code = SamplingLayer()(((latent_code, extra_latent_code)))
            return latent_code, extra_latent_code
        return latent_code


class MLPBlock(keras.Model):
    def __init__(self, output_dim, architecture, act_fn='relu', output_act_fn=None, name='mlp',
                 kernel_regularizer_l=0.001, **kwargs):
        super(MLPBlock, self).__init__(name=name, **kwargs)
        self.intermediate_layers = []
        self.architecture = architecture
        self.output_dim = output_dim
        for dim in architecture:
            self.intermediate_layers.append(
                DenseLayer(units=dim, activation=act_fn, kernel_regularizer_l=kernel_regularizer_l))
        self.output_layer = keras.layers.Dense(output_dim, kernel_initializer='he_normal',
                                               bias_initializer=keras.initializers.Constant(value=0.1),
                                               activation=output_act_fn)

    def __repr__(self):
        return utils.list_to_repr(self.architecture) + repr(self.output_dim)

    def call(self, inputs, training=True):
        for layer in self.intermediate_layers:
            inputs = layer(inputs, training=training)
        # if training is not None:
        #    self.output_layer.trainable = training
        result = self.output_layer(inputs)
        return result


class MLPBlockWithMask(keras.Model):
    def __init__(self, output_dim, architecture, shared_layer_num=1, act_fn='relu', output_act_fn=None, name='mlpm',
                 kernel_regularizer_l=0.001, **kwargs):
        super(MLPBlockWithMask, self).__init__(name=name, **kwargs)
        self.intermediate_layers = []
        self.architecture = architecture
        self.output_dim = output_dim
        self.shared_layer_num = shared_layer_num

        for dim in architecture[:shared_layer_num]:
            self.intermediate_layers.append(
                DenseLayer(units=dim, activation=act_fn, kernel_regularizer_l=kernel_regularizer_l))
        self.intermediate_layers.append(
            DenseLayer(units=architecture[shared_layer_num] * output_dim, activation=act_fn,
                       kernel_regularizer_l=kernel_regularizer_l))
        for dim in architecture[shared_layer_num + 1:]:
            self.intermediate_layers.append(
                DenseLayerWithMask(num_of_splits=self.output_dim, units_per_split=dim, activation=act_fn,
                                   kernel_regularizer_l=kernel_regularizer_l))
        self.output_layer = DenseLayerWithMask(num_of_splits=self.output_dim, units_per_split=1,
                                               activation=output_act_fn, bn_flag=False)

    def __repr__(self):
        return utils.list_to_repr(self.architecture) + repr(self.output_dim)

    def call(self, inputs, training=True):
        for layer in self.intermediate_layers:
            inputs = layer(inputs, training=training)
        # if training is not None:
        #    self.output_layer.trainable = training
        result = self.output_layer(inputs)
        return result


class Critic(keras.Model):
    def __init__(self, output_dim, architecture, act_fn='elu', output_act_fn=None, name='critic',
                 kernel_regularizer_l=0.001, **kwargs):
        super(Critic, self).__init__(name=name, **kwargs)
        self.intermediate_layers = []
        self.architecture = architecture
        self.output_dim = output_dim
        for dim in architecture:
            self.intermediate_layers.append(
                LayerNormLayer(units=dim, activation=act_fn, kernel_regularizer_l=kernel_regularizer_l))
        self.output_layer = keras.layers.Dense(output_dim, kernel_initializer='he_normal',
                                               bias_initializer=keras.initializers.Constant(value=0.1),
                                               activation=output_act_fn)

    def __repr__(self):
        return utils.list_to_repr(self.architecture) + repr(self.output_dim)

    def call(self, inputs, training=True):
        for layer in self.intermediate_layers:
            inputs = layer(inputs, training=training)
        # if training is not None:
        #    self.output_layer.trainable = training
        result = self.output_layer(inputs)
        return result


class AE(keras.Model):
    def __init__(self, latent_dim, output_dim, architecture, act_fn='relu', output_act_fn=None,
                 kernel_regularizer_l=0.001,
                 noise_fn=None, name='ae', **kwargs):
        super(AE, self).__init__(name=name, **kwargs)
        if noise_fn is not None:
            self.noise_layer = noise_fn(0.001)
        self.encoder = EncoderBlock(latent_dim=latent_dim, architecture=architecture, act_fn=act_fn,
                                    output_act_fn=None,
                                    kernel_regularizer_l=kernel_regularizer_l)
        self.decoder = MLPBlock(output_dim=output_dim, architecture=architecture[::-1], output_act_fn=output_act_fn,
                                act_fn=act_fn,
                                kernel_regularizer_l=kernel_regularizer_l,
                                name='decoder')

    def call(self, inputs, training=True):
        if training is True:
            inputs = self.noise_layer(inputs, training=training)
        latent_code = self.encoder(inputs, training=training)
        output = self.decoder(latent_code, training=training)
        return output


class VAE(keras.Model):
    # beta VAE
    def __init__(self, latent_dim, output_dim, architecture, act_fn='relu', output_act_fn=None, beta=1.0,
                 kernel_regularizer_l=0.001,
                 noise_fn=None, name='ae', **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        if noise_fn is not None:
            self.noise_layer = noise_fn(0.001)
        self.beta = beta
        self.encoder = EncoderBlock(latent_dim=latent_dim, architecture=architecture, output_act_fn=None,
                                    act_fn=act_fn,
                                    kernel_regularizer_l=kernel_regularizer_l,
                                    stochastic_flag=True)
        self.decoder = MLPBlock(output_dim=output_dim, architecture=architecture[::-1], output_act_fn=output_act_fn,
                                act_fn=act_fn,
                                kernel_regularizer_l=kernel_regularizer_l,
                                name='decoder')

    def call(self, inputs, training=True):
        # if training is True:
        #    inputs = self.noise_layer(inputs, training=training)
        latent_mean, latent_log_var = self.encoder(inputs, training=training)
        latent_code = SamplingLayer()(((latent_mean, latent_log_var)))
        # latent_code = self.encoder(inputs, training=training)
        output = self.decoder(latent_code, training=training)

        kl_loss = -0.5 * tf.reduce_sum(
            latent_log_var - tf.square(latent_mean) - tf.exp(latent_log_var) + 1)
        self.add_loss(self.beta * kl_loss)

        return output
