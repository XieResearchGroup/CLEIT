import utils
from layer import *


class EncoderBlock(keras.Model):
    def __init__(self, latent_dim, architecture, output_act_fn=None, name='encoder', stochastic_flag=False, **kwargs):
        super(EncoderBlock, self).__init__(name=name, **kwargs)
        self.intermediate_layers = []
        self.latent_dim = latent_dim
        self.architecture = architecture
        self.stochastic_flag = stochastic_flag
        for dim in architecture:
            self.intermediate_layers.append(DenseLayer(units=dim))
        self.output_layer = keras.layers.Dense(latent_dim, activation=output_act_fn)
        if self.stochastic_flag:
            self.extra_output_layer = keras.layers.Dense(latent_dim, activation=output_act_fn)

    def __repr__(self):
        if self.stochastic_flag:
            return 'stochastic_'+utils.list_to_repr(self.architecture) + repr(self.latent_dim)
        return utils.list_to_repr(self.architecture) + repr(self.latent_dim)

    def call(self, inputs, training=True, **kwargs):
        for layer in self.intermediate_layers:
            inputs = layer(inputs, training=training)
        # if training is not None:
        #    self.output_layer.trainable = training
        latent_code = self.output_layer(inputs)
        if self.stochastic_flag:
            # if training is not None:
            #    self.extra_output_layer.trainable = training
            extra_latent_code = self.extra_output_layer(inputs)
            #latent_code = SamplingLayer()(((latent_code, extra_latent_code)))
            return latent_code, extra_latent_code
        return latent_code


class MLPBlock(keras.Model):
    def __init__(self, output_dim, architecture, act_fn='relu', output_act_fn=None, name='mlp', **kwargs):
        super(MLPBlock, self).__init__(name=name, **kwargs)
        self.intermediate_layers = []
        self.architecture = architecture
        self.output_dim = output_dim
        for dim in architecture:
            self.intermediate_layers.append(DenseLayer(units=dim, activation=act_fn))
        self.output_layer = keras.layers.Dense(output_dim, activation=output_act_fn)

    def __repr__(self):
        return utils.list_to_repr(self.architecture) + repr(self.output_dim)

    def call(self, inputs, training=True, **kwargs):
        for layer in self.intermediate_layers:
            inputs = layer(inputs, training=training)
        # if training is not None:
        #    self.output_layer.trainable = training
        result = self.output_layer(inputs)
        return result



class Critic(keras.Model):
    def __init__(self, output_dim, architecture, act_fn='elu', output_act_fn=None, name='critic', **kwargs):
        super(Critic, self).__init__(name=name, **kwargs)
        self.intermediate_layers = []
        self.architecture = architecture
        self.output_dim = output_dim
        for dim in architecture:
            self.intermediate_layers.append(LayerNormLayer(units=dim, activation=act_fn))
        self.output_layer = keras.layers.Dense(output_dim, activation=output_act_fn)

    def __repr__(self):
        return utils.list_to_repr(self.architecture) + repr(self.output_dim)

    def call(self, inputs, training=True, **kwargs):
        for layer in self.intermediate_layers:
            inputs = layer(inputs, training=training)
        # if training is not None:
        #    self.output_layer.trainable = training
        result = self.output_layer(inputs)
        return result



class AE(keras.Model):
    def __init__(self, latent_dim, output_dim, architecture, output_act_fn=None, noise_fn=None, name='ae', **kwargs):
        super(AE, self).__init__(name=name, **kwargs)
        if noise_fn is not None:
            self.noise_layer = noise_fn(0.005)
        self.encoder = EncoderBlock(latent_dim=latent_dim, architecture=architecture, output_act_fn=output_act_fn)
        self.decoder = MLPBlock(output_dim=output_dim, architecture=architecture[::-1], output_act_fn=output_act_fn,
                                name='decoder')

    def call(self, inputs, training=True, **kwargs):
        if training is True:
            inputs = self.noise_layer(inputs, training=training)
        latent_code = self.encoder(inputs, training=training)
        output = self.decoder(latent_code, training=training)

        return output


class VAE(keras.Model):
    def __init__(self, latent_dim, output_dim, architecture, output_act_fn=None, noise_fn=None, name='ae', **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        if noise_fn is not None:
            self.noise_layer = noise_fn(0.1)
        self.encoder = EncoderBlock(latent_dim=latent_dim, architecture=architecture, output_act_fn=output_act_fn,
                                    stochastic_flag=True)
        self.decoder = MLPBlock(output_dim=output_dim, architecture=architecture[::-1], output_act_fn=output_act_fn,
                                name='decoder')

    def call(self, inputs, training=True, **kwargs):
        # if training is True:
        #    inputs = self.noise_layer(inputs, training=training)
        latent_mean, latent_log_var = self.encoder(inputs, training=training)
        latent_code = SamplingLayer()(((latent_mean, latent_log_var)))
        #latent_code = self.encoder(inputs, training=training)
        output = self.decoder(latent_code, training=training)

        kl_loss = -0.5 * tf.reduce_sum(
            latent_log_var - tf.square(latent_mean) - tf.exp(latent_log_var) + 1)
        self.add_loss(kl_loss)

        return output
