import tensorflow as tf
from tensorflow import keras


def dense_layer(units, kernel_regularizer_l=0.001, apply_batchnorm=True, apply_dropout=False):
    kernel_initializer = tf.initializers.he_normal()
    result = keras.Sequential()
    if kernel_regularizer_l:
        kernel_regularizer = keras.regularizers.l2(kernel_regularizer_l)
    else:
        kernel_regularizer = None
    result.add(tf.keras.layers.Dense(units,
                                     kernel_regularizer=kernel_regularizer,
                                     kernel_initializer=kernel_initializer))
    if apply_batchnorm:
        result.add(keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(keras.layers.Dropout())

    result.add(keras.layers.ReLU())

    return result


def Encoder(input_dim, latent_dim, architecture, output_act_fn=None, stochastic_flag=False):
    kernel_initializer = tf.initializers.he_normal()
    inputs = keras.layers.Input(shape=[input_dim])
    intermediate_layers = [dense_layer(units=units) for units in architecture]
    output_layer = keras.layers.Dense(units=latent_dim, activation=output_act_fn, kernel_initializer=kernel_initializer)
    x = inputs

    for layer in intermediate_layers:
        x = layer(x)

    latent_code = output_layer(x)
    if stochastic_flag:
        extra_output_layer = keras.layers.Dense(units=latent_dim, activation=output_act_fn,
                                                kernel_initializer=kernel_initializer)
        latent_sigma = extra_output_layer(x)
        return keras.Model(inputs=inputs, outputs=[latent_code, latent_sigma])

    return keras.Model(inputs=inputs, outputs=latent_code)


def MLP(input_dim, output_dim, architecture, output_act_fn=None, stochastic_flag=False):
    kernel_initializer = tf.initializers.he_normal()
    inputs = keras.layers.Input(shape=[input_dim])
    intermediate_layers = [dense_layer(units=units) for units in architecture]
    output_layer = keras.layers.Dense(units=output_dim, activation=output_act_fn, kernel_initializer=kernel_initializer)

    x = inputs
    for layer in intermediate_layers:
        x = layer(x)
    x = output_layer(x)

    return keras.Model(inputs=inputs, outputs=x)

