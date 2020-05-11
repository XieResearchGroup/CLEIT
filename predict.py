import tensorflow as tf
import module
import model_config
import os
from tensorflow import keras
from utils import *


def predict(inputs, pre_train_flag, transmitter_flag, dat_type, exp_type):
    encoder = module.EncoderBlock(latent_dim=model_config.encoder_latent_dimension,
                                  architecture=model_config.encoder_architecture,
                                  output_act_fn=model_config.encoder_output_act_fn,
                                  act_fn=model_config.encoder_act_fn,
                                  kernel_regularizer_l=model_config.kernel_regularizer_l,
                                  stochastic_flag=True)
    if pre_train_flag:
        if dat_type == 'gex':
            encoder.load_weights(
                os.path.join('saved_weights', dat_type, repr(encoder) + '_encoder_weights',
                             'pre_trained_encoder_weights')
            )
        else:
            encoder.load_weights(
                os.path.join('saved_weights', dat_type, exp_type, repr(encoder) + '_encoder_weights',
                             'pre_trained_encoder_weights')
            )

    else:
        encoder.load_weights(
            os.path.join('saved_weights', dat_type, exp_type, repr(encoder) + '_encoder_weights',
                         'fine_tuned_encoder_weights')
        )

    if transmitter_flag:
        transmitter = module.MLPBlock(architecture=model_config.transmitter_architecture,
                                      act_fn=model_config.transmitter_act_fn,
                                      output_act_fn=model_config.transmitter_output_act_fn,
                                      output_dim=model_config.transmitter_output_dim)
        if pre_train_flag:
            transmitter.load_weights(
                os.path.join('saved_weights', dat_type, exp_type, repr(encoder) + '_encoder_weights',
                             'pre_trained_transmitter_weights')
            )
        else:
            transmitter.load_weights(
                os.path.join('saved_weights', dat_type, exp_type, repr(encoder) + '_encoder_weights',
                             'fine_tuned_transmitter_weights')
            )

    regressor = module.MLPBlockWithMask(architecture=model_config.regressor_architecture,
                                        shared_layer_num=model_config.regressor_shared_layer_number,
                                        act_fn=model_config.regressor_act_fn,
                                        output_act_fn=model_config.regressor_output_act_fn,
                                        output_dim=model_config.regressor_output_dim)

    regressor.load_weights(
        os.path.join('saved_weights', dat_type, exp_type, repr(encoder) + '_encoder_weights', 'regressor_weights'))

    if repr(encoder).startswith('stochastic'):
        encoded_X = encoder(inputs, training=False)[0]
    else:
        encoded_X = encoder(inputs, training=False)

    if transmitter_flag:
        encoded_X = transmitter(encoded_X, training=False)

    preds = regressor(encoded_X, training=False).numpy()
    return preds
