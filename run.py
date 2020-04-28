from tensorflow import keras
import tensorflow as tf
import data
import data_config
import preprocess_ccle_gdsc_utils
import preprocess_xena_utils
import loss
import train
from module import AE, VAE
import os
import pickle
import utils
import model_config

if __name__ == '__main__':
    tf.keras.backend.clear_session()
    data_provider = data.DataProvider(feature_filter='FILE',
                                      omics=['gex', 'mut'])
    with tf.device('/physical_device:GPU:3'):
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (data_provider.unlabeled_data['gex'].values, data_provider.unlabeled_data['gex'].values))
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (data_provider.labeled_data['gex'].values, data_provider.labeled_data['gex'].values))

        gex_auto_encoder = VAE(latent_dim=model_config.encoder_latent_dimension,
                               output_dim=data_provider.shape_dict['gex'],
                               architecture=model_config.encoder_architecture,
                               noise_fn=keras.layers.GaussianNoise,
                               output_act_fn=keras.activations.relu,
                               kernel_regularizer_l=model_config.kernel_regularizer_l)

        gex_encoder, gex_pre_train_history_df = train.pre_train_gex_AE(auto_encoder=gex_auto_encoder,
                                                                       train_dataset=train_dataset,
                                                                       val_dataset=val_dataset,
                                                                       max_epoch=model_config.max_epoch,
                                                                       min_epoch=model_config.min_epoch,
                                                                       batch_size=model_config.batch_size)

        gex_fine_tune_train_history, gex_fine_tune_validation_history = train.fine_tune_gex_encoder(
            encoder=gex_encoder,
            raw_X=data_provider.labeled_data['gex'],
            target_df=data_provider.labeled_data['target'],
            mlp_architecture=model_config.regressor_architecture,
            mlp_output_act_fn=model_config.regressor_act_fn,
            max_epoch=model_config.max_epoch,
            min_epoch=model_config.min_epoch,
            gradual_unfreezing_flag=model_config.gradual_unfreezing_flag,
            unfrozen_epoch=model_config.unfrozen_epoch
        )

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (data_provider.unlabeled_data['mut'].loc[data_provider.matched_index].values,
             data_provider.unlabeled_data['mut'].loc[data_provider.matched_index].values,
             data_provider.unlabeled_data['gex'].loc[data_provider.matched_index].values))

        val_dataset = tf.data.Dataset.from_tensor_slices(
            (data_provider.labeled_data['mut'].values,
             data_provider.labeled_data['mut'].values,
             data_provider.labeled_data['gex'].values))

        mut_auto_encoder = VAE(latent_dim=model_config.encoder_latent_dimension,
                               output_dim=data_provider.shape_dict['mut'],
                               architecture=model_config.encoder_architecture,
                               noise_fn=keras.layers.GaussianNoise,
                               output_act_fn=keras.activations.relu,
                               kernel_regularizer_l=model_config.kernel_regularizer_l)

        mut_encoder, mut_pre_train_history_df = train.pre_train_mut_AE(auto_encoder=mut_auto_encoder,
                                                                       reference_encoder=gex_encoder,
                                                                       train_dataset=train_dataset,
                                                                       val_dataset=val_dataset,
                                                                       max_epoch=model_config.max_epoch,
                                                                       min_epoch=model_config.min_epoch,
                                                                       batch_size=model_config.batch_size,
                                                                       transmission_loss_fn=loss.contrastive_loss,
                                                                       alpha=model_config.alpha
                                                                       )
        # mut_encoder, mut_pre_train_history_df = train.pre_train_mut_AE(mut_auto_encoder, reference_encoder=gex_encoder, train_dataset=train_dataset, val_dataset=val_dataset,transmission_loss_fn=loss.mmd_loss)
        # mut_encoder, mut_pre_train_history_df = train.pre_train_mut_AE_with_GAN(mut_auto_encoder, reference_encoder=gex_encoder, train_dataset=train_dataset, val_dataset=val_dataset)

        mut_fine_tune_train_history, mut_fine_tune_validation_history = train.fine_tune_mut_encoder(
            encoder=mut_encoder,
            reference_encoder=gex_encoder,
            target_df=data_provider.labeled_data['target'],
            raw_X=data_provider.labeled_data['mut'],
            raw_reference_X=data_provider.labeled_data['gex'],
            mlp_architecture=model_config.regressor_architecture,
            mlp_output_act_fn=model_config.regressor_act_fn,
            max_epoch=model_config.max_epoch,
            min_epoch=model_config.min_epoch,
            gradual_unfreezing_flag=model_config.gradual_unfreezing_flag,
            unfrozen_epoch=model_config.unfrozen_epoch,
            transmission_loss_fn=loss.contrastive_loss,
            alpha=model_config.alpha
        )

        # mut_fine_tune_train_history, mut_fine_tune_validation_history = train.fine_tune_mut_encoder(
        #    mut_encoder,reference_encoder=gex_encoder,
        #    target_df=data_provider.labeled_data['target'],
        #    raw_X=data_provider.labeled_data['mut'],raw_reference_X=data_provider.labeled_data['gex'], transmission_loss_fn=loss.mmd_loss)

        # mut_fine_tune_train_history, mut_fine_tune_validation_history = train.fine_tune_mut_encoder_with_GAN(
        #    mut_encoder,
        #    target_df=data_provider.labeled_data['target'],
        #    raw_X=data_provider.labeled_data['mut'])

        utils.safe_make_dir('history')
        with open(os.path.join('history', 'history.pkl'), 'wb') as handle:
            pickle.dump(gex_pre_train_history_df)
            pickle.dump(gex_fine_tune_train_history)
            pickle.dump(gex_fine_tune_validation_history)
            pickle.dump(mut_pre_train_history_df)
            pickle.dump(mut_fine_tune_train_history)
            pickle.dump(mut_fine_tune_validation_history)


