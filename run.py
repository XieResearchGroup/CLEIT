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
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            # for gpu in gpus:
            #    tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[3], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    data_provider = data.DataProvider(feature_filter='FILE',
                                      omics=['gex', 'mut'])
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
                                                                   val_dataset=val_dataset)

    i = 0
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (data_provider.labeled_data['gex'].iloc[data_provider.get_k_folds()[i][0]].values,
         data_provider.labeled_data['target'].iloc[data_provider.get_k_folds()[i][0]].values))
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (data_provider.labeled_data['gex'].iloc[data_provider.get_k_folds()[i][1]].values,
         data_provider.labeled_data['target'].iloc[data_provider.get_k_folds()[i][1]].values))

    gex_fine_tune_train_history, gex_fine_tune_validation_history = train.fine_tune_gex_encoder(
        encoder=gex_encoder,
        train_dataset=train_dataset,
        val_dataset=val_dataset)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (data_provider.unlabeled_data['mut'].loc[data_provider.matched_index].append(data_provider.labeled_data['mut'].iloc[data_provider.get_k_folds()[i][0]]).values,
         data_provider.unlabeled_data['mut'].loc[data_provider.matched_index].append(data_provider.labeled_data['mut'].iloc[data_provider.get_k_folds()[i][0]]).values,
         data_provider.unlabeled_data['gex'].loc[data_provider.matched_index].append(data_provider.labeled_data['gex'].iloc[data_provider.get_k_folds()[i][0]]).values))



    val_dataset = tf.data.Dataset.from_tensor_slices(
        (data_provider.labeled_data['mut'].iloc[data_provider.get_k_folds()[i][1]].values,
         data_provider.labeled_data['mut'].iloc[data_provider.get_k_folds()[i][1]].values,
         data_provider.labeled_data['gex'].iloc[data_provider.get_k_folds()[i][1]].values))

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
                                                                   transmission_loss_fn=loss.contrastive_loss,
                                                                   )
    # # mut_encoder, mut_pre_train_history_df = train.pre_train_mut_AE(mut_auto_encoder, reference_encoder=gex_encoder, train_dataset=train_dataset, val_dataset=val_dataset,transmission_loss_fn=loss.mmd_loss)
    # # mut_encoder, mut_pre_train_history_df = train.pre_train_mut_AE_with_GAN(mut_auto_encoder, reference_encoder=gex_encoder, train_dataset=train_dataset, val_dataset=val_dataset)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (data_provider.labeled_data['mut'].iloc[data_provider.get_k_folds()[i][0]].values,
         data_provider.labeled_data['target'].iloc[data_provider.get_k_folds()[i][0]].values))
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (data_provider.labeled_data['mut'].iloc[data_provider.get_k_folds()[i][1]].values,
         data_provider.labeled_data['target'].iloc[data_provider.get_k_folds()[i][1]].values))

    mut_fine_tune_train_history, mut_fine_tune_validation_history = train.fine_tune_mut_encoder(
        encoder=mut_encoder,
        train_dataset=train_dataset,
        val_dataset=val_dataset
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
