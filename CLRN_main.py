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
import argparse
import predict
import pandas as pd

from tensorflow import keras
from functools import partial

# gex encoder pre-training can be shared
# gex fine-tuning needs to be retrained for different training folds, one time for the mutation-only
# mutation pre-training needs to be retrained for different training folds * different transmission function, one time for each transmission function for the mutation only
# mutation fine-tuning needs to be retrained per pre-training, in addition to fixed encoders (no fine-tuning on encoders)
# no pre-training on encoder, simply fine tuning (labeled data).
# no transmission


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Cross-Level information Regularization Network')
    parser.add_mutually_exclusive_group(required=False)
    parser.add_argument('--propagation', dest='propagation', action='store_true')
    parser.add_argument('--no-propagation', dest='propagation', action='store_false')
    parser.set_defaults(propagation=True)

    parser.add_argument('--transmitter', dest='transmitter_flag', action='store_true')
    parser.add_argument('--no-transmitter', dest='transmitter_flag', action='store_false')
    parser.set_defaults(transmitter_flag=False)

    parser.add_argument('--target', dest='target', nargs='?', default='AUC', choices=['AUC', 'LN_IC50'])
    parser.add_argument('--filter', dest='filter', nargs='?', default='FILE', choices=['MAD', 'FILE'])
    parser.add_argument('--feat_num', dest='feature_number', nargs='?', default=5000)
    parser.add_argument('--clr_fn', dest='clr_fn', nargs='?', default='contrastive',
                        choices=['contrastive', 'mmd', 'wgan'])
    parser.add_argument('--gpu', dest='gpu', type=int, nargs='?', default=0)
    parser.add_argument('--exp_type', dest='exp_type', nargs='?', default='cv', choices=['cv', 'test'])

    args = parser.parse_args()
    data_provider = data.DataProvider(feature_filter=args.filter, target=args.target,
                                      feature_number=args.feature_number,
                                      omics=['gex', 'mut'])

    #unlabeled gex has few small negative value
    data_provider.unlabeled_data['gex'].where(data_provider.unlabeled_data['gex'] > 0, 0, inplace=True)
    assert data_provider.unlabeled_data['gex'].min().min() >= 0

    if args.clr_fn == 'contrastive':
        pre_train_mut_AE_fn = partial(train.pre_train_mut_AE, transmission_loss_fn=loss.contrastive_loss)
    elif args.clr_fn == 'mmd':
        pre_train_mut_AE_fn = partial(train.pre_train_mut_AE, transmission_loss_fn=loss.mmd_loss)
    elif args.clr_fn == 'wgan':
        pre_train_mut_AE_fn = train.pre_train_mut_AE_with_GAN
    else:
        pre_train_mut_AE_fn = partial(train.pre_train_mut_AE, alpha=0.)

    tf.keras.backend.clear_session()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            # for gpu in gpus:
            #    tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[args.gpu], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (data_provider.unlabeled_data['gex'].values, data_provider.unlabeled_data['gex'].values))
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (data_provider.labeled_data['gex'].values, data_provider.labeled_data['gex'].values))

    gex_auto_encoder = VAE(latent_dim=model_config.encoder_latent_dimension,
                           output_dim=data_provider.shape_dict['gex'],
                           architecture=model_config.encoder_architecture,
                           noise_fn=keras.layers.GaussianNoise,
                           output_act_fn=keras.activations.relu,
                           act_fn=model_config.encoder_act_fn,
                           kernel_regularizer_l=model_config.kernel_regularizer_l)

    gex_encoder, gex_pre_train_history_df = train.pre_train_gex_AE(auto_encoder=gex_auto_encoder,
                                                                   train_dataset=train_dataset,
                                                                   val_dataset=val_dataset)

    if args.exp_type == 'cv':

        gex_prediction_df = pd.DataFrame(
            np.full_like(data_provider.labeled_data['target'], fill_value=-1),
            index=data_provider.labeled_data['target'].index,
            columns=data_provider.labeled_data['target'].columns)

        gex_fine_prediction_df = pd.DataFrame(
            np.full_like(data_provider.labeled_data['target'], fill_value=-1),
            index=data_provider.labeled_data['target'].index,
            columns=data_provider.labeled_data['target'].columns)

        mut_prediction_df = pd.DataFrame(
            np.full_like(data_provider.labeled_data['target'], fill_value=-1),
            index=data_provider.labeled_data['target'].index,
            columns=data_provider.labeled_data['target'].columns)

        mut_fine_prediction_df = pd.DataFrame(
            np.full_like(data_provider.labeled_data['target'], fill_value=-1),
            index=data_provider.labeled_data['target'].index,
            columns=data_provider.labeled_data['target'].columns)

        for i in range(len(data_provider.get_k_folds())):
            # gex fine tuning
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (data_provider.labeled_data['gex'].iloc[data_provider.get_k_folds()[i][0]].values,
                 data_provider.labeled_data['target'].iloc[data_provider.get_k_folds()[i][0]].values))
            val_dataset = tf.data.Dataset.from_tensor_slices(
                (data_provider.labeled_data['gex'].iloc[data_provider.get_k_folds()[i][1]].values,
                 data_provider.labeled_data['target'].iloc[data_provider.get_k_folds()[i][1]].values))
            gex_fine_tune_train_history, gex_fine_tune_validation_history = train.fine_tune_gex_encoder(
                encoder=gex_encoder,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                exp_type=args.exp_type
            )
            # mut encoder pre-training
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (data_provider.unlabeled_data['mut'].loc[data_provider.matched_index].append(
                    data_provider.labeled_data['mut'].iloc[data_provider.get_k_folds()[i][0]]).values,
                 data_provider.unlabeled_data['mut'].loc[data_provider.matched_index].append(
                     data_provider.labeled_data['mut'].iloc[data_provider.get_k_folds()[i][0]]).values,
                 data_provider.unlabeled_data['gex'].loc[data_provider.matched_index].append(
                     data_provider.labeled_data['gex'].iloc[data_provider.get_k_folds()[i][0]]).values))

            val_dataset = tf.data.Dataset.from_tensor_slices(
                (data_provider.labeled_data['mut'].iloc[data_provider.get_k_folds()[i][1]].values,
                 data_provider.labeled_data['mut'].iloc[data_provider.get_k_folds()[i][1]].values,
                 data_provider.labeled_data['gex'].iloc[data_provider.get_k_folds()[i][1]].values))

            mut_auto_encoder = VAE(latent_dim=model_config.encoder_latent_dimension,
                                   output_dim=data_provider.shape_dict['mut'],
                                   architecture=model_config.encoder_architecture,
                                   noise_fn=keras.layers.GaussianNoise,
                                   output_act_fn=keras.activations.relu,
                                   act_fn=model_config.encoder_act_fn,
                                   kernel_regularizer_l=model_config.kernel_regularizer_l)

            mut_encoder, mut_pre_train_history_df = pre_train_mut_AE_fn(auto_encoder=mut_auto_encoder,
                                                                        reference_encoder=gex_encoder,
                                                                        train_dataset=train_dataset,
                                                                        val_dataset=val_dataset,
                                                                        exp_type=args.exp_type,
                                                                        transmitter_flag=args.transmitter_flag
                                                                        )
            # mut fine-tuning
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (data_provider.labeled_data['mut'].iloc[data_provider.get_k_folds()[i][0]].values,
                 data_provider.labeled_data['target'].iloc[data_provider.get_k_folds()[i][0]].values))
            val_dataset = tf.data.Dataset.from_tensor_slices(
                (data_provider.labeled_data['mut'].iloc[data_provider.get_k_folds()[i][1]].values,
                 data_provider.labeled_data['target'].iloc[data_provider.get_k_folds()[i][1]].values))

            mut_fine_tune_train_history, mut_fine_tune_validation_history = train.fine_tune_mut_encoder(
                encoder=mut_encoder,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                exp_type=args.exp_type,
                transmitter_flag=args.transmitter_flag
            )

            gex_prediction_df.loc[data_provider.labeled_data['gex'].iloc[data_provider.get_k_folds()[i][1]].index,
            :] = predict.predict(
                data_provider.labeled_data['gex'].iloc[data_provider.get_k_folds()[i][1]].values, pre_train_flag=True,
                transmitter_flag=args.transmitter_flag,
                dat_type='gex',
                exp_type=args.exp_type)

            gex_fine_prediction_df.loc[data_provider.labeled_data['gex'].iloc[data_provider.get_k_folds()[i][1]].index,
            :] = predict.predict(
                data_provider.labeled_data['gex'].iloc[data_provider.get_k_folds()[i][1]].values, pre_train_flag=False,
                transmitter_flag=args.transmitter_flag,
                dat_type='gex',
                exp_type=args.exp_type)

            mut_prediction_df.loc[data_provider.labeled_data['mut'].iloc[data_provider.get_k_folds()[i][1]].index,
            :] = predict.predict(
                data_provider.labeled_data['mut'].iloc[data_provider.get_k_folds()[i][1]].values, pre_train_flag=True,
                transmitter_flag=args.transmitter_flag,
                dat_type='mut',
                exp_type=args.exp_type)

            mut_fine_prediction_df.loc[data_provider.labeled_data['mut'].iloc[data_provider.get_k_folds()[i][1]].index,
            :] = predict.predict(
                data_provider.labeled_data['mut'].iloc[data_provider.get_k_folds()[i][1]].values, pre_train_flag=False,
                transmitter_flag=args.transmitter_flag,
                dat_type='mut',
                exp_type=args.exp_type)

            gex_prediction_df.to_csv(os.path.join('predictions', args.target + '_' + args.clr_fn + '_' + str(
                args.transmitter_flag) + '_gex_pre_prediction.csv'),
                                          index_label='Sample')

            gex_fine_prediction_df.to_csv(os.path.join('predictions', args.target + '_' + args.clr_fn + '_' + str(
                args.transmitter_flag) + '_gex_fine_prediction.csv'),
                                          index_label='Sample')

            mut_prediction_df.to_csv(os.path.join('predictions', args.target + '_' + args.clr_fn + '_' + str(
                args.transmitter_flag) + '_mut_pre_prediction.csv'),
                                          index_label='Sample')

            mut_fine_prediction_df.to_csv(os.path.join('predictions', args.target + '_' + args.clr_fn + '_' + str(
                args.transmitter_flag) + '_mut_fine_prediction.csv'),
                                          index_label='Sample')

    else:
        # gex fine tuning
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (data_provider.labeled_data['gex'].values,
             data_provider.labeled_data['target'].values))

        val_dataset = tf.data.Dataset.from_tensor_slices(
            (data_provider.labeled_data['gex'].values,
             data_provider.labeled_data['target'].values))  # could be changed to None

        gex_fine_tune_train_history, gex_fine_tune_validation_history = train.fine_tune_gex_encoder(
            encoder=gex_encoder,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            exp_type=args.exp_type
        )
        # mut encoder pre-training
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (data_provider.unlabeled_data['mut'].loc[data_provider.matched_index].append(
                data_provider.labeled_data['mut']).values,
             data_provider.unlabeled_data['mut'].loc[data_provider.matched_index].append(
                 data_provider.labeled_data['mut']).values,
             data_provider.unlabeled_data['gex'].loc[data_provider.matched_index].append(
                 data_provider.labeled_data['gex']).values))

        val_dataset = tf.data.Dataset.from_tensor_slices(
            (data_provider.labeled_data['mut'].values,
             data_provider.labeled_data['mut'].values,
             data_provider.labeled_data['gex'].values))  # could be changed to None

        mut_auto_encoder = VAE(latent_dim=model_config.encoder_latent_dimension,
                               output_dim=data_provider.shape_dict['mut'],
                               architecture=model_config.encoder_architecture,
                               noise_fn=keras.layers.GaussianNoise,
                               output_act_fn=keras.activations.relu,
                               act_fn=model_config.encoder_act_fn,
                               kernel_regularizer_l=model_config.kernel_regularizer_l)

        mut_encoder, mut_pre_train_history_df = pre_train_mut_AE_fn(auto_encoder=mut_auto_encoder,
                                                                    reference_encoder=gex_encoder,
                                                                    train_dataset=train_dataset,
                                                                    val_dataset=val_dataset,
                                                                    exp_type=args.exp_type,
                                                                    transmitter_flag=args.transmitter_flag
                                                                    )
        # mut fine-tuning
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (data_provider.labeled_data['mut'].values,
             data_provider.labeled_data['target'].values))
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (data_provider.labeled_test_data['mut'].values,
             data_provider.labeled_test_data['target'].values))

        mut_fine_tune_train_history, mut_fine_tune_validation_history = train.fine_tune_mut_encoder(
            encoder=mut_encoder,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            exp_type=args.exp_type,
            transmitter_flag=args.transmitter_flag
        )

        mut_test_prediction_df = pd.DataFrame(np.full_like(data_provider.labeled_test_data['target'], fill_value=-1),
                                              index=data_provider.labeled_test_data['target'].index,
                                              columns=data_provider.labeled_test_data['target'].columns)

        mut_test_fine_prediction_df = pd.DataFrame(
            np.full_like(data_provider.labeled_test_data['target'], fill_value=-1),
            index=data_provider.labeled_test_data['target'].index,
            columns=data_provider.labeled_test_data['target'].columns)

        mut_test_prediction_df.loc[data_provider.labeled_test_data['mut'].index, :] = predict.predict(
            data_provider.labeled_test_data['mut'].values, pre_train_flag=True, transmitter_flag=args.transmitter_flag,
            dat_type='mut',
            exp_type=args.exp_type)

        mut_test_fine_prediction_df.loc[data_provider.labeled_test_data['mut'].index, :] = predict.predict(
            data_provider.labeled_test_data['mut'].values, pre_train_flag=False, transmitter_flag=args.transmitter_flag,
            dat_type='mut',
            exp_type=args.exp_type)

        mut_test_prediction_df.to_csv(os.path.join('predictions', args.target + '_' + args.clr_fn + '_' + str(
            args.transmitter_flag) + '_mut_only_pre_prediction.csv'),
                                      index_label='Sample')

        mut_test_fine_prediction_df.to_csv(os.path.join('predictions', args.target + '_' + args.clr_fn + '_' + str(
            args.transmitter_flag) + '_mut_only_fine_prediction.csv'),
                                      index_label='Sample')
