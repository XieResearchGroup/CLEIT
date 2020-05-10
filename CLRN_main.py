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

from tensorflow import keras
from functools import partial

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Cross-Level information Regularization Network')
    parser.add_mutually_exclusive_group(required=False)
    parser.add_argument('--propagation', dest='propagation', action='store_true')
    parser.add_argument('--no-propagation', dest='propagation', action='store_false')
    parser.set_defaults(propagation=True)
    parser.add_argument('--target', dest='target', nargs='?', default='AUC', choices=['AUC', 'LN_IC50'])
    parser.add_argument('--filter', dest='filter', nargs='?', default='FILE', choices=['MAD', 'FILE'])
    parser.add_argument('--feat_num', dest='feature_number', nargs='?', default=5000)
    parser.add_argument('--clr_fn', dest='clr_fn', nargs='?', default='contrastive', choices=['contrastive', 'mmd','wgan'])
    parser.add_argument('--gpu', dest='gpu', type=int, nargs='?', default=0)
    parser.add_argument('--exp_type', dest='exp_type', nargs='?', default='cn', choices=['cv', 'test'])


    args = parser.parse_args()
    data_provider = data.DataProvider(feature_filter=args.filter, target=args.target,
                                      feature_number=args.feature_number,
                                      omics=['gex', 'mut'])

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

    if args.exp_type == 'cv':
        for i in range(len(data_provider.get_k_folds())):
            pass
    else:
        pass

