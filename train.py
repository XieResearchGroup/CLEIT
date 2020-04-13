import os
import tensorflow as tf
import module
import pandas as pd
from tensorflow import keras
from utils import *
from loss import *
from collections import defaultdict
from sklearn.model_selection import KFold


def pre_train_gex_AE(auto_encoder, train_dataset, val_dataset,
                     batch_size=64,
                     optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                     loss_fn=keras.losses.MeanSquaredError(),
                     min_epochs=10,
                     max_epochs=100,
                     tolerance=10,
                     diff_threshold=1e-3):
    output_folder = repr(auto_encoder.encoder) + '_encoder_weights'
    safe_make_dir(output_folder)

    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)

    train_mse_metric = keras.metrics.MeanSquaredError()
    train_mae_metric = keras.metrics.MeanAbsoluteError()
    val_mse_metric = keras.metrics.MeanSquaredError()
    val_mae_metric = keras.metrics.MeanAbsoluteError()

    train_mse_list = []
    train_mae_list = []
    val_mse_list = []
    val_mae_list = []
    best_loss = float('inf')
    tolerance_count = 0

    for epoch in range(max_epochs):
        print('epoch: ', epoch)
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                preds = auto_encoder(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, preds)
                loss_value += sum(auto_encoder.losses)
                grads = tape.gradient(loss_value, auto_encoder.trainable_variables)
                optimizer.apply_gradients(zip(grads, auto_encoder.trainable_variables))
                train_mse_metric(y_batch_train, preds)
                train_mae_metric(y_batch_train, preds)

            if (step + 1) % 100 == 0:
                print('Training loss (for one batch) at step %s: %s' % (step + 1, float(loss_value)))
                print('Seen so far: %s samples' % ((step + 1) * batch_size))

        train_mse = train_mse_metric.result().numpy()
        train_mae = train_mae_metric.result().numpy()
        train_mse_list.append(train_mse)
        train_mae_list.append(train_mae)
        train_mse_metric.reset_states()
        train_mae_metric.reset_states()

        for x_batch_val, y_batch_val in val_dataset:
            val_preds = auto_encoder(x_batch_val, training=True)
            val_mse_metric(y_batch_val, val_preds)
            val_mae_metric(y_batch_val, val_preds)
        val_mse = val_mse_metric.result().numpy()
        val_mae = val_mae_metric.result().numpy()
        val_mse_list.append(val_mse)
        val_mae_list.append(val_mae)
        val_mae_metric.reset_states()
        val_mse_metric.reset_states()

        if val_mse < best_loss:
            auto_encoder.encoder.save_weights(os.path.join(output_folder, 'pre_trained_gex_encoder_weights'),
                                              save_format='tf')
            if val_mse + diff_threshold < best_loss:
                tolerance_count = 0
            else:
                tolerance_count += 1
            best_loss = val_mse
        else:
            tolerance_count += 1

        if epoch < min_epochs:
            tolerance_count = 0
        else:
            if tolerance_count > tolerance:
                auto_encoder.encoder.load_weights(os.path.join(output_folder, 'pre_trained_gex_encoder_weights'))
                break

    return auto_encoder.encoder, pd.DataFrame({
        'train_mse': train_mse_list,
        'train_mae': train_mae_list,
        'val_mse': val_mse_list,
        'val_mae': val_mae_list
    })


def fine_tune_gex_autoencoder(encoder, raw_X,
                              target_df,
                              validation_X=None,
                              validation_target_df=None,
                              mlp_architecture=[64, 32],
                              mlp_output_act_fn=keras.activations.sigmoid,
                              mlp_output_dim=1,
                              optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                              loss_fn=penalized_mean_squared_error,
                              max_epoch=100
                              ):
    output_folder = repr(encoder) + '_encoder_weights'
    safe_make_dir(output_folder)

    gex_supervisor_dict = dict()
    best_epoch_per_drug = dict()
    best_metric_per_drug = dict()

    training_history = {
        'train_pearson': defaultdict(list),
        'train_spearman': defaultdict(list),
        'train_mse': defaultdict(list),
        'train_mae': defaultdict(list)
    }

    validation_history = {
        'val_pearson': defaultdict(list),
        'val_spearman': defaultdict(list),
        'val_mse': defaultdict(list),
        'val_mae': defaultdict(list)
    }

    for drug in target_df.columns:
        gex_supervisor_dict[drug] = module.MLPBlock(architecture=mlp_architecture, output_act_fn=mlp_output_act_fn,
                                                    output_dim=mlp_output_dim)
        best_epoch_per_drug[drug] = 0
        best_metric_per_drug[drug] = 0

    for epoch in range(max_epoch):
        for drug in target_df.columns:
            model = keras.Sequential()
            model.add(encoder)
            model.add(gex_supervisor_dict[drug])

            y = target_df.loc[~target_df[drug].isna(), drug]
            y = y.astype('float32')
            X = raw_X.loc[y.index]
            X = X.astype('float32')

            if validation_X is None:
                kfold = KFold(n_splits=5, shuffle=True, random_state=2020)
                cv_splits = list(kfold.split(X))
                train_index, test_index = cv_splits[0]

                train_X, train_Y = X.iloc[train_index], y.iloc[train_index]
                assert all(train_X.index == train_Y.index)
                val_X, val_Y = X.iloc[test_index], y.iloc[test_index]
                assert all(val_X.index == val_Y.index)
                train_X, train_Y = train_X.values, train_Y.values
                val_X, val_Y = val_X.values, val_Y.values

            else:
                pass

            with tf.GradientTape() as tape:
                preds = tf.squeeze(model(train_X, training=True))
                loss_value = loss_fn(y_pred=preds, y_true=train_Y)
                print('Training loss (for %s) at epoch %s: %s' % (drug, epoch + 1, float(loss_value)))

                loss_value += sum(model.losses)
                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                training_history['train_pearson'][drug].append(
                    pearson_correlation(y_pred=preds, y_true=train_Y).numpy())
                training_history['train_spearman'][drug].append(
                    spearman_correlation(y_pred=preds, y_true=train_Y).numpy())
                training_history['train_mse'][drug].append(
                    keras.losses.mean_squared_error(y_pred=preds, y_true=train_Y).numpy())
                training_history['train_mae'][drug].append(
                    keras.losses.mean_absolute_error(y_pred=preds, y_true=train_Y).numpy())

            val_preds = tf.squeeze(model(val_X, training=False))
            validation_history['val_pearson'][drug].append(pearson_correlation(y_pred=val_preds, y_true=val_Y).numpy())
            validation_history['val_spearman'][drug].append(
                spearman_correlation(y_pred=val_preds, y_true=val_Y).numpy())
            validation_history['val_mse'][drug].append(
                keras.losses.mean_squared_error(y_pred=val_preds, y_true=val_Y).numpy())
            validation_history['val_mae'][drug].append(
                keras.losses.mean_absolute_error(y_pred=val_preds, y_true=val_Y).numpy())

            if abs(validation_history['val_pearson'][drug][-1]) > best_metric_per_drug[drug]:
                best_epoch_per_drug[drug] = epoch
                best_metric_per_drug[drug] = abs(validation_history['val_pearson'][drug][-1])
                encoder.save_weights(os.path.join(output_folder, drug + '_fine_tuned_gex_encoder_weights'),
                                     save_format='tf')
                gex_supervisor_dict[drug].save_weights(os.path.join(output_folder, drug + '_regressor_weights'),
                                                       save_format='tf')

    return best_epoch_per_drug, training_history, validation_history
