from collections import defaultdict
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow import keras
import module
from loss import *
from utils import *


def pre_train_gex_AE(auto_encoder, train_dataset, val_dataset,
                     batch_size=64,
                     optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                     loss_fn=keras.losses.MeanSquaredError(),
                     min_epochs=10,
                     max_epochs=100,
                     tolerance=10,
                     diff_threshold=1e-3):
    output_folder = os.path.join('saved_weights', 'gex', repr(auto_encoder.encoder) + '_encoder_weights')
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
            val_preds = auto_encoder(x_batch_val, training=False)
            val_mse_metric(y_batch_val, val_preds)
            val_mae_metric(y_batch_val, val_preds)
        val_mse = val_mse_metric.result().numpy()
        val_mae = val_mae_metric.result().numpy()
        val_mse_list.append(val_mse)
        val_mae_list.append(val_mae)
        val_mae_metric.reset_states()
        val_mse_metric.reset_states()

        if val_mse < best_loss:
            auto_encoder.encoder.save_weights(os.path.join(output_folder, 'pre_trained_encoder_weights'),
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
                auto_encoder.encoder.load_weights(os.path.join(output_folder, 'pre_trained_encoder_weights'))
                break

    return auto_encoder.encoder, pd.DataFrame({
        'train_mse': train_mse_list,
        'train_mae': train_mae_list,
        'val_mse': val_mse_list,
        'val_mae': val_mae_list
    })


def fine_tune_gex_encoder(encoder, raw_X,
                              target_df,
                              validation_X=None,
                              validation_target_df=None,
                              mlp_architecture=None,
                              mlp_output_act_fn=keras.activations.sigmoid,
                              mlp_output_dim=1,
                              optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                              loss_fn=penalized_mean_squared_error,
                              validation_monitoring_metric='pearson',
                              max_epoch=100,
                              min_epoch=10,
                              gradual_unfreezing_flag=True,
                              unfrozen_epoch=5
                              ):
    if mlp_architecture is None:
        mlp_architecture = [64, 32]

    output_folder = os.path.join('saved_weights', 'gex', repr(encoder) + '_encoder_weights')
    safe_make_dir(output_folder)

    gex_supervisor_dict = dict()
    best_overall_metric = 0.

    training_history = {
        'train_pearson': defaultdict(list),
        'train_spearman': defaultdict(list),
        'train_mse': defaultdict(list),
        'train_mae': defaultdict(list),
        'train_total': defaultdict(list)
    }

    validation_history = {
        'val_pearson': defaultdict(list),
        'val_spearman': defaultdict(list),
        'val_mse': defaultdict(list),
        'val_mae': defaultdict(list),
        'val_total': defaultdict(list)
    }
    free_layers = len(encoder.layers)

    if gradual_unfreezing_flag:
        encoder.trainable = False

    for drug in target_df.columns:
        gex_supervisor_dict[drug] = module.MLPBlock(architecture=mlp_architecture, output_act_fn=mlp_output_act_fn,
                                                    output_dim=mlp_output_dim)

    for epoch in range(max_epoch):

        total_train_pearson = 0.
        total_train_spearman = 0.
        total_train_mse = 0.
        total_train_mae = 0.

        total_val_pearson = 0.
        total_val_spearman = 0.
        total_val_mse = 0.
        total_val_mae = 0.

        with tf.GradientTape() as tape:
            total_loss = sum(encoder.losses)
            to_train_variables = encoder.trainable_variables
            if gradual_unfreezing_flag:
                if epoch > min_epoch:
                    free_layers -= 1
                    if (epoch - min_epoch) % unfrozen_epoch == 0:
                        free_layers -= 1
                    for i in range(len(encoder.layers) - 1, free_layers - 1, -1):
                        to_train_variables.extend(encoder.layers[i].trainable_variables)
                    if free_layers <= 0:
                        gradual_unfreezing_flag = False
                        encoder.trainable = True

            for drug in target_df.columns[:10]:
                # model = keras.Sequential()
                # model.add(encoder)
                # model.add(gex_supervisor_dict[drug])

                y = target_df.loc[~target_df[drug].isna(), drug]
                y = y.astype('float32')
                X = raw_X.loc[y.index]
                X = X.astype('float32')

                if validation_X is None:
                    kfold = KFold(n_splits=5, shuffle=True, random_state=2020)
                    cv_splits = list(kfold.split(X))
                    train_index, test_index = cv_splits[0]

                    train_X, train_Y = X.iloc[train_index], y.iloc[train_index]
                    # assert all(train_X.index == train_Y.index)
                    val_X, val_Y = X.iloc[test_index], y.iloc[test_index]
                    # assert all(val_X.index == val_Y.index)
                    train_X, train_Y = train_X.values, train_Y.values
                    val_X, val_Y = val_X.values, val_Y.values

                else:
                    pass
                if repr(encoder).startswith('stochastic'):
                    encoded_X = encoder(train_X, training=True)[0]
                else:
                    encoded_X = encoder(train_X, training=True)

                preds = tf.squeeze(gex_supervisor_dict[drug](encoded_X, training=True))
                loss_value = loss_fn(y_pred=preds, y_true=train_Y)

                print('Training loss (for %s) at epoch %s: %s' % (drug, epoch + 1, float(loss_value)))
                train_pearson = pearson_correlation(y_pred=preds, y_true=train_Y).numpy()
                train_spearman = spearman_correlation(y_pred=preds, y_true=train_Y).numpy()
                train_mse = keras.losses.mean_squared_error(y_pred=preds, y_true=train_Y).numpy()
                train_mae = keras.losses.mean_absolute_error(y_pred=preds, y_true=train_Y).numpy()

                total_train_pearson += train_pearson / float(target_df.shape[-1])
                total_train_spearman += train_spearman / float(target_df.shape[-1])
                total_train_mse += train_mse / float(target_df.shape[-1])
                total_train_mae += train_mae / float(target_df.shape[-1])

                training_history['train_pearson'][drug].append(train_pearson)
                training_history['train_spearman'][drug].append(train_spearman)
                training_history['train_mse'][drug].append(train_mse)
                training_history['train_mae'][drug].append(train_mae)

                if repr(encoder).startswith('stochastic'):
                    encoded_val_X = encoder(val_X, training=False)[0]
                else:
                    encoded_val_X = encoder(val_X, training=False)

                val_preds = tf.squeeze(gex_supervisor_dict[drug](encoded_val_X, training=False))

                val_pearson = pearson_correlation(y_pred=val_preds, y_true=val_Y).numpy()
                val_spearman = spearman_correlation(y_pred=val_preds, y_true=val_Y).numpy()
                val_mse = keras.losses.mean_squared_error(y_pred=val_preds, y_true=val_Y).numpy()
                val_mae = keras.losses.mean_absolute_error(y_pred=val_preds, y_true=val_Y).numpy()

                total_val_pearson += val_pearson / float(target_df.shape[-1])
                total_val_spearman += val_spearman / float(target_df.shape[-1])
                total_val_mse += val_mse / float(target_df.shape[-1])
                total_val_mae += val_mae / float(target_df.shape[-1])

                validation_history['val_pearson'][drug].append(val_pearson)
                validation_history['val_spearman'][drug].append(val_spearman)
                validation_history['val_mse'][drug].append(val_mse)
                validation_history['val_mae'][drug].append(val_mae)

                total_loss += loss_value / float(target_df.shape[-1])
                total_loss += sum(gex_supervisor_dict[drug].losses)
                to_train_variables.extend(gex_supervisor_dict[drug].trainable_variables)

            print('Training loss (total) at epoch %s: %s' % (epoch + 1, float(total_loss)))
            training_history['train_total']['pearson'].append(total_train_pearson)
            training_history['train_total']['spearman'].append(total_train_spearman)
            training_history['train_total']['mse'].append(total_train_mse)
            training_history['train_total']['mae'].append(total_train_mae)

            validation_history['val_total']['pearson'].append(total_val_pearson)
            validation_history['val_total']['spearman'].append(total_val_spearman)
            validation_history['val_total']['mse'].append(total_val_mse)
            validation_history['val_total']['mae'].append(total_val_mae)

            if validation_history['val_total'][validation_monitoring_metric][-1] > best_overall_metric:
                best_overall_metric = validation_history['val_total'][validation_monitoring_metric][-1]
                encoder.save_weights(os.path.join(output_folder, 'fine_tuned_encoder_weights'), save_format='tf')
                for drug in target_df.columns:
                    gex_supervisor_dict[drug].save_weights(os.path.join(output_folder, drug + '_regressor_weights'),
                                                           save_format='tf')

            # print(len(to_train_variables))
            grads = tape.gradient(total_loss, to_train_variables)
            optimizer.apply_gradients(zip(grads, to_train_variables))

    return training_history, validation_history


def pre_train_mut_AE(auto_encoder, reference_encoder, train_dataset, val_dataset,
                     transmission_loss_fn,
                     alpha=1.,
                     batch_size=64,
                     optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                     loss_fn=keras.losses.MeanSquaredError(),
                     min_epochs=10,
                     max_epochs=100,
                     tolerance=10,
                     diff_threshold=1e-3):
    output_folder = os.path.join('saved_weights', 'mut', repr(auto_encoder.encoder) + '_encoder_weights')
    safe_make_dir(output_folder)

    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)

    train_loss_history = []
    val_loss_history = []

    best_val_loss = float('inf')
    tolerance_count = 0

    for epoch in range(max_epochs):
        total_train_loss = 0.
        total_train_steps = 0
        total_val_loss = 0.
        total_val_steps = 0
        print('epoch: ', epoch)
        for step, (x_batch_train, y_batch_train, reference_x_batch_train) in enumerate(train_dataset):
            total_train_steps += 1
            with tf.GradientTape() as tape:
                if repr(auto_encoder.encoder).startswith('stochastic'):
                    encoded_X = auto_encoder.encoder(x_batch_train, training=True)[0]
                    reference_encoded_x = reference_encoder(reference_x_batch_train, training=False)[0]
                else:
                    encoded_X = auto_encoder.encoder(x_batch_train, training=True)
                    reference_encoded_x = reference_encoder(reference_x_batch_train, training=False)

                preds = auto_encoder(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, preds)
                loss_value += alpha * transmission_loss_fn(reference_encoded_x, encoded_X)
                total_train_loss += loss_value
                loss_value += sum(auto_encoder.losses)

                grads = tape.gradient(loss_value, auto_encoder.trainable_variables)
                optimizer.apply_gradients(zip(grads, auto_encoder.trainable_variables))

            if (step + 1) % 100 == 0:
                print('Training loss (for one batch) at step %s: %s' % (step + 1, float(loss_value)))
                print('Seen so far: %s samples' % ((step + 1) * batch_size))
        train_loss_history.append(total_train_loss.numpy() / float(total_train_steps))

        for step, (x_batch_val, y_batch_val, reference_x_batch_val) in enumerate(val_dataset):
            if repr(auto_encoder.encoder).startswith('stochastic'):
                encoded_X_val = auto_encoder.encoder(x_batch_val, training=False)[0]
                reference_encoded_x_val = reference_encoder(reference_x_batch_val, training=False)[0]
            else:
                encoded_X_val = auto_encoder.encoder(x_batch_val, training=False)
                reference_encoded_x_val = reference_encoder(reference_x_batch_val, training=False)
            val_preds = auto_encoder(x_batch_val, training=False)
            val_loss_value = loss_fn(y_batch_val, val_preds)
            val_loss_value += alpha * transmission_loss_fn(reference_encoded_x_val, encoded_X_val)
            total_val_loss += val_loss_value
        val_loss_history.append(total_val_loss.numpy() / float(total_val_steps))

        if val_loss_history[-1] < best_val_loss:
            auto_encoder.encoder.save_weights(os.path.join(output_folder, 'pre_trained_encoder_weights'),
                                              save_format='tf')
            if val_loss_history[-1] + diff_threshold < best_val_loss:
                tolerance_count = 0
            else:
                tolerance_count += 1
            best_val_loss = val_loss_history[-1]
        else:
            tolerance_count += 1

        if epoch < min_epochs:
            tolerance_count = 0
        else:
            if tolerance_count > tolerance:
                auto_encoder.encoder.load_weights(os.path.join(output_folder, 'pre_trained_encoder_weights'))
                break

    return auto_encoder.encoder, pd.DataFrame({
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
    })


def fine_tune_mut_encoder(encoder, reference_encoder, raw_X, raw_reference_X,
                              target_df,
                              transmission_loss_fn,
                              alpha=1.,
                              validation_X=None,
                              validation_target_df=None,
                              mlp_architecture=None,
                              mlp_output_act_fn=keras.activations.sigmoid,
                              mlp_output_dim=1,
                              optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                              loss_fn=penalized_mean_squared_error,
                              validation_monitoring_metric='pearson',
                              max_epoch=100,
                              min_epoch=10,
                              gradual_unfreezing_flag=True,
                              unfrozen_epoch=5
                              ):
    if mlp_architecture is None:
        mlp_architecture = [64, 32]

    output_folder = os.path.join('saved_weights', 'mut', repr(encoder) + '_encoder_weights')
    reference_folder = os.path.join('saved_weights', 'gex', repr(encoder) + '_encoder_weights')
    safe_make_dir(output_folder)

    mut_supervisor_dict = dict()
    best_overall_metric = 0.

    training_history = {
        'train_pearson': defaultdict(list),
        'train_spearman': defaultdict(list),
        'train_mse': defaultdict(list),
        'train_mae': defaultdict(list),
        'train_total': defaultdict(list)
    }

    validation_history = {
        'val_pearson': defaultdict(list),
        'val_spearman': defaultdict(list),
        'val_mse': defaultdict(list),
        'val_mae': defaultdict(list),
        'val_total': defaultdict(list)
    }
    free_layers = len(encoder.layers)

    if gradual_unfreezing_flag:
        encoder.trainable = False

    for drug in target_df.columns:
        mut_supervisor_dict[drug] = module.MLPBlock(architecture=mlp_architecture, output_act_fn=mlp_output_act_fn,
                                                    output_dim=mlp_output_dim)
        mut_supervisor_dict[drug].load_weights(os.path.join(reference_folder, drug + '_regressor_weights'))

    for epoch in range(max_epoch):
        total_train_pearson = 0.
        total_train_spearman = 0.
        total_train_mse = 0.
        total_train_mae = 0.

        total_val_pearson = 0.
        total_val_spearman = 0.
        total_val_mse = 0.
        total_val_mae = 0.

        with tf.GradientTape() as tape:
            total_loss = sum(encoder.losses)
            to_train_variables = encoder.trainable_variables
            if gradual_unfreezing_flag:
                if epoch > min_epoch:
                    free_layers -= 1
                    if (epoch - min_epoch) % unfrozen_epoch == 0:
                        free_layers -= 1
                    for i in range(len(encoder.layers) - 1, free_layers - 1, -1):
                        to_train_variables.extend(encoder.layers[i].trainable_variables)
                    if free_layers <= 0:
                        gradual_unfreezing_flag = False
                        encoder.trainable = True

            for drug in target_df.columns[:10]:
                # model = keras.Sequential()
                # model.add(encoder)
                # model.add(gex_supervisor_dict[drug])

                y = target_df.loc[~target_df[drug].isna(), drug]
                y = y.astype('float32')
                X = raw_X.loc[y.index]
                X = X.astype('float32')
                reference_X = raw_reference_X.loc[y.index]
                reference_X = reference_X.astype('float32')

                if validation_X is None:
                    kfold = KFold(n_splits=5, shuffle=True, random_state=2020)
                    cv_splits = list(kfold.split(X))
                    train_index, test_index = cv_splits[0]

                    train_X, train_Y, train_reference_X = X.iloc[train_index], y.iloc[train_index], reference_X.iloc[
                        train_index]
                    # assert all(train_X.index == train_Y.index)
                    val_X, val_Y, val_reference_X = X.iloc[test_index], y.iloc[test_index], reference_X.iloc[test_index]
                    # assert all(val_X.index == val_Y.index)
                    train_X, train_Y, train_reference_X = train_X.values, train_Y.values, train_reference_X.values
                    val_X, val_Y, val_reference_X = val_X.values, val_Y.values, val_reference_X.values

                else:
                    pass
                if repr(encoder).startswith('stochastic'):
                    encoded_X = encoder(train_X, training=True)[0]
                    reference_encoded_x = reference_encoder(train_reference_X, training=False)[0]
                else:
                    encoded_X = encoder(train_X, training=True)
                    reference_encoded_x = reference_encoder(train_reference_X, training=False)

                preds = tf.squeeze(mut_supervisor_dict[drug](encoded_X, training=True))
                loss_value = loss_fn(y_pred=preds, y_true=train_Y)
                loss_value += alpha * transmission_loss_fn(reference_encoded_x, encoded_X)

                print('Training loss (for %s) at epoch %s: %s' % (drug, epoch + 1, float(loss_value)))

                train_pearson = pearson_correlation(y_pred=preds, y_true=train_Y).numpy()
                train_spearman = spearman_correlation(y_pred=preds, y_true=train_Y).numpy()
                train_mse = keras.losses.mean_squared_error(y_pred=preds, y_true=train_Y).numpy()
                train_mae = keras.losses.mean_absolute_error(y_pred=preds, y_true=train_Y).numpy()

                total_train_pearson += train_pearson / float(target_df.shape[-1])
                total_train_spearman += train_spearman / float(target_df.shape[-1])
                total_train_mse += train_mse / float(target_df.shape[-1])
                total_train_mae += train_mae / float(target_df.shape[-1])

                training_history['train_pearson'][drug].append(train_pearson)
                training_history['train_spearman'][drug].append(train_spearman)
                training_history['train_mse'][drug].append(train_mse)
                training_history['train_mae'][drug].append(train_mae)

                if repr(encoder).startswith('stochastic'):
                    encoded_val_X = encoder(val_X, training=False)[0]
                    # reference_encoded_val_x = reference_encoder(val_reference_X, training = False)[0]
                else:
                    encoded_val_X = encoder(val_X, training=False)
                    # reference_encoded_val_x = reference_encoder(val_reference_X, training = False)

                val_preds = tf.squeeze(mut_supervisor_dict[drug](encoded_val_X, training=False))

                val_pearson = pearson_correlation(y_pred=val_preds, y_true=val_Y).numpy()
                val_spearman = spearman_correlation(y_pred=val_preds, y_true=val_Y).numpy()
                val_mse = keras.losses.mean_squared_error(y_pred=val_preds, y_true=val_Y).numpy()
                val_mae = keras.losses.mean_absolute_error(y_pred=val_preds, y_true=val_Y).numpy()

                total_val_pearson += val_pearson / float(target_df.shape[-1])
                total_val_spearman += val_spearman / float(target_df.shape[-1])
                total_val_mse += val_mse / float(target_df.shape[-1])
                total_val_mae += val_mae / float(target_df.shape[-1])

                validation_history['val_pearson'][drug].append(val_pearson)
                validation_history['val_spearman'][drug].append(val_spearman)
                validation_history['val_mse'][drug].append(val_mse)
                validation_history['val_mae'][drug].append(val_mae)

                total_loss += loss_value / float(target_df.shape[-1])
                total_loss += sum(mut_supervisor_dict[drug].losses)
                to_train_variables.extend(mut_supervisor_dict[drug].trainable_variables)

            print('Training loss (total) at epoch %s: %s' % (epoch + 1, float(total_loss)))
            training_history['train_total']['pearson'].append(total_train_pearson)
            training_history['train_total']['spearman'].append(total_train_spearman)
            training_history['train_total']['mse'].append(total_train_mse)
            training_history['train_total']['mae'].append(total_train_mae)

            validation_history['val_total']['pearson'].append(total_val_pearson)
            validation_history['val_total']['spearman'].append(total_val_spearman)
            validation_history['val_total']['mse'].append(total_val_mse)
            validation_history['val_total']['mae'].append(total_val_mae)

            if validation_history['val_total'][validation_monitoring_metric][-1] > best_overall_metric:
                best_overall_metric = validation_history['val_total'][validation_monitoring_metric][-1]
                encoder.save_weights(os.path.join(output_folder, 'fine_tuned_encoder_weights'), save_format='tf')
                for drug in target_df.columns:
                    mut_supervisor_dict[drug].save_weights(os.path.join(output_folder, drug + '_regressor_weights'),
                                                           save_format='tf')

            print(len(to_train_variables))
            grads = tape.gradient(total_loss, to_train_variables)
            optimizer.apply_gradients(zip(grads, to_train_variables))

    return training_history, validation_history


def pre_train_mut_AE_with_GAN(auto_encoder, reference_encoder, train_dataset, val_dataset,
                              alpha=1.,
                              batch_size=64,
                              optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                              loss_fn=keras.losses.MeanSquaredError(),
                              min_epochs=10,
                              max_epochs=100,
                              n_critic=5,
                              tolerance=10,
                              diff_threshold=1e-3):
    # track validation critic loss

    output_folder = os.path.join('saved_weights', 'mut', repr(auto_encoder.encoder) + '_encoder_weights')
    safe_make_dir(output_folder)

    critic = module.Critic(architecture=[128, 64, 32], output_dim=1)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)

    train_loss_history = []
    val_loss_history = []
    train_gen_loss_history = []
    val_gen_loss_history = []

    best_val_loss = float('inf')
    tolerance_count = 0

    for epoch in range(max_epochs * n_critic):
        total_train_loss = 0.
        total_train_gen_loss = 0.
        total_train_steps = 0

        total_val_loss = 0.
        total_val_gen_loss = 0.
        total_val_steps = 0
        print('epoch: ', epoch)
        for step, (x_batch_train, y_batch_train, reference_x_batch_train) in enumerate(train_dataset):
            total_train_steps += 1
            with tf.GradientTape() as tape:
                if repr(auto_encoder.encoder).startswith('stochastic'):
                    encoded_X = auto_encoder.encoder(x_batch_train, training=True)[0]
                    reference_encoded_x = reference_encoder(reference_x_batch_train, training=False)[0]
                else:
                    encoded_X = auto_encoder.encoder(x_batch_train, training=True)
                    reference_encoded_x = reference_encoder(reference_x_batch_train, training=False)

                critic_real = critic(reference_encoded_x)
                critic_fake = critic(encoded_X)
                critic_loss = tf.reduce_mean(critic_fake - critic_real, axis=0)
                total_train_loss += critic_loss

                eps = tf.random.uniform([encoded_X.shape[0], 1], 0., 1.)
                x_hat = eps * reference_encoded_x + (1 - eps) * encoded_X
                with tf.GradientTape() as sub_tape:
                    sub_tape.watch(x_hat)
                    critic_hat = critic(x_hat)
                    gp_vec = tape.gradient(critic_hat, x_hat)

                grad_penalty = tf.reduce_mean((tf.norm(gp_vec, axis=1) - 1.0) ** 2, axis=0)
                loss_value = critic_loss + 10. * grad_penalty
                loss_value += sum(critic.losses)
                grads = tape.gradient(loss_value, critic.trainable_variables)
                optimizer.apply_gradients(zip(grads, critic.trainable_variables))

            if (epoch + 1) % n_critic == 0:
                preds = auto_encoder(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, preds)
                loss_value -= alpha * critic_fake
                loss_value += sum(auto_encoder.losses)
                total_train_gen_loss -= critic_fake

                grads = tape.gradient(loss_value, auto_encoder.trainable_variables)
                optimizer.apply_gradients(zip(grads, auto_encoder.trainable_variables))
            if (step + 1) % 100 == 0:
                print('Training loss (for one batch) at step %s: %s' % (step + 1, float(loss_value)))
                print('Seen so far: %s samples' % ((step + 1) * batch_size))
        train_loss_history.append(total_train_loss.numpy() / float(total_train_steps))
        if (epoch + 1) % n_critic == 0:
            train_gen_loss_history.append(total_train_gen_loss.numpy() / float(total_train_steps))

        for step, (x_batch_val, y_batch_val, reference_x_batch_val) in enumerate(val_dataset):
            if repr(auto_encoder.encoder).startswith('stochastic'):
                encoded_X_val = auto_encoder.encoder(x_batch_val, training=False)[0]
                reference_encoded_x_val = reference_encoder(reference_x_batch_val, training=False)[0]
            else:
                encoded_X_val = auto_encoder.encoder(x_batch_val, training=False)
                reference_encoded_x_val = reference_encoder(reference_x_batch_val, training=False)
            critic_val_fake = critic(encoded_X_val)
            critic_val_real = critic(reference_encoded_x_val)
            critic_val_loss = tf.reduce_mean(critic_val_fake - critic_val_real, axis=0)

            # val_preds = auto_encoder(x_batch_val, training=False)
            # val_loss_value = loss_fn(y_batch_val, val_preds)
            # val_loss_value -= alpha * critic_val_fake
            total_val_loss += critic_val_loss
            if (epoch + 1) % n_critic == 0:
                total_val_gen_loss -= critic_fake
        val_loss_history.append(total_val_loss.numpy() / float(total_val_steps))
        if (epoch + 1) % n_critic == 0:
            val_gen_loss_history.append(total_val_gen_loss.numpy() / float(total_val_steps))

        if val_loss_history[-1] < best_val_loss:
            auto_encoder.encoder.save_weights(os.path.join(output_folder, 'pre_trained_encoder_weights'),
                                              save_format='tf')
            if val_loss_history[-1] + diff_threshold < best_val_loss:
                tolerance_count = 0
            else:
                tolerance_count += 1
            best_val_loss = val_loss_history[-1]
        else:
            tolerance_count += 1

        if epoch < min_epochs:
            tolerance_count = 0
        else:
            if tolerance_count > tolerance:
                auto_encoder.encoder.load_weights(os.path.join(output_folder, 'pre_trained_encoder_weights'))
                break

    return auto_encoder.encoder, pd.DataFrame({
        'train_critic_loss': train_loss_history,
        'val_critic_loss': val_loss_history,
        'train_gen_loss': train_gen_loss_history,
        'val_gen_loss': val_gen_loss_history
    })


def fine_tune_mut_encoder_with_GAN(encoder, raw_X,
                                       target_df,
                                       validation_X=None,
                                       validation_target_df=None,
                                       mlp_architecture=None,
                                       mlp_output_act_fn=keras.activations.sigmoid,
                                       mlp_output_dim=1,
                                       optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                                       loss_fn=penalized_mean_squared_error,
                                       validation_monitoring_metric='pearson',
                                       max_epoch=100,
                                       min_epoch=10,
                                       gradual_unfreezing_flag=True,
                                       unfrozen_epoch=5
                                       ):
    if mlp_architecture is None:
        mlp_architecture = [64, 32]

    output_folder = os.path.join('saved_weights', 'mut', repr(encoder) + '_encoder_weights')
    reference_folder = os.path.join('saved_weights', 'gex', repr(encoder) + '_encoder_weights')
    safe_make_dir(output_folder)

    mut_supervisor_dict = dict()
    best_overall_metric = 0.

    training_history = {
        'train_pearson': defaultdict(list),
        'train_spearman': defaultdict(list),
        'train_mse': defaultdict(list),
        'train_mae': defaultdict(list),
        'train_total': defaultdict(list)
    }

    validation_history = {
        'val_pearson': defaultdict(list),
        'val_spearman': defaultdict(list),
        'val_mse': defaultdict(list),
        'val_mae': defaultdict(list),
        'val_total': defaultdict(list)
    }
    free_layers = len(encoder.layers)

    if gradual_unfreezing_flag:
        encoder.trainable = False

    for drug in target_df.columns:
        mut_supervisor_dict[drug] = module.MLPBlock(architecture=mlp_architecture, output_act_fn=mlp_output_act_fn,
                                                    output_dim=mlp_output_dim)
        mut_supervisor_dict[drug].load_weights(os.path.join(reference_folder, drug + '_regressor_weights'))

    for epoch in range(max_epoch):

        total_train_pearson = 0.
        total_train_spearman = 0.
        total_train_mse = 0.
        total_train_mae = 0.

        total_val_pearson = 0.
        total_val_spearman = 0.
        total_val_mse = 0.
        total_val_mae = 0.

        with tf.GradientTape() as tape:
            total_loss = sum(encoder.losses)
            to_train_variables = encoder.trainable_variables
            if gradual_unfreezing_flag:
                if epoch > min_epoch:
                    free_layers -= 1
                    if (epoch - min_epoch) % unfrozen_epoch == 0:
                        free_layers -= 1
                    for i in range(len(encoder.layers) - 1, free_layers - 1, -1):
                        to_train_variables.extend(encoder.layers[i].trainable_variables)
                    if free_layers <= 0:
                        gradual_unfreezing_flag = False
                        encoder.trainable = True

            for drug in target_df.columns[:10]:
                # model = keras.Sequential()
                # model.add(encoder)
                # model.add(mut_supervisor_dict[drug])

                y = target_df.loc[~target_df[drug].isna(), drug]
                y = y.astype('float32')
                X = raw_X.loc[y.index]
                X = X.astype('float32')

                if validation_X is None:
                    kfold = KFold(n_splits=5, shuffle=True, random_state=2020)
                    cv_splits = list(kfold.split(X))
                    train_index, test_index = cv_splits[0]

                    train_X, train_Y = X.iloc[train_index], y.iloc[train_index]
                    # assert all(train_X.index == train_Y.index)
                    val_X, val_Y = X.iloc[test_index], y.iloc[test_index]
                    # assert all(val_X.index == val_Y.index)
                    train_X, train_Y = train_X.values, train_Y.values
                    val_X, val_Y = val_X.values, val_Y.values

                else:
                    pass
                if repr(encoder).startswith('stochastic'):
                    encoded_X = encoder(train_X, training=True)[0]
                else:
                    encoded_X = encoder(train_X, training=True)

                preds = tf.squeeze(mut_supervisor_dict[drug](encoded_X, training=True))
                loss_value = loss_fn(y_pred=preds, y_true=train_Y)

                print('Training loss (for %s) at epoch %s: %s' % (drug, epoch + 1, float(loss_value)))
                train_pearson = pearson_correlation(y_pred=preds, y_true=train_Y).numpy()
                train_spearman = spearman_correlation(y_pred=preds, y_true=train_Y).numpy()
                train_mse = keras.losses.mean_squared_error(y_pred=preds, y_true=train_Y).numpy()
                train_mae = keras.losses.mean_absolute_error(y_pred=preds, y_true=train_Y).numpy()

                total_train_pearson += train_pearson / float(target_df.shape[-1])
                total_train_spearman += train_spearman / float(target_df.shape[-1])
                total_train_mse += train_mse / float(target_df.shape[-1])
                total_train_mae += train_mae / float(target_df.shape[-1])

                training_history['train_pearson'][drug].append(train_pearson)
                training_history['train_spearman'][drug].append(train_spearman)
                training_history['train_mse'][drug].append(train_mse)
                training_history['train_mae'][drug].append(train_mae)

                if repr(encoder).startswith('stochastic'):
                    encoded_val_X = encoder(val_X, training=False)[0]
                else:
                    encoded_val_X = encoder(val_X, training=False)

                val_preds = tf.squeeze(mut_supervisor_dict[drug](encoded_val_X, training=False))

                val_pearson = pearson_correlation(y_pred=val_preds, y_true=val_Y).numpy()
                val_spearman = spearman_correlation(y_pred=val_preds, y_true=val_Y).numpy()
                val_mse = keras.losses.mean_squared_error(y_pred=val_preds, y_true=val_Y).numpy()
                val_mae = keras.losses.mean_absolute_error(y_pred=val_preds, y_true=val_Y).numpy()

                total_val_pearson += val_pearson / float(target_df.shape[-1])
                total_val_spearman += val_spearman / float(target_df.shape[-1])
                total_val_mse += val_mse / float(target_df.shape[-1])
                total_val_mae += val_mae / float(target_df.shape[-1])

                validation_history['val_pearson'][drug].append(val_pearson)
                validation_history['val_spearman'][drug].append(val_spearman)
                validation_history['val_mse'][drug].append(val_mse)
                validation_history['val_mae'][drug].append(val_mae)

                total_loss += loss_value / float(target_df.shape[-1])
                total_loss += sum(mut_supervisor_dict[drug].losses)
                to_train_variables.extend(mut_supervisor_dict[drug].trainable_variables)

            print('Training loss (total) at epoch %s: %s' % (epoch + 1, float(total_loss)))
            training_history['train_total']['pearson'].append(total_train_pearson)
            training_history['train_total']['spearman'].append(total_train_spearman)
            training_history['train_total']['mse'].append(total_train_mse)
            training_history['train_total']['mae'].append(total_train_mae)

            validation_history['val_total']['pearson'].append(total_val_pearson)
            validation_history['val_total']['spearman'].append(total_val_spearman)
            validation_history['val_total']['mse'].append(total_val_mse)
            validation_history['val_total']['mae'].append(total_val_mae)

            if validation_history['val_total'][validation_monitoring_metric][-1] > best_overall_metric:
                best_overall_metric = validation_history['val_total'][validation_monitoring_metric][-1]
                encoder.save_weights(os.path.join(output_folder, 'fine_tuned_encoder_weights'), save_format='tf')
                for drug in target_df.columns:
                    mut_supervisor_dict[drug].save_weights(os.path.join(output_folder, drug + '_regressor_weights'),
                                                           save_format='tf')

            # print(len(to_train_variables))
            grads = tape.gradient(total_loss, to_train_variables)
            optimizer.apply_gradients(zip(grads, to_train_variables))

    return training_history, validation_history
