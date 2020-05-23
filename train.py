import tensorflow as tf
import module
import model_config
from collections import defaultdict
from sklearn.model_selection import KFold
from tensorflow import keras
from loss import *
from utils import *


# @tf.function
def pre_train_gex_AE(auto_encoder, train_dataset, val_dataset,
                     batch_size=model_config.batch_size,
                     optimizer=keras.optimizers.Adam(learning_rate=model_config.pre_training_lr),
                     loss_fn=keras.losses.MeanSquaredError(),
                     min_epoch=model_config.min_epoch,
                     max_epoch=model_config.max_epoch,
                     tolerance=20,
                     diff_threshold=1e-2,
                     gradient_threshold=model_config.gradient_threshold):
    output_folder = os.path.join('saved_weights', 'gex', repr(auto_encoder.encoder) + '_encoder_weights')
    safe_make_dir(output_folder)

    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    train_mse_metric = keras.metrics.MeanSquaredError()
    train_mae_metric = keras.metrics.MeanAbsoluteError()

    if val_dataset is not None:
        val_dataset = val_dataset.batch(batch_size)
        val_mse_metric = keras.metrics.MeanSquaredError()
        val_mae_metric = keras.metrics.MeanAbsoluteError()
        val_mse_list = []
        val_mae_list = []

    train_mse_list = []
    train_mae_list = []
    best_loss = float('inf')
    best_epoch = 0
    tolerance_count = 0
    #
    # @tf.function
    # def train_step(x_batch_train, y_batch_train):
    #     nonlocal grad_norm
    #     with tf.GradientTape() as tape:
    #         preds = auto_encoder(x_batch_train, training=True)
    #         loss_value = loss_fn(y_batch_train, preds)
    #         loss_value += sum(auto_encoder.losses)
    #     grads = tape.gradient(loss_value, auto_encoder.trainable_variables)
    #     optimizer.apply_gradients(zip(grads, auto_encoder.trainable_variables))
    #     grad_norm += tf.linalg.global_norm(grads)

    for epoch in range(max_epoch):
        print('epoch: ', epoch)
        grad_norm = 0.
        counts = 0.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                preds = auto_encoder(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, preds)
                loss_value += sum(auto_encoder.losses)
                grads = tape.gradient(loss_value, auto_encoder.trainable_variables)
                grad_norm += tf.linalg.global_norm(grads)
            optimizer.apply_gradients(zip(grads, auto_encoder.trainable_variables))
            #train_step(x_batch_train=x_batch_train, y_batch_train=y_batch_train)
            counts += 1
            train_mse_metric(y_batch_train, preds)
            train_mae_metric(y_batch_train, preds)

            if (step + 1) % 10 == 0:
                print('Training loss (for one batch) at step %s: %s' % (step + 1, float(loss_value)))
                print('Seen so far: %s samples' % ((step + 1) * batch_size))

        train_mse = train_mse_metric.result()
        train_mae = train_mae_metric.result()
        train_mse_list.append(train_mse)
        train_mae_list.append(train_mae)
        train_mse_metric.reset_states()
        train_mae_metric.reset_states()

        if val_dataset is not None:
            for x_batch_val, y_batch_val in val_dataset:
                val_preds = auto_encoder(x_batch_val, training=False)
                val_mse_metric(y_batch_val, val_preds)
                val_mae_metric(y_batch_val, val_preds)
            val_mse = val_mse_metric.result()
            val_mae = val_mae_metric.result()
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
                best_epoch = epoch
            else:
                tolerance_count += 1

            if epoch < min_epoch:
                tolerance_count = 0
            else:
                if tolerance_count > tolerance:
                    break

        if gradient_threshold is not None:
            if grad_norm / counts < gradient_threshold:
                break
    if val_dataset is None:
        auto_encoder.encoder.save_weights(os.path.join(output_folder, 'pre_trained_encoder_weights'),
                                          save_format='tf')
        return best_epoch, auto_encoder.encoder, pd.DataFrame({
            'train_mse': train_mse_list,
            'train_mae': train_mae_list
        })
    else:
        auto_encoder.encoder.load_weights(os.path.join(output_folder, 'pre_trained_encoder_weights'))
        return best_epoch, auto_encoder.encoder, pd.DataFrame({
            'train_mse': train_mse_list,
            'train_mae': train_mae_list,
            'val_mse': val_mse_list,
            'val_mae': val_mae_list
        })


def fine_tune_gex_encoder(encoder,
                          train_dataset,
                          val_dataset,
                          batch_size=model_config.batch_size,
                          regressor_mlp_architecture=model_config.regressor_architecture,
                          regressor_shared_layer_num=model_config.regressor_shared_layer_number,
                          regressor_act_fn=model_config.regressor_act_fn,
                          regressor_output_dim=model_config.regressor_output_dim,
                          regressor_output_act_fn=model_config.regressor_output_act_fn,
                          loss_fn=penalized_mean_squared_error,
                          validation_monitoring_metric='pearson',
                          max_epoch=model_config.max_epoch,
                          min_epoch=model_config.min_epoch,
                          gradual_unfreezing_flag=model_config.gradual_unfreezing_flag,
                          unfrozen_epoch=model_config.unfrozen_epoch,
                          gradient_threshold=model_config.gradient_threshold,
                          exp_type='cv'
                          ):
    output_folder = os.path.join('saved_weights', 'gex', exp_type, repr(encoder) + '_encoder_weights')
    safe_make_dir(output_folder)

    best_overall_metric = float('-inf')
    best_epoch = 0

    train_dataset = train_dataset.shuffle(buffer_size=512).batch(batch_size)

    if val_dataset is not None:
        val_dataset = val_dataset.batch(batch_size)

    training_history = defaultdict(list)
    validation_history = defaultdict(list)

    num_encoder_layers = len(encoder.layers)
    if repr(encoder).startswith('stochastic'):
        num_encoder_layers -= 1
    free_layers = num_encoder_layers

    encoder.trainable = False
    # shared_regressor_module = module.MLPBlock(architecture=model_config.shared_regressor_architecture,
    #                                           output_act_fn=model_config.shared_regressor_act_fn,
    #                                           output_dim=model_config.shared_regressor_output_dim,
    #                                           kernel_regularizer_l=model_config.kernel_regularizer_l)
    # for drug in target_df.columns:
    #     gex_supervisor_dict[drug] = module.MLPBlock(architecture=mlp_architecture,
    #                                                 output_act_fn=mlp_output_act_fn,
    #                                                 output_dim=mlp_output_dim,
    #                                                 kernel_regularizer_l=model_config.kernel_regularizer_l)
    # drug_regressor_module = module.MLPBlockWithMask(
    #     architecture=model_config.regressor_architecture,
    #     output_act_fn=model_config.regressor_act_fn,
    #     output_dim=model_config.drug_num,
    #     kernel_regularizer_l=model_config.kernel_regularizer_l
    # )
    regressor = module.MLPBlockWithMask(architecture=regressor_mlp_architecture,
                                        shared_layer_num=regressor_shared_layer_num,
                                        act_fn=regressor_act_fn,
                                        output_act_fn=regressor_output_act_fn,
                                        output_dim=regressor_output_dim,
                                        kernel_regularizer_l=model_config.kernel_regularizer_l)
    lr = model_config.fine_tuning_lr
    regressor_tuning_flag = False

    for epoch in range(max_epoch):
        grad_norm = 0.

        # total_train_pearson = 0.
        # total_train_spearman = 0.
        # total_train_mse = 0.
        # total_train_mae = 0.
        #
        # total_val_pearson = 0.
        # total_val_spearman = 0.
        # total_val_mse = 0.
        # total_val_mae = 0.
        to_train_variables = encoder.trainable_variables
        if gradual_unfreezing_flag:
            if epoch >= min_epoch:
                if (epoch - min_epoch) % unfrozen_epoch == 0:
                    free_layers -= 1
                    lr *= model_config.decay
                for i in range(num_encoder_layers-1, free_layers - 1, -1):
                    to_train_variables.extend(encoder.layers[i].trainable_variables)
                if free_layers <= 0:
                    gradual_unfreezing_flag = False
                    encoder.trainable = True

        # regressor = keras.Sequential()
        # regressor.add(shared_regressor_module)
        # regressor.add(drug_regressor_module)
        train_epoch_loss = 0.
        train_epoch_pearson = 0.
        train_epoch_spearman = 0.
        train_epoch_mse = 0.
        train_epoch_mae = 0.
        counts = 0.
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                tape.watch(to_train_variables)
                if repr(encoder).startswith('stochastic'):
                    encoded_X = encoder(x_batch_train, training=True)[0]
                else:
                    encoded_X = encoder(x_batch_train, training=True)

                preds = regressor(encoded_X, training=True)
                loss_value = loss_fn(y_pred=preds, y_true=y_batch_train)
                train_epoch_loss += loss_value
                train_epoch_pearson += pearson_correlation(y_pred=preds, y_true=y_batch_train)
                train_epoch_spearman += spearman_correlation(y_pred=preds, y_true=y_batch_train)
                train_epoch_mse += mse(y_pred=preds, y_true=y_batch_train)
                train_epoch_mae += mae(y_pred=preds, y_true=y_batch_train)

                loss_value += sum(encoder.losses)
                loss_value += sum(regressor.losses)
                if not regressor_tuning_flag:
                    to_train_variables.extend(regressor.trainable_variables)
                    regressor_tuning_flag = True
                grads = tape.gradient(loss_value, to_train_variables)
                grad_norm += tf.linalg.global_norm(grads)
                optimizer.apply_gradients(zip(grads, to_train_variables))
                counts += 1.


        print('Training loss  at epoch %s: %s' % (epoch + 1, float(train_epoch_loss / counts)))
        regressor_tuning_flag = False

        training_history['loss'].append(train_epoch_loss / counts)
        training_history['pearson'].append(train_epoch_pearson / counts)
        training_history['spearman'].append(train_epoch_spearman / counts)
        training_history['mse'].append(train_epoch_mse / counts)
        training_history['mae'].append(train_epoch_mae / counts)
        train_counts = counts

        if val_dataset is not None:
            val_epoch_loss = 0.
            val_epoch_pearson = 0.
            val_epoch_spearman = 0.
            val_epoch_mse = 0.
            val_epoch_mae = 0.
            counts = 0.

            for x_batch_val, y_batch_val in val_dataset:
                if repr(encoder).startswith('stochastic'):
                    encoded_val_X = encoder(x_batch_val, training=False)[0]
                else:
                    encoded_val_X = encoder(x_batch_val, training=False)
                val_preds = regressor(encoded_val_X, training=False)

                val_loss_value = loss_fn(y_pred=val_preds, y_true=y_batch_val)
                val_epoch_loss += val_loss_value

                val_epoch_pearson += pearson_correlation(y_pred=val_preds, y_true=y_batch_val)
                val_epoch_spearman += spearman_correlation(y_pred=val_preds, y_true=y_batch_val)
                val_epoch_mse += mse(y_pred=val_preds, y_true=y_batch_val)
                val_epoch_mae += mae(y_pred=val_preds, y_true=y_batch_val)
                counts += 1.

                # total_val_pearson += val_pearson / float(target_df.shape[-1])
                # total_val_spearman += val_spearman / float(target_df.shape[-1])
                # total_val_mse += val_mse / float(target_df.shape[-1])
                # total_val_mae += val_mae / float(target_df.shape[-1])
            validation_history['loss'].append(val_epoch_loss / counts)
            validation_history['pearson'].append(val_epoch_pearson / counts)
            validation_history['spearman'].append(val_epoch_spearman / counts)
            validation_history['mse'].append(val_epoch_mse / counts)
            validation_history['mae'].append(val_epoch_mae / counts)

            # print(validation_history['val_total'][validation_monitoring_metric][-1])
            # print(best_overall_metric)

            if validation_history[validation_monitoring_metric][-1] > best_overall_metric:
                best_overall_metric = validation_history[validation_monitoring_metric][-1]
                best_epoch = epoch
                encoder.save_weights(os.path.join(output_folder, 'fine_tuned_encoder_weights'), save_format='tf')
                regressor.save_weights(os.path.join(output_folder, 'regressor_weights'), save_format='tf')
                # print(len(to_train_variables))

        if gradient_threshold is not None:
            if grad_norm / train_counts < gradient_threshold:
                break

    if val_dataset is None:
        encoder.save_weights(os.path.join(output_folder, 'fine_tuned_encoder_weights'), save_format='tf')
        regressor.save_weights(os.path.join(output_folder, 'regressor_weights'), save_format='tf')

    return best_epoch, training_history, validation_history


def pre_train_mut_AE(auto_encoder, reference_encoder, train_dataset, val_dataset,
                     transmission_loss_fn,
                     transmitter_flag=False,
                     alpha=model_config.alpha,
                     batch_size=model_config.batch_size,
                     optimizer=keras.optimizers.Adam(learning_rate=model_config.pre_training_lr),
                     loss_fn=keras.losses.MeanSquaredError(),
                     min_epoch=model_config.min_epoch,
                     max_epoch=model_config.max_epoch,
                     tolerance=20,
                     diff_threshold=1e-2,
                     gradient_threshold=model_config.gradient_threshold,
                     exp_type='cv'):
    output_folder = os.path.join('saved_weights', 'mut', exp_type, repr(auto_encoder.encoder) + '_encoder_weights')
    safe_make_dir(output_folder)

    reference_folder = os.path.join('saved_weights', 'gex', exp_type, repr(reference_encoder) + '_encoder_weights')
    reference_encoder.load_weights(os.path.join(reference_folder, 'fine_tuned_encoder_weights'))

    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    if val_dataset is not None:
        val_dataset = val_dataset.batch(batch_size)
        val_loss_history = []
        best_val_loss = float('inf')
    best_epoch = 0
    train_loss_history = []

    tolerance_count = 0
    reference_encoder.trainable = False

    if transmitter_flag:
        transmitter = module.MLPBlock(architecture=model_config.transmitter_architecture,
                                      act_fn=model_config.transmitter_act_fn,
                                      output_act_fn=model_config.transmitter_output_act_fn,
                                      output_dim=model_config.transmitter_output_dim,
                                      kernel_regularizer_l=model_config.kernel_regularizer_l,
                                      name='transmitter')

    for epoch in range(max_epoch):
        total_train_loss = 0.
        total_train_steps = 0
        grad_norm = 0.

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
                loss_value = (1 - alpha) * loss_fn(y_batch_train, preds)
                if transmitter_flag:
                    loss_value += alpha * transmission_loss_fn(reference_encoded_x,
                                                               transmitter(encoded_X, training=True))
                else:
                    loss_value += alpha * transmission_loss_fn(reference_encoded_x, encoded_X)

                total_train_loss += loss_value
                loss_value += sum(auto_encoder.losses)
                if transmitter_flag:
                    loss_value += sum(transmitter.losses)
                    grads = tape.gradient(loss_value,
                                          auto_encoder.trainable_variables + transmitter.trainable_variables)
                    optimizer.apply_gradients(
                        zip(grads, auto_encoder.trainable_variables + transmitter.trainable_variables))
                else:
                    grads = tape.gradient(loss_value, auto_encoder.trainable_variables)
                    optimizer.apply_gradients(zip(grads, auto_encoder.trainable_variables))
                grad_norm += tf.linalg.global_norm(grads)

            if (step + 1) % 10 == 0:
                print('Training loss (for one batch) at step %s: %s' % (step + 1, float(loss_value)))
                print('Seen so far: %s samples' % ((step + 1) * batch_size))
        train_loss_history.append(total_train_loss / float(total_train_steps))

        if val_dataset is not None:
            total_val_loss = 0.
            total_val_steps = 0.
            for step, (x_batch_val, y_batch_val, reference_x_batch_val) in enumerate(val_dataset):
                total_val_steps += 1
                if repr(auto_encoder.encoder).startswith('stochastic'):
                    encoded_X_val = auto_encoder.encoder(x_batch_val, training=False)[0]
                    reference_encoded_x_val = reference_encoder(reference_x_batch_val, training=False)[0]
                else:
                    encoded_X_val = auto_encoder.encoder(x_batch_val, training=False)
                    reference_encoded_x_val = reference_encoder(reference_x_batch_val, training=False)
                val_preds = auto_encoder(x_batch_val, training=False)
                val_loss_value = (1 - alpha) * loss_fn(y_batch_val, val_preds)
                if transmitter_flag:
                    val_loss_value += alpha * transmission_loss_fn(reference_encoded_x_val,
                                                                   transmitter(encoded_X_val, training=False))
                else:
                    val_loss_value += alpha * transmission_loss_fn(reference_encoded_x_val, encoded_X_val)
                total_val_loss += val_loss_value
            val_loss_history.append(total_val_loss / float(total_val_steps))

            if val_loss_history[-1] < best_val_loss:
                auto_encoder.encoder.save_weights(os.path.join(output_folder, 'pre_trained_encoder_weights'),
                                                  save_format='tf')
                if transmitter_flag:
                    transmitter.save_weights(os.path.join(output_folder, 'pre_trained_transmitter_weights'),
                                             save_format='tf')
                if val_loss_history[-1] + diff_threshold < best_val_loss:
                    tolerance_count = 0
                else:
                    tolerance_count += 1
                best_val_loss = val_loss_history[-1]
                best_epoch = epoch
            else:
                tolerance_count += 1

            if epoch < min_epoch:
                tolerance_count = 0
            else:
                if tolerance_count > tolerance and gradient_threshold is None:
                    break

        if gradient_threshold is not None:
            if grad_norm / float(total_train_steps) < gradient_threshold:
                break

    if val_dataset is None:
        auto_encoder.encoder.save_weights(os.path.join(output_folder, 'pre_trained_encoder_weights'),
                                          save_format='tf')
        if transmitter_flag:
            transmitter.save_weights(os.path.join(output_folder, 'pre_trained_transmitter_weights'),
                                     save_format='tf')
        return best_epoch, auto_encoder.encoder, pd.DataFrame({
            'train_loss': train_loss_history
        })

    else:
        best_epoch, auto_encoder.encoder.load_weights(os.path.join(output_folder, 'pre_trained_encoder_weights'))
        return best_epoch, auto_encoder.encoder, pd.DataFrame({
            'train_loss': train_loss_history,
            'val_loss': val_loss_history
        })


def pre_train_mut_AE_with_GAN(auto_encoder, reference_encoder, train_dataset, val_dataset,
                              alpha=model_config.alpha,
                              transmitter_flag=False,
                              batch_size=model_config.batch_size,
                              optimizer=keras.optimizers.Adam(learning_rate=model_config.pre_training_lr),
                              loss_fn=keras.losses.MeanSquaredError(),
                              min_epoch=model_config.min_epoch,
                              max_epoch=model_config.max_epoch,
                              n_critic=5,
                              tolerance=10,
                              diff_threshold=1e-2,
                              gradient_threshold=model_config.gradient_threshold,
                              exp_type='cv'):
    # track validation critic loss

    output_folder = os.path.join('saved_weights', 'mut', exp_type, repr(auto_encoder.encoder) + '_encoder_weights')
    safe_make_dir(output_folder)

    reference_folder = os.path.join('saved_weights', 'gex', exp_type, repr(reference_encoder) + '_encoder_weights')
    reference_encoder.load_weights(os.path.join(reference_folder, 'fine_tuned_encoder_weights'))

    critic = module.Critic(architecture=[128, 64, 32], output_dim=1)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    if val_dataset is not None:
        val_dataset = val_dataset.batch(batch_size)
        val_loss_history = []
        val_gen_loss_history = []
        best_val_loss = float('inf')
    best_epoch = 0
    if transmitter_flag:
        transmitter = module.MLPBlock(architecture=model_config.transmitter_architecture,
                                      act_fn=model_config.transmitter_act_fn,
                                      output_act_fn=model_config.transmitter_output_act_fn,
                                      output_dim=model_config.transmitter_output_dim,
                                      kernel_regularizer_l=model_config.kernel_regularizer_l,
                                      name='transmitter')

    train_loss_history = []
    train_gen_loss_history = []

    tolerance_count = 0
    reference_encoder.trainable = False

    gp_optimizer = keras.optimizers.RMSprop()

    for epoch in range(max_epoch):
        total_train_loss = 0.
        total_train_gen_loss = 0.
        total_train_steps = 0
        grad_norm = 0.

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
                if transmitter_flag:
                    critic_fake = critic(transmitter(encoded_X, training=True))
                else:
                    critic_fake = critic(encoded_X)

                critic_loss = tf.reduce_mean(critic_fake - critic_real, axis=0)
                total_train_loss += critic_loss

                eps = tf.random.uniform([encoded_X.shape[0], 1], 0., 1.)
                x_hat = eps * reference_encoded_x + (1 - eps) * encoded_X
                with tf.GradientTape() as sub_tape:
                    sub_tape.watch(x_hat)
                    critic_hat = critic(x_hat)
                    gp_vec = sub_tape.gradient(critic_hat, x_hat)

                grad_penalty = tf.reduce_mean((tf.norm(gp_vec, axis=1) - 1.0) ** 2, axis=0)
                loss_value = critic_loss + 10. * grad_penalty
                loss_value += sum(critic.losses)
                grads = tape.gradient(loss_value, critic.trainable_variables)
                gp_optimizer.apply_gradients(zip(grads, critic.trainable_variables))

            if (step + 1) % n_critic == 0:
                with tf.GradientTape() as tape:
                    preds = auto_encoder(x_batch_train, training=True)
                    loss_value = (1 - alpha) * loss_fn(y_batch_train, preds)
                    loss_value -= alpha * tf.reduce_mean(critic_fake, axis=0)

                    total_train_gen_loss -= tf.reduce_mean(critic_fake, axis=0)

                    loss_value += sum(auto_encoder.losses)
                    if transmitter_flag:
                        loss_value += sum(transmitter.losses)
                        grads = tape.gradient(loss_value,
                                              auto_encoder.trainable_variables + transmitter.trainable_variables)
                        optimizer.apply_gradients(
                            zip(grads, auto_encoder.trainable_variables + transmitter.trainable_variables))
                    else:
                        grads = tape.gradient(loss_value, auto_encoder.trainable_variables)
                        optimizer.apply_gradients(zip(grads, auto_encoder.trainable_variables))
                    grad_norm += tf.linalg.global_norm(grads)

            if (step + 1) % 10 == 0:
                print('Training loss (for one batch) at step %s: %s' % (step + 1, float(loss_value)))
                print('Seen so far: %s samples' % ((step + 1) * batch_size))

        train_loss_history.append(total_train_loss / float(total_train_steps))
        train_gen_loss_history.append(total_train_gen_loss / float(total_train_steps // n_critic))

        if val_dataset is not None:
            total_val_loss = 0.
            total_val_gen_loss = 0.
            total_val_steps = 0
            for step, (x_batch_val, y_batch_val, reference_x_batch_val) in enumerate(val_dataset):
                total_val_steps += 1
                if repr(auto_encoder.encoder).startswith('stochastic'):
                    encoded_X_val = auto_encoder.encoder(x_batch_val, training=False)[0]
                    reference_encoded_x_val = reference_encoder(reference_x_batch_val, training=False)[0]
                else:
                    encoded_X_val = auto_encoder.encoder(x_batch_val, training=False)
                    reference_encoded_x_val = reference_encoder(reference_x_batch_val, training=False)
                if transmitter_flag:
                    critic_val_fake = critic(transmitter(encoded_X_val, training=False))
                else:
                    critic_val_fake = critic(encoded_X_val)

                critic_val_real = critic(reference_encoded_x_val)
                critic_val_loss = tf.reduce_mean(critic_val_fake - critic_val_real, axis=0)

                # val_preds = auto_encoder(x_batch_val, training=False)
                # val_loss_value = loss_fn(y_batch_val, val_preds)
                # val_loss_value -= alpha * critic_val_fake
                total_val_loss += critic_val_loss
                total_val_gen_loss -= tf.reduce_mean(critic_val_fake, axis=0)
            val_loss_history.append(total_val_loss / float(total_val_steps))
            val_gen_loss_history.append(total_val_gen_loss / float(total_val_steps))

            if val_loss_history[-1] < best_val_loss:
                auto_encoder.encoder.save_weights(os.path.join(output_folder, 'pre_trained_encoder_weights'),
                                                  save_format='tf')
                if transmitter_flag:
                    transmitter.save_weights(os.path.join(output_folder, 'pre_trained_transmitter_weights'),
                                             save_format='tf')
                if val_loss_history[-1] + diff_threshold < best_val_loss:
                    tolerance_count = 0
                else:
                    tolerance_count += 1
                best_val_loss = val_loss_history[-1]
                best_epoch = epoch
            else:
                tolerance_count += 1

            if epoch < min_epoch:
                tolerance_count = 0
            else:
                if tolerance_count > tolerance and gradient_threshold is None:
                    break

        if gradient_threshold is not None:
            if grad_norm / (float(total_train_steps // n_critic)) < gradient_threshold:
                break

    if val_dataset is None:
        auto_encoder.encoder.save_weights(os.path.join(output_folder, 'pre_trained_encoder_weights'),
                                          save_format='tf')
        if transmitter_flag:
            transmitter.save_weights(os.path.join(output_folder, 'pre_trained_transmitter_weights'),
                                     save_format='tf')
        return best_epoch, auto_encoder.encoder, pd.DataFrame({
            'train_critic_loss': train_loss_history,
            'train_gen_loss': train_gen_loss_history
        })
    else:
        auto_encoder.encoder.load_weights(os.path.join(output_folder, 'pre_trained_encoder_weights'))
        return best_epoch, auto_encoder.encoder, pd.DataFrame({
            'train_critic_loss': train_loss_history,
            'val_critic_loss': val_loss_history,
            'train_gen_loss': train_gen_loss_history,
            'val_gen_loss': val_gen_loss_history
        })


def fine_tune_mut_encoder(encoder, train_dataset,
                          val_dataset, batch_size=model_config.batch_size,
                          regressor_mlp_architecture=model_config.regressor_architecture,
                          regressor_shared_layer_num=model_config.regressor_shared_layer_number,
                          regressor_act_fn=model_config.regressor_act_fn,
                          regressor_output_dim=model_config.regressor_output_dim,
                          regressor_output_act_fn=model_config.regressor_output_act_fn,
                          regressor_flag=True,
                          transmitter_flag=False,
                          loss_fn=penalized_mean_squared_error,
                          validation_monitoring_metric='pearson',
                          max_epoch=model_config.max_epoch,
                          min_epoch=model_config.min_epoch,
                          gradual_unfreezing_flag=model_config.gradual_unfreezing_flag,
                          unfrozen_epoch=model_config.unfrozen_epoch,
                          gradient_threshold=model_config.gradient_threshold,
                          exp_type='cv'
                          ):
    output_folder = os.path.join('saved_weights', 'mut', exp_type, repr(encoder) + '_encoder_weights')
    reference_folder = os.path.join('saved_weights', 'gex', exp_type, repr(encoder) + '_encoder_weights')

    safe_make_dir(output_folder)

    best_overall_metric = float('-inf')
    best_epoch = 0
    train_dataset = train_dataset.shuffle(buffer_size=512).batch(batch_size)

    if val_dataset is not None:
        val_dataset = val_dataset.batch(batch_size)

    training_history = defaultdict(list)
    validation_history = defaultdict(list)

    num_encoder_layers = len(encoder.layers)
    if repr(encoder).startswith('stochastic'):
        num_encoder_layers -= 1
    free_encoder_layers = num_encoder_layers

    if gradual_unfreezing_flag:
        encoder.trainable = False

    regressor = module.MLPBlockWithMask(architecture=regressor_mlp_architecture,
                                        shared_layer_num=regressor_shared_layer_num,
                                        act_fn=regressor_act_fn,
                                        output_act_fn=regressor_output_act_fn,
                                        kernel_regularizer_l=model_config.kernel_regularizer_l,
                                        output_dim=regressor_output_dim)
    if regressor_flag:
        regressor.load_weights(os.path.join(reference_folder, 'regressor_weights'))

    if transmitter_flag:
        transmitter = module.MLPBlock(architecture=model_config.transmitter_architecture,
                                      act_fn=model_config.transmitter_act_fn,
                                      output_act_fn=model_config.transmitter_output_act_fn,
                                      output_dim=model_config.transmitter_output_dim,
                                      kernel_regularizer_l=model_config.kernel_regularizer_l,
                                      name='transmitter')
        transmitter.load_weights(os.path.join(output_folder, 'pre_trained_transmitter_weights'))
        if gradual_unfreezing_flag:
            transmitter.trainable = False
        free_transmitter_layers = len(transmitter.layers)

    lr = model_config.fine_tuning_lr
    regressor_tuning_flag = False

    for epoch in range(max_epoch):
        grad_norm = 0.

        to_train_variables = encoder.trainable_variables
        if transmitter_flag:
            to_train_variables.extend(transmitter.trainable_variables)

        if gradual_unfreezing_flag:
            if epoch >= min_epoch:
                if transmitter_flag and free_transmitter_layers > 0:
                    if (epoch - min_epoch) % unfrozen_epoch == 0:
                        free_transmitter_layers -= 1
                        lr *= model_config.decay
                    for i in range(num_encoder_layers, free_transmitter_layers - 1, -1):
                        to_train_variables.extend(transmitter.layers[i].trainable_variables)
                    if free_transmitter_layers <= 0:
                        transmitter.trainable = True
                else:
                    if (epoch - min_epoch) % unfrozen_epoch == 0:
                        free_encoder_layers -= 1
                        lr *= model_config.decay
                    for i in range(num_encoder_layers-1, free_encoder_layers - 1, -1):
                        to_train_variables.extend(encoder.layers[i].trainable_variables)
                    if free_encoder_layers <= 0:
                        gradual_unfreezing_flag = False
                        encoder.trainable = True

        train_epoch_loss = 0.
        train_epoch_pearson = 0.
        train_epoch_spearman = 0.
        train_epoch_mse = 0.
        train_epoch_mae = 0.
        counts = 0.
        optimizer = keras.optimizers.Adam(learning_rate=lr)

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                tape.watch(to_train_variables)
                if repr(encoder).startswith('stochastic'):
                    encoded_X = encoder(x_batch_train, training=True)[0]
                else:
                    encoded_X = encoder(x_batch_train, training=True)

                if transmitter_flag:
                    encoded_X = transmitter(encoded_X, training=True)

                preds = regressor(encoded_X, training=True)
                loss_value = loss_fn(y_pred=preds, y_true=y_batch_train)
                train_epoch_loss += loss_value
                train_epoch_pearson += pearson_correlation(y_pred=preds, y_true=y_batch_train)
                train_epoch_spearman += spearman_correlation(y_pred=preds, y_true=y_batch_train)
                train_epoch_mse += mse(y_pred=preds, y_true=y_batch_train)
                train_epoch_mae += mae(y_pred=preds, y_true=y_batch_train)

                loss_value += sum(encoder.losses)
                loss_value += sum(regressor.losses)
                if transmitter_flag:
                    loss_value += sum(transmitter.losses)

                if not regressor_tuning_flag:
                    to_train_variables.extend(regressor.trainable_variables)
                    regressor_tuning_flag = True

                grads = tape.gradient(loss_value, to_train_variables)
                grad_norm += tf.linalg.global_norm(grads)
                optimizer.apply_gradients(zip(grads, to_train_variables))
                counts += 1

        print('Training loss  at epoch %s: %s' % (epoch + 1, float(train_epoch_loss / counts)))
        regressor_tuning_flag = False

        training_history['loss'].append(train_epoch_loss / counts)
        training_history['pearson'].append(train_epoch_pearson / counts)
        training_history['spearman'].append(train_epoch_spearman / counts)
        training_history['mse'].append(train_epoch_mse / counts)
        training_history['mae'].append(train_epoch_mae / counts)
        train_counts = counts

        if val_dataset is not None:
            val_epoch_loss = 0.
            val_epoch_pearson = 0.
            val_epoch_spearman = 0.
            val_epoch_mse = 0.
            val_epoch_mae = 0.
            counts = 0.

            for x_batch_val, y_batch_val in val_dataset:
                if repr(encoder).startswith('stochastic'):
                    encoded_val_X = encoder(x_batch_val, training=False)[0]
                else:
                    encoded_val_X = encoder(x_batch_val, training=False)

                if transmitter_flag:
                    encoded_val_X = transmitter(encoded_val_X, training=False)

                val_preds = regressor(encoded_val_X, training=False)
                val_loss_value = loss_fn(y_pred=val_preds, y_true=y_batch_val)
                val_epoch_loss += val_loss_value
                val_epoch_pearson += pearson_correlation(y_pred=val_preds, y_true=y_batch_val)
                val_epoch_spearman += spearman_correlation(y_pred=val_preds, y_true=y_batch_val)
                val_epoch_mse += mse(y_pred=val_preds, y_true=y_batch_val)
                val_epoch_mae += mae(y_pred=val_preds, y_true=y_batch_val)
                counts += 1.

            validation_history['loss'].append(val_epoch_loss / counts)
            validation_history['pearson'].append(val_epoch_pearson / counts)
            validation_history['spearman'].append(val_epoch_spearman / counts)
            validation_history['mse'].append(val_epoch_mse / counts)
            validation_history['mae'].append(val_epoch_mae / counts)

            if validation_history[validation_monitoring_metric][-1] > best_overall_metric:
                best_overall_metric = validation_history[validation_monitoring_metric][-1]
                best_epoch = epoch
                encoder.save_weights(os.path.join(output_folder, 'fine_tuned_encoder_weights'), save_format='tf')
                regressor.save_weights(os.path.join(output_folder, 'regressor_weights'), save_format='tf')
                if transmitter_flag:
                    transmitter.save_weights(os.path.join(output_folder, 'fine_tuned_transmitter_weights'),
                                             save_format='tf')

        if gradient_threshold is not None:
            if grad_norm / train_counts < gradient_threshold:
                break
    if val_dataset is None:
        encoder.save_weights(os.path.join(output_folder, 'fine_tuned_encoder_weights'), save_format='tf')
        regressor.save_weights(os.path.join(output_folder, 'regressor_weights'), save_format='tf')
        if transmitter_flag:
            transmitter.save_weights(os.path.join(output_folder, 'fine_tuned_transmitter_weights'),
                                     save_format='tf')

    return best_epoch, training_history, validation_history

# def fine_tune_mut_encoder(encoder, reference_encoder, raw_X, raw_reference_X,
#                           target_df,
#                           transmission_loss_fn,
#                           alpha=1.,
#                           validation_X=None,
#                           validation_target_df=None,
#                           mlp_architecture=None,
#                           mlp_output_act_fn=keras.activations.sigmoid,
#                           mlp_output_dim=1,
#                           optimizer=keras.optimizers.Adam(learning_rate=model_config.fine_tuning_lr),
#                           loss_fn=penalized_mean_squared_error,
#                           validation_monitoring_metric='pearson',
#                           max_epoch=100,
#                           min_epoch=10,
#                           gradual_unfreezing_flag=True,
#                           unfrozen_epoch=5
#                           ):
#     if mlp_architecture is None:
#         mlp_architecture = [64, 32]
#
#     output_folder = os.path.join('saved_weights', 'mut', repr(encoder) + '_encoder_weights')
#     reference_folder = os.path.join('saved_weights', 'gex', repr(encoder) + '_encoder_weights')
#     safe_make_dir(output_folder)
#
#     mut_supervisor_dict = dict()
#     best_overall_metric = float('-inf')
#
#     training_history = {
#         'train_pearson': defaultdict(list),
#         'train_spearman': defaultdict(list),
#         'train_mse': defaultdict(list),
#         'train_mae': defaultdict(list),
#         'train_total': defaultdict(list)
#     }
#
#     validation_history = {
#         'val_pearson': defaultdict(list),
#         'val_spearman': defaultdict(list),
#         'val_mse': defaultdict(list),
#         'val_mae': defaultdict(list),
#         'val_total': defaultdict(list)
#     }
#     free_layers = len(encoder.layers)
#     reference_encoder.trainable = False
#
#     if gradual_unfreezing_flag:
#         encoder.trainable = False
#
#     shared_regressor_module = module.MLPBlock(architecture=model_config.shared_regressor_architecture,
#                                               output_act_fn=model_config.shared_regressor_act_fn,
#                                               output_dim=model_config.shared_regressor_output_dim,
#                                               kernel_regularizer_l=model_config.kernel_regularizer_l)
#
#     shared_regressor_module.load_weights(os.path.join(reference_folder, 'shared_regressor_weights'))
#
#     for drug in target_df.columns:
#         mut_supervisor_dict[drug] = module.MLPBlock(architecture=mlp_architecture,
#                                                     output_act_fn=mlp_output_act_fn,
#                                                     output_dim=mlp_output_dim,
#                                                     kernel_regularizer_l=model_config.kernel_regularizer_l)
#         mut_supervisor_dict[drug].load_weights(os.path.join(reference_folder, drug + '_regressor_weights'))
#
#     for epoch in range(max_epoch):
#         total_train_pearson = 0.
#         total_train_spearman = 0.
#         total_train_mse = 0.
#         total_train_mae = 0.
#
#         total_val_pearson = 0.
#         total_val_spearman = 0.
#         total_val_mse = 0.
#         total_val_mae = 0.
#
#         with tf.GradientTape() as tape:
#             total_loss = sum(encoder.losses)
#             total_loss += sum(shared_regressor_module.losses)
#             to_train_variables = encoder.trainable_variables
#             if gradual_unfreezing_flag:
#                 if epoch > min_epoch:
#                     free_layers -= 1
#                     if (epoch - min_epoch) % unfrozen_epoch == 0:
#                         free_layers -= 1
#                     for i in range(len(encoder.layers) - 1, free_layers - 1, -1):
#                         to_train_variables.extend(encoder.layers[i].trainable_variables)
#                     if free_layers <= 0:
#                         gradual_unfreezing_flag = False
#                         encoder.trainable = True
#             tape.watch(to_train_variables)
#             to_train_variables.extend(shared_regressor_module.trainable_variables)
#
#             for drug in target_df.columns:
#                 # model = keras.Sequential()
#                 # model.add(encoder)
#                 # model.add(gex_supervisor_dict[drug])
#
#                 y = target_df.loc[~target_df[drug].isna(), drug]
#                 y = y.astype('float32')
#                 X = raw_X.loc[y.index]
#                 X = X.astype('float32')
#                 reference_X = raw_reference_X.loc[y.index]
#                 reference_X = reference_X.astype('float32')
#
#                 if validation_X is None:
#                     kfold = KFold(n_splits=5, shuffle=True, random_state=2020)
#                     cv_splits = list(kfold.split(X))
#                     train_index, test_index = cv_splits[0]
#
#                     train_X, train_Y, train_reference_X = X.iloc[train_index], y.iloc[train_index], reference_X.iloc[
#                         train_index]
#                     # assert all(train_X.index == train_Y.index)
#                     val_X, val_Y, val_reference_X = X.iloc[test_index], y.iloc[test_index], reference_X.iloc[test_index]
#                     # assert all(val_X.index == val_Y.index)
#                     train_X, train_Y, train_reference_X = train_X.values, train_Y.values, train_reference_X.values
#                     val_X, val_Y, val_reference_X = val_X.values, val_Y.values, val_reference_X.values
#
#                 else:
#                     pass
#                 if repr(encoder).startswith('stochastic'):
#                     encoded_X = encoder(train_X, training=True)[0]
#                     reference_encoded_x = reference_encoder(train_reference_X, training=False)[0]
#                 else:
#                     encoded_X = encoder(train_X, training=True)
#                     reference_encoded_x = reference_encoder(train_reference_X, training=False)
#
#                 regressor = keras.Sequential()
#                 regressor.add(shared_regressor_module)
#                 regressor.add(mut_supervisor_dict[drug])
#
#                 preds = tf.squeeze(regressor(encoded_X, training=True))
#                 loss_value = loss_fn(y_pred=preds, y_true=train_Y)
#                 loss_value += alpha * transmission_loss_fn(reference_encoded_x, encoded_X)
#
#                 #print('Training loss (for %s) at epoch %s: %s' % (drug, epoch + 1, float(loss_value)))
#
#                 train_pearson = pearson_correlation(y_pred=preds, y_true=train_Y)
#                 train_spearman = spearman_correlation(y_pred=preds, y_true=train_Y)
#                 train_mse = keras.losses.mean_squared_error(y_pred=preds, y_true=train_Y)
#                 train_mae = keras.losses.mean_absolute_error(y_pred=preds, y_true=train_Y)
#
#                 total_train_pearson += train_pearson / float(target_df.shape[-1])
#                 total_train_spearman += train_spearman / float(target_df.shape[-1])
#                 total_train_mse += train_mse / float(target_df.shape[-1])
#                 total_train_mae += train_mae / float(target_df.shape[-1])
#
#                 training_history['train_pearson'][drug].append(train_pearson)
#                 training_history['train_spearman'][drug].append(train_spearman)
#                 training_history['train_mse'][drug].append(train_mse)
#                 training_history['train_mae'][drug].append(train_mae)
#
#                 if repr(encoder).startswith('stochastic'):
#                     encoded_val_X = encoder(val_X, training=False)[0]
#                     # reference_encoded_val_x = reference_encoder(val_reference_X, training = False)[0]
#                 else:
#                     encoded_val_X = encoder(val_X, training=False)
#                     # reference_encoded_val_x = reference_encoder(val_reference_X, training = False)
#
#                 val_preds = tf.squeeze(regressor(encoded_val_X, training=False))
#
#                 val_pearson = pearson_correlation(y_pred=val_preds, y_true=val_Y)
#                 val_spearman = spearman_correlation(y_pred=val_preds, y_true=val_Y)
#                 val_mse = keras.losses.mean_squared_error(y_pred=val_preds, y_true=val_Y)
#                 val_mae = keras.losses.mean_absolute_error(y_pred=val_preds, y_true=val_Y)
#
#                 total_val_pearson += val_pearson / float(target_df.shape[-1])
#                 total_val_spearman += val_spearman / float(target_df.shape[-1])
#                 total_val_mse += val_mse / float(target_df.shape[-1])
#                 total_val_mae += val_mae / float(target_df.shape[-1])
#
#                 validation_history['val_pearson'][drug].append(val_pearson)
#                 validation_history['val_spearman'][drug].append(val_spearman)
#                 validation_history['val_mse'][drug].append(val_mse)
#                 validation_history['val_mae'][drug].append(val_mae)
#
#                 total_loss += loss_value / float(target_df.shape[-1])
#                 total_loss += sum(mut_supervisor_dict[drug].losses)
#                 to_train_variables.extend(mut_supervisor_dict[drug].trainable_variables)
#
#             print('Training loss (total) at epoch %s: %s' % (epoch + 1, float(total_loss)))
#             training_history['train_total']['pearson'].append(total_train_pearson)
#             training_history['train_total']['spearman'].append(total_train_spearman)
#             training_history['train_total']['mse'].append(total_train_mse)
#             training_history['train_total']['mae'].append(total_train_mae)
#
#             validation_history['val_total']['pearson'].append(total_val_pearson)
#             validation_history['val_total']['spearman'].append(total_val_spearman)
#             validation_history['val_total']['mse'].append(total_val_mse)
#             validation_history['val_total']['mae'].append(total_val_mae)
#
#             print(validation_history['val_total'][validation_monitoring_metric][-1])
#             print(best_overall_metric)
#             if validation_history['val_total'][validation_monitoring_metric][-1] > best_overall_metric:
#
#                 print('best!')
#                 print('best!')
#                 print('best!')
#                 print('best!')
#                 print('best!')
#
#                 best_overall_metric = validation_history['val_total'][validation_monitoring_metric][-1]
#                 encoder.save_weights(os.path.join(output_folder, 'fine_tuned_encoder_weights'), save_format='tf')
#                 shared_regressor_module.save_weights(os.path.join(output_folder, 'shared_regressor_weights'), save_format='tf')
#                 for drug in target_df.columns:
#                     mut_supervisor_dict[drug].save_weights(os.path.join(output_folder, drug + '_regressor_weights'),
#                                                            save_format='tf')
#
#             print(len(to_train_variables))
#             grads = tape.gradient(total_loss, to_train_variables)
#             optimizer.apply_gradients(zip(grads, to_train_variables))
#
#     return training_history, validation_history

# def fine_tune_mut_encoder_with_GAN(encoder, raw_X,
#                                    target_df,
#                                    validation_X=None,
#                                    validation_target_df=None,
#                                    mlp_architecture=None,
#                                    mlp_output_act_fn=keras.activations.sigmoid,
#                                    mlp_output_dim=1,
#                                    optimizer=keras.optimizers.Adam(learning_rate=model_config.fine_tuning_lr),
#                                    loss_fn=penalized_mean_squared_error,
#                                    validation_monitoring_metric='pearson',
#                                    max_epoch=100,
#                                    min_epoch=10,
#                                    gradual_unfreezing_flag=True,
#                                    unfrozen_epoch=5
#                                    ):
#     if mlp_architecture is None:
#         mlp_architecture = [64, 32]
#
#     output_folder = os.path.join('saved_weights', 'mut', repr(encoder) + '_encoder_weights')
#     reference_folder = os.path.join('saved_weights', 'gex', repr(encoder) + '_encoder_weights')
#     safe_make_dir(output_folder)
#
#     mut_supervisor_dict = dict()
#     best_overall_metric = float('-inf')
#
#     training_history = {
#         'train_pearson': defaultdict(list),
#         'train_spearman': defaultdict(list),
#         'train_mse': defaultdict(list),
#         'train_mae': defaultdict(list),
#         'train_total': defaultdict(list)
#     }
#
#     validation_history = {
#         'val_pearson': defaultdict(list),
#         'val_spearman': defaultdict(list),
#         'val_mse': defaultdict(list),
#         'val_mae': defaultdict(list),
#         'val_total': defaultdict(list)
#     }
#     free_layers = len(encoder.layers)
#
#     if gradual_unfreezing_flag:
#         encoder.trainable = False
#     shared_regressor_module = module.MLPBlock(architecture=model_config.shared_regressor_architecture,
#                                               output_act_fn=model_config.shared_regressor_act_fn,
#                                               output_dim=model_config.shared_regressor_output_dim)
#     shared_regressor_module.load_weights(os.path.join(reference_folder, 'shared_regressor_weights'))
#     for drug in target_df.columns:
#         mut_supervisor_dict[drug] = module.MLPBlock(architecture=mlp_architecture, output_act_fn=mlp_output_act_fn,
#                                                     output_dim=mlp_output_dim)
#         mut_supervisor_dict[drug].load_weights(os.path.join(reference_folder, drug + '_regressor_weights'))
#
#     for epoch in range(max_epoch):
#
#         total_train_pearson = 0.
#         total_train_spearman = 0.
#         total_train_mse = 0.
#         total_train_mae = 0.
#
#         total_val_pearson = 0.
#         total_val_spearman = 0.
#         total_val_mse = 0.
#         total_val_mae = 0.
#
#         with tf.GradientTape() as tape:
#             total_loss = sum(encoder.losses)
#             total_loss += sum(shared_regressor_module.losses)
#             to_train_variables = encoder.trainable_variables
#             if gradual_unfreezing_flag:
#                 if epoch > min_epoch:
#                     free_layers -= 1
#                     if (epoch - min_epoch) % unfrozen_epoch == 0:
#                         free_layers -= 1
#                     for i in range(len(encoder.layers) - 1, free_layers - 1, -1):
#                         to_train_variables.extend(encoder.layers[i].trainable_variables)
#                     if free_layers <= 0:
#                         gradual_unfreezing_flag = False
#                         encoder.trainable = True
#
#             to_train_variables.extend(shared_regressor_module)
#             for drug in target_df.columns:
#                 # model = keras.Sequential()
#                 # model.add(encoder)
#                 # model.add(mut_supervisor_dict[drug])
#
#                 y = target_df.loc[~target_df[drug].isna(), drug]
#                 y = y.astype('float32')
#                 X = raw_X.loc[y.index]
#                 X = X.astype('float32')
#
#                 if validation_X is None:
#                     kfold = KFold(n_splits=5, shuffle=True, random_state=2020)
#                     cv_splits = list(kfold.split(X))
#                     train_index, test_index = cv_splits[0]
#
#                     train_X, train_Y = X.iloc[train_index], y.iloc[train_index]
#                     # assert all(train_X.index == train_Y.index)
#                     val_X, val_Y = X.iloc[test_index], y.iloc[test_index]
#                     # assert all(val_X.index == val_Y.index)
#                     train_X, train_Y = train_X.values, train_Y.values
#                     val_X, val_Y = val_X.values, val_Y.values
#
#                 else:
#                     pass
#                 if repr(encoder).startswith('stochastic'):
#                     encoded_X = encoder(train_X, training=True)[0]
#                 else:
#                     encoded_X = encoder(train_X, training=True)
#
#                 regressor = keras.Sequential()
#                 regressor.add(shared_regressor_module)
#                 regressor.add(mut_supervisor_dict[drug])
#
#                 preds = tf.squeeze(regressor(encoded_X, training=True))
#                 loss_value = loss_fn(y_pred=preds, y_true=train_Y)
#
#                 print('Training loss (for %s) at epoch %s: %s' % (drug, epoch + 1, float(loss_value)))
#                 train_pearson = pearson_correlation(y_pred=preds, y_true=train_Y)
#                 train_spearman = spearman_correlation(y_pred=preds, y_true=train_Y)
#                 train_mse = keras.losses.mean_squared_error(y_pred=preds, y_true=train_Y)
#                 train_mae = keras.losses.mean_absolute_error(y_pred=preds, y_true=train_Y)
#
#                 total_train_pearson += train_pearson / float(target_df.shape[-1])
#                 total_train_spearman += train_spearman / float(target_df.shape[-1])
#                 total_train_mse += train_mse / float(target_df.shape[-1])
#                 total_train_mae += train_mae / float(target_df.shape[-1])
#
#                 training_history['train_pearson'][drug].append(train_pearson)
#                 training_history['train_spearman'][drug].append(train_spearman)
#                 training_history['train_mse'][drug].append(train_mse)
#                 training_history['train_mae'][drug].append(train_mae)
#
#                 if repr(encoder).startswith('stochastic'):
#                     encoded_val_X = encoder(val_X, training=False)[0]
#                 else:
#                     encoded_val_X = encoder(val_X, training=False)
#
#                 val_preds = tf.squeeze(regressor(encoded_val_X, training=False))
#
#                 val_pearson = pearson_correlation(y_pred=val_preds, y_true=val_Y)
#                 val_spearman = spearman_correlation(y_pred=val_preds, y_true=val_Y)
#                 val_mse = keras.losses.mean_squared_error(y_pred=val_preds, y_true=val_Y)
#                 val_mae = keras.losses.mean_absolute_error(y_pred=val_preds, y_true=val_Y)
#
#                 total_val_pearson += val_pearson / float(target_df.shape[-1])
#                 total_val_spearman += val_spearman / float(target_df.shape[-1])
#                 total_val_mse += val_mse / float(target_df.shape[-1])
#                 total_val_mae += val_mae / float(target_df.shape[-1])
#
#                 validation_history['val_pearson'][drug].append(val_pearson)
#                 validation_history['val_spearman'][drug].append(val_spearman)
#                 validation_history['val_mse'][drug].append(val_mse)
#                 validation_history['val_mae'][drug].append(val_mae)
#
#                 total_loss += loss_value / float(target_df.shape[-1])
#                 total_loss += sum(mut_supervisor_dict[drug].losses)
#                 to_train_variables.extend(mut_supervisor_dict[drug].trainable_variables)
#
#             print('Training loss (total) at epoch %s: %s' % (epoch + 1, float(total_loss)))
#             training_history['train_total']['pearson'].append(total_train_pearson)
#             training_history['train_total']['spearman'].append(total_train_spearman)
#             training_history['train_total']['mse'].append(total_train_mse)
#             training_history['train_total']['mae'].append(total_train_mae)
#
#             validation_history['val_total']['pearson'].append(total_val_pearson)
#             validation_history['val_total']['spearman'].append(total_val_spearman)
#             validation_history['val_total']['mse'].append(total_val_mse)
#             validation_history['val_total']['mae'].append(total_val_mae)
#
#             if validation_history['val_total'][validation_monitoring_metric][-1] > best_overall_metric:
#                 best_overall_metric = validation_history['val_total'][validation_monitoring_metric][-1]
#                 encoder.save_weights(os.path.join(output_folder, 'fine_tuned_encoder_weights'), save_format='tf')
#                 shared_regressor_module.save_weights(os.path.join(output_folder, 'shared_regressor_weights'), save_format='tf')
#                 for drug in target_df.columns:
#                     mut_supervisor_dict[drug].save_weights(os.path.join(output_folder, drug + '_regressor_weights'),
#                                                            save_format='tf')
#
#             # print(len(to_train_variables))
#             grads = tape.gradient(total_loss, to_train_variables)
#             optimizer.apply_gradients(zip(grads, to_train_variables))
#
#     return training_history, validation_history
