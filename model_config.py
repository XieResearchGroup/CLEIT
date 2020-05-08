from tensorflow import keras
#encoder configurations
encoder_architecture = [512, 256, 128]
encoder_latent_dimension = 128
#regressor module configurations
regressor_architecture = [128, 128, 64, 16]
regressor_shared_layer_number = 2
regressor_act_fn = keras.activations.relu
regressor_output_dim = 265
regressor_output_act_fn =keras.activations.sigmoid

#learning configurations
kernel_regularizer_l = 0.0001
pre_training_lr = 1e-4
fine_tuning_lr = 1e-4
decay = 0.5
max_epoch = 20
min_epoch = 10
gradual_unfreezing_flag = True
unfrozen_epoch = 5
batch_size = 64
alpha = 0.5
