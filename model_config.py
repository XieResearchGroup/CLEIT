from tensorflow import keras
#encoder configurations
encoder_architecture = [512, 256, 128]
encoder_latent_dimension = 128
#shared regressor module configurations
shared_regressor_architecture = [128, 128]
shared_regressor_output_dim = 64
shared_regressor_act_fn = keras.activations.relu
#individual regressor module configurations
regressor_architecture = [32, 16]
regressor_act_fn = keras.activations.sigmoid

#learning configurations
kernel_regularizer_l = 0.0001
pre_training_lr = 1e-4
fine_tuning_lr = 1e-5
max_epoch = 100
min_epoch = 10
gradual_unfreezing_flag = True
unfrozen_epoch = 50
batch_size = 64
alpha = 1.0
