from tensorflow import keras
#encoder configurations
encoder_architecture = [512, 256, 128]
encoder_latent_dimension = 128
encoder_act_fn = keras.activations.relu
encoder_output_act_fn = None

#regressor module configurations
regressor_architecture = [128, 128, 64, 16]
regressor_shared_layer_number = 2
regressor_act_fn = keras.activations.relu
regressor_output_dim = 265
regressor_output_act_fn =keras.activations.sigmoid
#transmitter module configurations
transmitter_architecture = [128, 128]
transmitter_act_fn = keras.activations.relu
transmitter_output_act_fn = None
transmitter_output_dim = encoder_latent_dimension
#learning configurations
kernel_regularizer_l = 0.0001
pre_training_lr = 5e-4 #fixed, may need to loose regularization for pre-training
fine_tuning_lr = 1e-3 #too small learning rates may leads to over-fitting like behavior
decay = 0.5
max_epoch = 300
min_epoch = 80
gradual_unfreezing_flag = True
unfrozen_epoch = 20
batch_size = 32
alpha = 0.9
gradient_threshold = 1e-3
noise_fn = keras.layers.GaussianNoise(0.001),
