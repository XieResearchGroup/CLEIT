from tensorflow import keras
#encoder configurations
encoder_architecture = [512, 256, 128]
encoder_latent_dimension = 128
encoder_pre_training_learning_rate = 1.0
#shared regressor module configurations
shared_regressor_architecture = []
shared_regressor_output_dim = 64
shared_regressor_act_fn = keras.activations.relu

#individual regressor module configurations

regressor_architecture = []
regressor_output_dim = 1
regressor_act_fn = keras.activations.relu