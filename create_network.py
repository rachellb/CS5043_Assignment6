import tensorflow as tf
from tensorflow.keras.layers import (Input, Embedding, SimpleRNN, Dense,
                                     BatchNormalization, Dropout, Activation)

from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, Sequential

from metrics_binarized import *

def create_network(vocab_size,
                   output_dim,
                   len_max,
                   dense_layers,
                   n_neurons=10,
                   activation=None,
                   activation_dense=None,
                   lambda_regularization=None,
                   use_gru=False,
                   dropout=None,
                   lrate=0.0001):


    if lambda_regularization is not None:
       lambda_regularization = tf.keras.regularizers.l2(lambda_regularization)

    # Initialize the model
    model = Sequential()

    # Adds an embedding layer
    # Input_dim = size of the vocabulary
    # Output_dim = length of the vector for each word (essentially a hyperparameter)
    # input_length = maximum length of a sequence
    model.add(Embedding(input_dim=vocab_size, output_dim=output_dim, input_length=len_max))

    model.add(SimpleRNN(n_neurons,
                        activation=activation,
                        use_bias=True,
                        return_sequences=False, # Produce entire sequence of outputs
                        kernel_initializer='random_uniform',
                        kernel_regularizer=lambda_regularization,
                        unroll=True)) # Laying this out on the GPU.
                                      # Different timesteps can get allocated to different parts of GPU
                                      # Runs really fast if enough space on GPU

    for i in range(len(dense_layers)):
        model.add(Dense(units=dense_layers[i]['units'],
                        activation=activation_dense,
                        use_bias=True,
                        kernel_initializer='random_uniform',
                        kernel_regularizer=lambda_regularization))

        if dropout:
            model.add(Dropout(rate=dropout))


    model.add(Dense(units=1,
                    use_bias=True,
                    kernel_initializer='random_uniform',
                    activation='sigmoid',
                    kernel_regularizer=lambda_regularization))

    # The optimizer determines how the gradient descent is to be done
    opt = tf.keras.optimizers.Adam(learning_rate=lrate, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)


    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.426),
                                                                      'AUC'])




    return model