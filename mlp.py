import functools

import tensorflow as tf

def prepare_mlp_model(input_shape, num_labels, hidden_nodes = 50):
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(functools.reduce(lambda x, y: x*y, input_shape), activation='relu'),
        tf.keras.layers.Dense(hidden_nodes, activation='relu'),
        tf.keras.layers.Dense(num_labels, activation='softmax'),
    ])

    return model
