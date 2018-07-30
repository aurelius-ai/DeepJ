import tensorflow as tf
from tensorflow import keras
from constants import *

def build_model(style_units=32):
    model = keras.Sequential()
    # Model will take as input arrays of shape (*, NUM_STYLES)
    # and output arrays of arrays of shape (*, style_units)
    # After first layer, you don't need to specify the size of input anymore
    # Distributed style representation
    model.add(keras.layers.Dense(style_units, activation=tf.nn.relu, input_shape=(SEQ_LEN,)))
    # TODO: Embedding layer
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, epsilon=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
