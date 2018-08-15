import tensorflow as tf
from tensorflow import keras
from constants import *

NUM_UNITS = 512
STYLE_UNITS = 32

def deepj():
    style_dense = keras.layers.Dense(STYLE_UNITS, activation=tf.nn.relu, input_shape=(NUM_STYLES,))
    output_dense = keras.layers.Dense(NUM_ACTIONS, activation=tf.nn.softmax)

    def f(x, style):
        seq_len = x.shape[1]

        style = style_dense(style)
        style = keras.layers.Lambda(lambda z: tf.expand_dims(z, 1))(style)
        style = keras.layers.Lambda(lambda z: tf.tile(z, [1, seq_len, 1]))(style)
        # style = tf.to_float(style)
        x = keras.layers.Embedding(1, NUM_UNITS)(x)
        x = keras.layers.Concatenate(axis=2)([x, style])
        x = keras.layers.LSTM(512, return_sequences=True)(x)
        x = output_dense(x)
        return x
    return f

def build_model(style_units=32):
    seq_in = keras.Input(shape=(SEQ_LEN - 1,))
    style_in = keras.Input(shape=(NUM_STYLES,))
    # seq = keras.layers.Dropout(0.2)(seq_in)
    # style = keras.layers.Dropout(0.2)(style_in)
    # output = keras.layers.Dense(NUM_ACTIONS, activation=tf.nn.softmax)
    output = deepj()(seq_in, style_in)
    model = keras.Model([seq_in, style_in], [output])
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, epsilon=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
