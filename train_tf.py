import numpy as np
import tensorflow as tf
from tensorflow import keras
from dataset import *
from constants import *
from model_tf import *
from util import *

BATCH_SIZE = 16

def main():
    model = build_model()
    model.summary()
    train(model)

def train(model):
    print('Loading data...')
    data = process(load(tensorflow=True), tensorflow=True)
    train_data_and_labels, val_data_and_labels = validation_split(data, tensorflow=True)
    train_batcher = batcher(sampler(train_data_and_labels, tensorflow=True), BATCH_SIZE, tensorflow=True)
    val_batcher = batcher(sampler(val_data_and_labels, tensorflow=True), BATCH_SIZE, tensorflow=True)
    train_notes, train_styles = train_batcher()
     # Convert style labels to one hot vectors
    train_styles = one_hot_batch(train_styles, NUM_STYLES, tensorflow=True)
    train_notes = train_notes.astype(float)
    inputs = train_notes[:, :-1]
    targets = train_notes[:, 1:]
    # Tensorflow cross entropy loss expects one hot vectors for targets
    targets = one_hot_seq(targets, NUM_ACTIONS, tensorflow=True)
    train_notes = [inputs, train_styles]

    cbs = [
        keras.callbacks.EarlyStopping(monitor='loss', patience=20)
    ]

    print('Training...')
    model.fit(train_notes, targets, epochs=1000, callbacks=cbs, batch_size=BATCH_SIZE)

if __name__ == '__main__':
    main()
