import argparse
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow import keras
from dataset import *
from constants import *
from model_tf import *
from util import *

def generator(batcher):
    """
    Generate tuples of note inputs and targets in order to use fit_generator
    """
    while True:
        notes, styles = batcher()
        # Convert style labels to one hot vectors
        styles = one_hot_batch(styles, NUM_STYLES, tensorflow=True)
        notes = notes.astype(float)
        inputs = notes[:, :-1]
        targets = notes[:, 1:]
        # Tensorflow cross entropy loss expects targets to be in categorical format
        targets = keras.utils.to_categorical(targets, NUM_ACTIONS)
        notes = [inputs, styles]
        yield (notes, targets)

def train(model, batch_size, epochs):
    print('Loading data...')
    data = process(load(tensorflow=True), tensorflow=True)
    train_data_and_labels, val_data_and_labels = validation_split(data, tensorflow=True)
    train_batcher = batcher(sampler(train_data_and_labels, tensorflow=True), batch_size, tensorflow=True)
    val_batcher = batcher(sampler(val_data_and_labels, tensorflow=True), batch_size, tensorflow=True)
    # train_notes, train_styles = train_batcher()
    # # Convert style labels to one hot vectors
    # train_styles = one_hot_batch(train_styles, NUM_STYLES, tensorflow=True)
    # train_notes = train_notes.astype(float)
    # inputs = train_notes[:, :-1]
    # targets = train_notes[:, 1:]
    # # Tensorflow cross entropy loss expects targets to be in categorical format
    # targets = keras.utils.to_categorical(targets, NUM_ACTIONS)
    # train_notes = [inputs, train_styles]

    cbs = [
        keras.callbacks.ModelCheckpoint(MODEL_FILE, monitor='loss', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='loss', patience=20)
    ]

    print('Training...')
    # model.fit(train_notes, targets, epochs=epochs, callbacks=cbs, batch_size=batch_size)
    train_generator = generator(train_batcher)
    model.fit_generator(train_generator, steps_per_epoch=TRAIN_CYCLES, epochs=epochs, callbacks=cbs)
    tfjs.converters.save_keras_model(model, OUT_DIR)

def main():
    parser = argparse.ArgumentParser(description='Trains model')
    parser.add_argument('--batch-size', default=128, type=int, help='Size of the batch')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of epochs to train on')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    args = parser.parse_args()

    print('Batch Size: {}'.format(args.batch_size))
    print('Epochs: {}'.format(args.epochs))
    print('Learning Rate: {}'.format(args.lr))
    print()

    model = build_model(args.lr)
    model.summary()
    train(model, args.batch_size, args.epochs)

if __name__ == '__main__':
    main()
