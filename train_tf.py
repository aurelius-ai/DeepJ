import tensorflow as tf
from tensorflow import keras
from dataset import *
from constants import *
from model_tf import *

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
    train_data, train_labels = train_batcher()

    cbs = [
        keras.callbacks.EarlyStopping(monitor='loss', patience=20)
    ]

    print('Training...')
    model.fit(train_data, train_labels, epochs=1000, callbacks=cbs, batch_size=BATCH_SIZE)

if __name__ == '__main__':
    main()
