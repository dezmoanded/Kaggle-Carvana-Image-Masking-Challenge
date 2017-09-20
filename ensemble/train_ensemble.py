import cv2
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
import os
import sys
sys.path.insert(0,'..')

import params
from ensemble_model import get_ensemble_model, batch_size
from compress import decompress

def run_length_decode(rle, orig_width, orig_height):
    runs = np.array(rle.split(' ')).astype(int)
    runs[1::2] = runs[1::2] + runs[:-1:2]
    inds = np.zeros(orig_height * orig_width)
    runs = runs - 1
    starts = runs[::2]
    ends = runs[1::2]
    for start, end in zip(starts, ends):
        inds[start:end] = 1
    mask = inds.reshape(orig_height, orig_width)
    return mask

epochs = 100

model_names = ["A", "C", "D", "E"]

model = get_ensemble_model([params.orig_width, params.orig_height, len(model_names)])

df_train = pd.read_csv('../input/train_masks.csv')
ids_train = df_train['img'].map(lambda s: s.split('.')[0])

ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.1, random_state=42)

def generator(folder, ids_split):
    while True:
        for start in range(0, len(ids_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_split))
            ids_batch = ids_split[start:end]
            for id in ids_batch.values:
                def load_file(name):
                    compressed = pd.read_pickle(name)
                    prediction = decompress(compressed, params.orig_height, params.orig_width)
                    return np.expand_dims(prediction, axis=2)

                predictions = [load_file("model{}/train/{}/{}.pkl".format(model_name, folder, id))
                              for model_name in model_names]
                predictions = np.concatenate(predictions, axis=2)

                mask = cv2.imread('../input/train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(predictions)
                y_batch.append(mask)

            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch

def train_generator():
    for x_batch, y_batch in generator("train_predictions", ids_train_split):
        yield x_batch, y_batch

def valid_generator():
    for x_batch, y_batch in generator("valid_predictions", ids_valid_split):
        yield x_batch, y_batch

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=8,
                           verbose=1,
                           min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               epsilon=1e-4),
             ModelCheckpoint(monitor='val_loss',
                             filepath='weights/best_weights.hdf5',
                             save_best_only=True,
                             save_weights_only=True),
             TensorBoard(log_dir='logs')]

model.fit_generator(generator=train_generator(),
                    steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
                    epochs=epochs,
                    verbose=2,
                    callbacks=callbacks,
                    validation_data=valid_generator(),
                    validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)))