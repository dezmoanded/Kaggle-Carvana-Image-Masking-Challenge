import cv2
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
import os
import sys
sys.path.insert(0,'..')
from tqdm import tqdm

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

model_names = ["C", "D", "E", "G"]

model = get_ensemble_model([params.orig_height, params.orig_width, len(model_names) + 3])

df_train = pd.read_csv('../input/train_masks.csv')
ids_train = df_train['img'].map(lambda s: s.split('.')[0])

ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.1, random_state=42)


def get_model_rows(prefix):
    return {model: pd.read_csv("{}{}.csv.gz".format(prefix, model)).iterrows() for model in model_names}

def get_x_batch_factory(models_rows, data_dir='../input/train_hq'):
    def get_x_batch(ids, start):
        x_batch = []
        end = min(start + batch_size, len(ids))
        ids_batch = ids[start:end]
        for id in ids_batch.values:
            img = cv2.imread('{}/{}.jpg'.format(data_dir, id))
            img = img / 255

            def load_file(model):
                row = next(models_rows[model])[1]
                # row = model_df[model_df.img == "{}.jpg".format(id)]
                rle = row.rle_mask
                prob = run_length_decode(rle, params.orig_width, params.orig_height)
                return np.expand_dims(prob, axis=2)

            predictions = [np.expand_dims(img[:, :, i], axis=2) for i in range(3)]
            # img = ""
            # for model_name in model_names:
            #     img, prob = load_file(model_name)
            #     predictions += [prob]
            predictions += [load_file(model_name)
                            for model_name in model_names]
            predictions = np.concatenate(predictions, axis=2)
            x_batch.append(predictions)

        x_batch = np.array(x_batch, np.float32)
        return x_batch

    return get_x_batch

def generator(folder, ids_split):
    while True:
        print("Loading predicted masks")

        models_rows = get_model_rows('train_submit/{}_submission'.format(folder))
    
        for start in range(0, len(ids_split), batch_size):
            y_batch = []
            end = min(start + batch_size, len(ids_split))
            ids_batch = ids_split[start:end]

            for id in ids_batch.values:
                mask = cv2.imread('../input/train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
                mask = np.expand_dims(mask, axis=2)
                y_batch.append(mask)

            y_batch = np.array(y_batch, np.float32) / 255

            x_batch = get_x_batch(ids_split, start, models_rows)
            yield x_batch, y_batch

def train_generator():
    for x_batch, y_batch in generator("train_submission", ids_train_split):
        yield x_batch, y_batch

def valid_generator():
    for x_batch, y_batch in generator("valid_submission", ids_valid_split):
        yield x_batch, y_batch

if __name__ == "__main__":
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

    model.load_weights('weights/best_weights.hdf5')
    model.fit_generator(generator=train_generator(),
                        steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
                        epochs=epochs,
                        verbose=2,
                        callbacks=callbacks,
                        validation_data=valid_generator(),
                        validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)),
                        initial_epoch=14)