import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.insert(0,'..')

from test_submit_multithreaded import predict, run_length_encode
from params import *

from predict_train_data import ModelConfig
from ensemble_model import batch_size
from train_ensemble import get_model_rows, get_x_batch_factory, model

rles = []

def predict_test_data(model_name):
    df_test = pd.read_csv('../input/sample_submission.csv')
    ids_test = df_test['img'].map(lambda s: s.split('.')[0])

    names = []
    for id in ids_test:
        names.append('{}.jpg'.format(id))

    def test_callback(prob, id):
        global rles
        mask = prob > threshold
        rle = run_length_encode(mask)
        rles.append(rle)

        # np.save("{}/{}.npy".format(test_dir, id), prob)

    model_config = ModelConfig(model,
                               'weights/best_weights.hdf5',
                               -1,
                               batch_size)

    models_rows = get_model_rows("../submit/submission")
    get_x_batch = get_x_batch_factory(models_rows, "../input/test_hq")

    print('Predicting {} samples'.format(len(ids_test)))
    predict(ids_test, test_callback, model_config, '../input/test_hq', get_x_batch)

    global rles
    print("Generating submission file...")
    df = pd.DataFrame({'img': names, 'rle_mask': rles})
    df.to_csv("../submit/submission{}.csv.gz".format(model_name), index=False, compression='gzip')

if __name__ == "__main__":
    predict_test_data("Ensemble")