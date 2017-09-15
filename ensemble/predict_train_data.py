import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0,'..')

from test_submit_multithreaded import predict

def predict_train_data(model_name = "A"):
    df_train = pd.read_csv('../input/train_masks.csv')
    ids_train = df_train['img'].map(lambda s: s.split('.')[0])

    ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.1, random_state=42)

    def train_callback(prob, id):
        np.save("model{}/train/train_predictions/{}".format(model_name, id), prob)

    def valid_callback(prob, id):
        np.save("model{}/train/valid_predictions/{}".format(model_name, id), prob)

    print('Predicting {} samples'.format(len(ids_train_split)))
    predict(ids_train_split, train_callback, '../weights/best_weights{}.hdf5'.format(model_name), '../input/train_hq')

    print('Predicting {} samples'.format(len(ids_valid_split)))
    predict(ids_valid_split, valid_callback, '../weights/best_weights{}.hdf5'.format(model_name), '../input/train_hq')

if __name__ == "__main__":
    predict_train_data(sys.argv[1])