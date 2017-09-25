import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.insert(0,'..')

from test_submit_multithreaded import predict, ModelConfig
from model.u_net import get_unet_128, get_unet_256, get_unet_512, get_unet_1024, get_unet_1024_heng, get_unet_1920x1280
import model.trained.modelD as modelD

model_configs = {
    "A": ModelConfig(get_unet_128(),
                     "../weights/best_weightsA.hdf5",
                     128,
                     16),
    "C": ModelConfig(get_unet_1024(),
                     "../weights/best_weightsC.hdf5",
                     1024,
                     6),
    "D": ModelConfig(modelD.get_unet_1024(),
                     "../weights/best_weightsD.hdf5",
                     1024,
                     5),
    "E": ModelConfig(get_unet_1024(),
                     "../weights/best_weightsE.hdf5",
                     1024,
                     7),
    "G": ModelConfig(get_unet_1920x1280(),
                     "../weights/best_weightsG.hdf5",
                     -1,
                     4)
}

def predict_train_data(model_name):
    df_train = pd.read_csv('../input/train_masks.csv')
    ids_train = df_train['img'].map(lambda s: s.split('.')[0])

    ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.1, random_state=42)

    train_dir = "/home/pl57/data/carvana/model{}/train/train_predictions".format(model_name)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    def train_callback(prob, id):
        # df = compress(prob)
        df.to_pickle("{}/{}.pkl".format(train_dir, id))

    valid_dir = "/home/pl57/data/carvana/model{}/train/valid_predictions".format(model_name)
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)

    def valid_callback(prob, id):
        # df = compress(prob)
        df.to_pickle("{}/{}.pkl".format(valid_dir, id))

    model_config = model_configs[model_name]

    print('Predicting {} samples'.format(len(ids_train_split)))
    predict(ids_train_split, train_callback, model_config, '../input/train_hq')

    print('Predicting {} samples'.format(len(ids_valid_split)))
    predict(ids_valid_split, valid_callback, model_config, '../input/train_hq')

if __name__ == "__main__":
    predict_train_data(sys.argv[1])