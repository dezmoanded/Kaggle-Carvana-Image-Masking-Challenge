import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.insert(0,'..')

from test_submit_multithreaded import predict, ModelConfig, run_length_encode
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
                     4,
                     resize=False,
                     pad=True)
}

rles = []

def predict_train_data(model_name):
    df_train = pd.read_csv('../input/train_masks.csv')
    ids_train = df_train['img'].map(lambda s: s.split('.')[0])

    ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.1, random_state=42)

    threshold = .5

    def callback(prob, id):
        global rles
        mask = prob > threshold
        rle = run_length_encode(mask)
        rles.append(rle)

    model_config = model_configs[model_name]

    global rles
    
    rles = []
    print('Predicting {} samples'.format(len(ids_train_split)))
    predict(ids_train_split, callback, model_config, '../input/train_hq')

    names = []
    for id in ids_train_split:
        names.append('{}.jpg'.format(id))
    print("Generating submission file...")
    df = pd.DataFrame({'img': names, 'rle_mask': rles})
    df.to_csv("train_submit/train_submission{}.csv.gz".format(model_name), index=False, compression='gzip')
    
    rles = []
    print('Predicting {} samples'.format(len(ids_valid_split)))
    predict(ids_valid_split, callback, model_config, '../input/train_hq')

    names = []
    for id in ids_valid_split:
        names.append('{}.jpg'.format(id))
    print("Generating submission file...")
    df = pd.DataFrame({'img': names, 'rle_mask': rles})
    df.to_csv("train_submit/valid_submission{}.csv.gz".format(model_name), index=False, compression='gzip')

if __name__ == "__main__":
    predict_train_data(sys.argv[1])