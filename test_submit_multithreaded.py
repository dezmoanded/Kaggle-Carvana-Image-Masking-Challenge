import cv2
import numpy as np
import pandas as pd
import threading
import queue
import tensorflow as tf
from tqdm import tqdm

import params

input_size = params.input_size
batch_size = params.batch_size
orig_width = params.orig_width
orig_height = params.orig_height
threshold = params.threshold
model = params.model_factory()

# https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle

class ModelConfig:
    def __init__(self, model, weights_path, input_size, batch_size, resize=True, pad=False):
        self.model = model
        self.weights_path = weights_path
        self.input_size = input_size
        self.batch_size = batch_size
        self.resize = resize
        self.pad = pad

def predict(ids, callback, model_config, data_dir='input/test_hq'):
    model = model_config.model
    input_size = model_config.input_size
    batch_size = model_config.batch_size

    model.load_weights(filepath=model_config.weights_path)
    graph = tf.get_default_graph()
    q_size = 10

    def data_loader(q, ):
        for start in range(0, len(ids), batch_size):
            x_batch = []
            end = min(start + batch_size, len(ids))
            ids_test_batch = ids[start:end]
            for id in ids_test_batch.values:
                img = cv2.imread('{}/{}.jpg'.format(data_dir, id))
                if model_config.resize:
                    img = cv2.resize(img, (input_size, input_size))
                if model_config.pad:
                    img = cv2.copyMakeBorder(img, 0, 0, 1, 1, cv2.BORDER_REPLICATE)
                x_batch.append(img)
            x_batch = np.array(x_batch, np.float32) / 255
            q.put(x_batch)

    def predictor(q, ):
        for i in tqdm(range(0, len(ids), batch_size)):
            x_batch = q.get()
            with graph.as_default():
                preds = model.predict_on_batch(x_batch)
            preds = np.squeeze(preds, axis=3)
            end = min(i + batch_size, len(ids))
            ids_test_batch = ids[i:end]
            for pred, id in zip(preds, ids_test_batch.values):
                if model_config.resize:
                    prob = cv2.resize(pred, (orig_width, orig_height))
                if model_config.pad:
                    prob = pred[:, 1:-1]
                q2.put((prob, id))
                callback(prob, id)

    def upload(q, ):
        for i in tqdm(range(0, len(ids))):
            prob, id = q2.get()
            callback(prob, id)

    q = queue.Queue(maxsize=q_size)
    q2 = queue.Queue(maxsize=q_size * batch_size)
    t1 = threading.Thread(target=data_loader, name='DataLoader', args=(q,))
    t2 = threading.Thread(target=predictor, name='Predictor', args=(q,))
    t3 = threading.Thread(target=upload, name='Data upload', args=(q,))
    print('Predicting on {} samples with batch_size = {}...'.format(len(ids), batch_size))
    t1.start()
    t2.start()
    t3.start()
    # Wait for both threads to finish
    t1.join()
    t2.join()
    t3.join()


if __name__ == "__main__":
    df_test = pd.read_csv('input/sample_submission.csv')
    ids_test = df_test['img'].map(lambda s: s.split('.')[0])

    names = []
    for id in ids_test:
        names.append('{}.jpg'.format(id))

    rles = []

    def callback(prob, id):
        mask = prob > threshold
        rle = run_length_encode(mask)
        rles.append(rle)

    predict(ids_test, callback, 'weights/best_weights.hdf5', 'input/test_hq')

    print("Generating submission file...")
    df = pd.DataFrame({'img': names, 'rle_mask': rles})
    df.to_csv('submit/submission.csv.gz', index=False, compression='gzip')
