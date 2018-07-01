from sklearn.svm import SVR
from neural_network import generate_data
import numpy as np
import argparse


def train():
    generate = args.get('generate')
    feature_train, label_train, feature_test, label_test = generate_data(generate=generate)
    model = args.get('model')
    if model == 'svr':
        clf = SVR()
        clf.fit(feature_train, label_train[:, 0])
        score = clf.score(feature_test, label_test[:, 0])
        print(score)


if __name__ == '__main__':
    print('svr')
    ap = argparse.ArgumentParser('keras deep learning method')
    ap.add_argument('--generate', type=int, required=True)
    ap.add_argument('--model', type=str, required=True)
    args = vars(ap.parse_args())
