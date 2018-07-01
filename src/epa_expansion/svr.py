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
        for axis in range(0, 3):
            label_train_axis = label_train[:, axis]
            label_test_axis = label_test[:, axis]
            print('start training')
            clf.fit(feature_train, label_train_axis)
            score = clf.score(feature_test, label_test_axis)
            print('score %s' % score)
            label_test_axis_pre = clf.predict(feature_test)
            mae = np.mean(np.abs(label_test_axis_pre - label_test_axis))
            print('mae: %s' % mae)


if __name__ == '__main__':
    ap = argparse.ArgumentParser('keras deep learning method')
    ap.add_argument('--generate', type=int, required=True)
    ap.add_argument('--model', type=str, required=True)
    args = vars(ap.parse_args())
    train()
