from sklearn.svm import SVR
from neural_network import generate_data
from propagate_labels import word_dataset_base
from gen_data import wv_map
from time import time
from sample_seeds import __uni2norm
import os
import json
import numpy as np
import argparse


def train(wv):
    generate = args.get('generate')
    uniform = args.get('uniform')
    feature_train, label_train, feature_test, label_test = generate_data(generate=generate, uniform=uniform)
    model = args.get('model')
    if model == 'svr':
        clf = SVR(kernel='rbf', epsilon=0.05, gamma='auto', C=10)
        gg_label_pred, gh_label_pred = [], []
        for axis in range(0, 3):
            label_train_axis = label_train[:, axis]
            label_test_axis = label_test[:, axis]
            print('start training')
            clf.fit(feature_train, label_train_axis)
            score = clf.score(feature_test, label_test_axis)
            print('score %s' % score)
            label_test_axis_pre = clf.predict(feature_test)
            with open(os.path.join(word_dataset_base, '%s_result_%s' % (model, axis)), 'w') as fp:
                zipped = list(zip(label_test_axis_pre.tolist(), label_test_axis.tolist()))
                json.dump(zipped, fp)
            mae = np.mean(np.abs(label_test_axis_pre - label_test_axis))
            print('mae: %s' % mae)
            label_space = []
            for w in wv:
                label_space.append(clf.predict(wv[w]))
            gh_label_pred.append(label_space[:, 0])
            gg_label_pred.append(label_space[:, 1])
        gh_label_pred = np.transpose(gh_label_pred)
        gg_label_pred = np.transpose(gg_label_pred)
        print('label pred shape')
        print(gh_label_pred.shape)
        print(gg_label_pred.shape)
        if uniform:
            gh_label_pred = __uni2norm(gh_label_pred)
            gg_label_pred = __uni2norm(gg_label_pred)
        print('google')
        print(np.mean(gg_label_pred, axis=0))
        print(np.mean(np.abs(gg_label_pred), axis=0))
        print(np.std(gg_label_pred, axis=0))
        print('github')
        print(np.mean(gh_label_pred, axis=0))
        print(np.mean(np.abs(gh_label_pred), axis=0))
        print(np.std(gh_label_pred, axis=0))


if __name__ == '__main__':
    ap = argparse.ArgumentParser('keras deep learning method')
    ap.add_argument('--generate', type=int, required=True)
    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--uniform', type=bool, required=True)
    args = vars(ap.parse_args())
    train(wv_map())
