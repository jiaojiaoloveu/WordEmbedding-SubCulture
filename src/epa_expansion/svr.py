from sklearn.svm import SVR
from neural_network import generate_data
from propagate_labels import word_dataset_base
from gen_data import wv_map
from time import time
import os
import json
import numpy as np
import argparse


def train(wv):
    generate = args.get('generate')
    feature_train, label_train, feature_test, label_test = generate_data(generate=generate)
    model = args.get('model')
    if model == 'svr':
        clf = SVR(kernel='rbf', epsilon=0.05, gamma=0.0, C=10)
        for axis in range(0, 3):
            start = time()
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
            label_space = np.array(label_space)
            print(label_space)
            print('time %s' % (time() - start))
            print('mean google radius')
            gg_radius = label_space[:, 1]
            print(np.mean(np.abs(gg_radius)))
            print(np.std(gg_radius))
            print('mean github radius')
            gh_radius = label_space[:, 0]
            print(np.mean(np.abs(gh_radius)))
            print(np.std(gh_radius))


if __name__ == '__main__':
    ap = argparse.ArgumentParser('keras deep learning method')
    ap.add_argument('--generate', type=int, required=True)
    ap.add_argument('--model', type=str, required=True)
    args = vars(ap.parse_args())
    train(wv_map())
