from sklearn.svm import SVR
from neural_network import generate_data
from propagate_labels import word_dataset_base
from gen_data import wv_map
from sample_seeds import __uni2norm, __norm2uni
import os
import json
import numpy as np
import argparse


def train():
    generate = args.get('generate')
    seed_size = args.get('seed')
    eval_size = args.get('eval')
    epa = args.get('epa')
    uniform = (args.get('uniform') == 1)
    feature_train, label_train, feature_test, label_test = generate_data(2, seed_size, eval_size, epa)

    clf = SVR(kernel='rbf', epsilon=0.05, C=10)
    if uniform:
        label_train = __norm2uni(label_train)
        label_test = __norm2uni(label_test)
    label_pred = []
    for axis in range(0, 3):
        label_train_axis = label_train[:, axis]
        print('start training')
        clf.fit(feature_train, label_train_axis)
        label_pred.append(clf.predict(feature_test))
    label_pred = np.array(label_pred)
    print(label_pred.shape)
    #     label_space = []
    #     for w in wv:
    #         label_space.append(clf.predict(wv[w]))
    #     label_space = np.array(label_space)
    #     print('label space shape')
    #     print(label_space.shape)
    #     gh_label_pred.append(label_space[:, 0])
    #     gg_label_pred.append(label_space[:, 1])
    # gh_label_pred = np.transpose(gh_label_pred)
    # gg_label_pred = np.transpose(gg_label_pred)
    # print('label pred shape')
    # print(gh_label_pred.shape)
    # print(gh_label_pred)
    # print(gg_label_pred.shape)
    # print(gg_label_pred)
    # if uniform:
    #     gh_label_pred = __uni2norm(gh_label_pred)
    #     gg_label_pred = __uni2norm(gg_label_pred)
    # print('after uni2norm')
    # print(gh_label_pred)
    # print(gg_label_pred)
    # print('google')
    # print(np.mean(gg_label_pred, axis=0))
    # print(np.mean(np.abs(gg_label_pred), axis=0))
    # print(np.std(gg_label_pred, axis=0))
    # print('github')
    # print(np.mean(gh_label_pred, axis=0))
    # print(np.mean(np.abs(gh_label_pred), axis=0))
    # print(np.std(gh_label_pred, axis=0))


if __name__ == '__main__':
    ap = argparse.ArgumentParser('keras deep learning method')
    ap.add_argument('--generate', type=int, required=False)
    ap.add_argument('--uniform', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--eval', type=int, required=True)
    ap.add_argument('--epa', type=float, required=True)
    args = vars(ap.parse_args())
    train()
