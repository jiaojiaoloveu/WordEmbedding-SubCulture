from sklearn.svm import SVR
from neural_network import generate_data
from propagate_labels import word_dataset_base
from propagate_labels import load_google_word_vectors, load_github_word_vectors
from align_wv_space import align_models
from time import time
import os
import json
import numpy as np
import argparse


def get_tokens():
    tokens = []
    with open(os.path.join(word_dataset_base, 'wikitext-wordlist'), 'r') as fp:
        tokens = json.load(fp)
    print('comparing %s' % len(tokens))
    return tokens


def wv_map():
    dic = {}
    gg_model = load_google_word_vectors('../models/embedding/GoogleNews-vectors-negative300.bin')
    gh_model = load_github_word_vectors('../models/embedding/github/word2vec_sg_0_size_300_mincount_5')
    print('align wv space')
    gh_model, gg_model = align_models(gh_model, gg_model)
    print('align done')
    for w in get_tokens():
        if w in gg_model.vocab.keys() and w in gh_model.wv.vocab.keys():
            gg = gg_model[w]
            gh = gh_model.wv[w]
            dic[w] = (gg, gh)
    return dic


def train(wv):
    generate = args.get('generate')
    feature_train, label_train, feature_test, label_test = generate_data(generate=generate)
    model = args.get('model')
    if model == 'svr':
        clf = SVR()
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
                gg = wv[w][0]
                gh = wv[w][1]
                label_space.append(clf.predict([gg, gh]))
            label_space = np.array(label_space)
            print(label_space)
            print('time %s' % (time() - start))
            print('mean google radius')
            gg_radius = np.abs(label_space[:, 0])
            print(np.mean(gg_radius))
            print(np.std(gg_radius))
            print('mean github radius')
            gh_radius = np.abs(label_space[:, 1])
            print(np.mean(gh_radius))
            print(np.std(gh_radius))


if __name__ == '__main__':
    ap = argparse.ArgumentParser('keras deep learning method')
    ap.add_argument('--generate', type=int, required=True)
    ap.add_argument('--model', type=str, required=True)
    args = vars(ap.parse_args())
    train(wv_map())
