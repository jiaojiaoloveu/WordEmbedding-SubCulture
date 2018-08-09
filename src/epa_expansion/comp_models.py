from collections import OrderedDict
from scipy import spatial
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from align_wv_space import align_nn_model, align_svd_model
import os
import json
import copy
import numpy as np


def align(source, target, method):
    if method == 'nn':
        model = align_nn_model(source, target, 10000)
        return model.predict(source.vectors)
    elif method == 'svd':
        w = align_svd_model(source, target, 10000)
        return np.matmul(source.vectors, w)


def main():
    gg_model = KeyedVectors.load_word2vec_format('../models/embedding/GoogleNews-vectors-negative300.bin', binary=True)
    model_name = 'word2vec_sg_0_size_300_mincount_20'
    gh_model = Word2Vec.load('../models/embedding/github/%s' % model_name)
    gh_vec_new = align(gh_model.wv, gg_model, 'svd')
    gh_model_new = copy.deepcopy(gh_model)
    gh_model_new.wv.vectors = gh_vec_new

    w_list = list(set(gh_model_new.wv.vocab.keys()) & set(gg_model.vocab.keys()))

    distance = dict()
    for w in w_list:
        dis = spatial.distance.cosine(gh_model_new.wv[w], gg_model[w])
        distance[w] = dis
    distance_ordered = OrderedDict(sorted(distance.items(), key=lambda it: it[1], reverse=True))
    cmp_path = '../result/cmp'
    os.makedirs(cmp_path, exist_ok=True)
    with open(os.path.join(cmp_path, 'github'), 'w') as fp:
        json.dump(distance_ordered, fp)
    model_path = '../models/embedding/github_aligned'
    os.makedirs(model_path, exist_ok=True)
    gh_model_new.save(os.path.join(model_path, model_name))
    del gg_model
    del gh_model
    del gh_model_new


def cmp():
    gg_model = KeyedVectors.load_word2vec_format('../models/embedding/GoogleNews-vectors-negative300.bin', binary=True)
    model_name = 'word2vec_sg_0_size_300_mincount_20'
    gh_model = Word2Vec.load('../models/embedding/github_aligned/%s' % model_name)
    tokens = ['bug', 'debug', 'commit', 'push', 'pull', 'thread', 'crash', 'fix', 'git', 'merge', 'password',
              'happy', 'glad', 'sad', 'sorry', 'regret', 'reject', 'angry', 'proud', 'cautious', 'excited',
              'man', 'woman', 'male', 'female', 'boy', 'girl', 'programmer', 'sex', 'gay', 'civic']
    for w in tokens:
        print('==========')
        print(w)
        print(spatial.distance.cosine(gg_model[w], gh_model.wv[w]))
        print('google')
        print(gg_model.most_similar(w, topn=20))
        print('github')
        print(gh_model.wv.most_similar(w, topn=20))


if __name__ == '__main__':
    cmp()
