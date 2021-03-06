# from align_models import align_space
import numpy as np
import copy
import random
import argparse
from nltk.corpus import stopwords
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from scipy import spatial
import json


def __comparison(arr1, arr2):
    diff = arr1 - arr2
    print(np.mean(diff, axis=0))
    print(np.mean(np.abs(diff), axis=0))
    print(np.std(diff, axis=0))


def get_anchor_words():
    anchor = []
    stop_words = stopwords.words('english')
    anchor.extend(stop_words)
    return anchor


def get_training_dataset(source, target, seed_count=20000):
    stop_words = get_anchor_words()
    overlap_words = set(source.vocab.keys()) & set(target.vocab.keys())
    source_mat = []
    target_mat = []
    for word in stop_words:
        if word in overlap_words:
            source_mat.append(source[word])
            target_mat.append(target[word])
    source_mat = np.array(source_mat)
    target_mat = np.array(target_mat)
    # source_mat_rand, target_mat_rand = get_sample_dataset(source, target, k=20000)
    source_mat_rand, target_mat_rand = get_sample_dataset(source, target, k=seed_count)
    source_mat = np.concatenate((source_mat, source_mat_rand))
    target_mat = np.concatenate((target_mat, target_mat_rand))
    print('shape training')
    print(source_mat.shape)
    print(target_mat.shape)
    return source_mat, target_mat


def get_sample_dataset(source, target, k=1000):
    overlap_words = set(source.vocab.keys()) & set(target.vocab.keys())
    sample_overlap_words = random.sample(overlap_words, k)
    source_mat = []
    target_mat = []
    for word in sample_overlap_words:
        source_mat.append(source[word])
        target_mat.append(target[word])
    source_mat = np.array(source_mat)
    target_mat = np.array(target_mat)
    return source_mat, target_mat


def sgd_model():
    model = Sequential()
    model.add(Dense(300, kernel_initializer='normal', input_dim=300))
    # model.add(Dense(300, kernel_initializer='normal', input_dim=300))
    sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model


def cal_cosine_dis(pred, label):
    zipped = zip(pred, label)
    res = []
    for u, v in zipped:
        res.append(spatial.distance.cosine(u, v))
    res = np.array(res)
    print(np.mean(res))
    print(np.std(res))
    return res.tolist()


def __nn_eval(source, target, model):
    source_eval, target_eval = get_sample_dataset(source, target)
    score = model.evaluate(source_eval, target_eval, batch_size=5)
    print(score)
    source_pred = model.predict(source_eval)
    print('distance')
    return cal_cosine_dis(source_pred, target_eval)


def align_nn_model(source, target, seed_count=20000):
    source_mat, target_mat = get_training_dataset(source, target, seed_count)
    print('align train datasize %s' % str(source_mat.shape))
    model = sgd_model()
    model.fit(source_mat, target_mat, epochs=20, batch_size=100)
    score = model.evaluate(source_mat, target_mat, batch_size=5)
    print('align train score')
    print(score)
    print('eval on training dataset')
    source_pred = model.predict(source_mat)
    train = cal_cosine_dis(source_pred, target_mat)
    print('align on testing dataset')
    test = __nn_eval(source, target, model)
    with open('../result/align_space/nn/%s' % seed_count, 'w') as fp:
        json.dump((train, test), fp)
    return model


def __svd_eval(source, target, w):
    source_eval, target_eval = get_sample_dataset(source, target)
    source_pred = np.matmul(source_eval, w)
    return cal_cosine_dis(source_pred, target_eval)


def align_svd_model(source, target, seed_count=20000):
    source_dataset, target_dataset = get_sample_dataset(source, target, k=seed_count)
    product = np.matmul(source_dataset.transpose(), target_dataset)
    U, s, V = np.linalg.svd(product)
    w = np.matmul(U, V)
    print('eval on training dataset')
    source_pred = np.matmul(source_dataset, w)
    train = cal_cosine_dis(source_pred, target_dataset)
    print('eval on testing dataset')
    test = __svd_eval(source, target, w)
    with open('../result/align_space/svd/%s' % seed_count, 'w') as fp:
        json.dump((train, test), fp)
    return w


def get_aligned_wv(source, target, tokens, method='nn', seed_count=20000):
    aligned_wv = {}
    aligned_source_wv = {}
    print('method type %s' % method)
    if method == 'nn':
        model = align_nn_model(source, target, seed_count)
        for word in tokens:
            if word in source.vocab.keys() and word in target.vocab.keys():
                s_wv = model.predict(np.reshape(source[word], (1, 300)))[0]
                t_wv = target[word]
                aligned_wv[word] = np.array([s_wv, t_wv])
        for word in source.vocab.keys():
            s_wv = model.predict(np.reshape(source[word], (1, 300)))[0]
            aligned_source_wv[word] = s_wv
    elif method == 'svd':
        w_mat = align_svd_model(source, target, seed_count)
        for word in tokens:
            if word in source.vocab.keys() and word in target.vocab.keys():
                s_wv = np.matmul(source[word], w_mat)
                t_wv = target[word]
                aligned_wv[word] = np.array([s_wv, t_wv])
        for word in source.vocab.keys():
            s_wv = np.matmul(source[word], w_mat)
            aligned_source_wv[word] = s_wv
    return aligned_wv, aligned_source_wv


# def align_models(source, target):
#     w = align_space(source, target)
#     new_source = copy.deepcopy(source)
#     new_source.wv.vectors = np.matmul(new_source.wv.vectors, w)
#     return new_source, target


if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    ap = argparse.ArgumentParser("align wv space")
    ap.add_argument('--method', type=str, required=True)
    args = vars(ap.parse_args())
    gg_model = KeyedVectors.load_word2vec_format('../models/embedding/GoogleNews-vectors-negative300.bin', binary=True)
    gh_model = Word2Vec.load('../models/embedding/github/word2vec_sg_0_size_300_mincount_5')

    w_list = ['happy', 'sad']

    for seed_count in range(2000, 25000, 2000):
        print('seed %s' % seed_count)
        wv_dict, _ = get_aligned_wv(gh_model.wv, gg_model, w_list, method=args.get('method'), seed_count=seed_count)
        distance = list()
        for w in wv_dict.keys():
            wv = wv_dict[w]
            dis = spatial.distance.cosine(wv[0], wv[1])
            distance.append(dis)
            print('%s: %s' % (w, dis))
        print(np.mean(distance))
        print(np.std(distance))

