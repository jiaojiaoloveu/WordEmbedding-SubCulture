# from align_models import align_space
import numpy as np
import copy
import random
from nltk.corpus import stopwords
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from scipy import spatial


def get_training_dataset(source, target):
    stop_words = stopwords.words('english')
    overlap_words = set(source.vocab.keys()) & set(target.vocab.keys())
    source_mat = []
    target_mat = []
    for word in stop_words:
        if word in overlap_words:
            source_mat.append(source[word])
            target_mat.append(target[word])
    source_mat = np.array(source_mat)
    target_mat = np.array(target_mat)
    print('shape training')
    print(source_mat.shape)
    print(target_mat.shape)
    return source_mat, target_mat


def get_eval_dataset(source, target):
    overlap_words = set(source.vocab.keys()) & set(target.vocab.keys())
    sample_overlap_words = random.sample(overlap_words, 1000)
    source_mat = []
    target_mat = []
    for word in sample_overlap_words:
        source_mat.append(source[word])
        target_mat.append(target[word])
    source_mat = np.array(source_mat)
    target_mat = np.array(target_mat)
    print('shape eval')
    print(source_mat.shape)
    print(target_mat.shape)
    return source_mat, target_mat


def sgd_model():
    model = Sequential()
    model.add(Dense(300, kernel_initializer='normal', input_dim=300))
    sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    return model


def cal_cosine_dis(pred, label):
    zipped = zip(pred, label)
    res = []
    for u, v in zipped:
        res.append(spatial.distance.cosine(u, v))
    res = np.array(res)
    print(np.mean(res))
    print(np.std(res))


def align_space(source, target):
    source_mat, target_mat = get_training_dataset(source, target)
    model = sgd_model()
    model.fit(source_mat, target_mat, epochs=100, batch_size=5)
    score = model.evaluate(source_mat, target_mat, batch_size=5)
    print('model train')
    print(score)
    source_pred = model.predict(source_mat)
    cal_cosine_dis(source_pred, target_mat)
    return model


def align_models_nn(source, target):
    model = align_space(source, target)
    source_eval, target_eval = get_eval_dataset(source, target)
    print('model eval')
    score = model.evaluate(source_eval, target_eval, batch_size=5)
    print(score)
    source_pred = model.predict(source_eval)
    cal_cosine_dis(source_pred, target_eval)
    return model


def get_aligned_wv(source, target, tokens):
    aligned_wv = {}
    model = align_models_nn(source, target)
    for word in tokens:
        if word in source.vocab.keys() and word in target.vocab.keys():
            s_wv = model.predict(np.reshape(source[word], (1, 300)))[0]
            t_wv = target[word]
            aligned_wv[word] = np.array([s_wv, t_wv])
            print(aligned_wv[word].shape)
    return aligned_wv


# def align_models(source, target):
#     w = align_space(source, target)
#     new_source = copy.deepcopy(source)
#     new_source.wv.vectors = np.matmul(new_source.wv.vectors, w)
#     return new_source, target


if __name__ == '__main__':
    gg_model = KeyedVectors.load_word2vec_format('../models/embedding/GoogleNews-vectors-negative300.bin', binary=True)
    gh_model = Word2Vec.load('../models/embedding/github/word2vec_sg_0_size_300_mincount_5')
    align_models_nn(gg_model, gh_model.wv)
