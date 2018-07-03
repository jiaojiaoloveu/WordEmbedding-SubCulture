# from align_models import align_space
import numpy as np
import copy
from nltk.corpus import stopwords
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from scipy import spatial


def get_dataset(source, target):
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
    print('shape')
    print(source_mat.shape)
    print(target_mat.shape)
    return source_mat, target_mat


def sgd_model():
    model = Sequential()
    model.add(Dense(300, kernel_initializer='normal', input_dim=300))
    sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    return model


def align_space(source, target):
    source_mat, target_mat = get_dataset(source, target)
    model = sgd_model()
    model.fit(source_mat, target_mat, epochs=100, batch_size=5)
    source_pred = model.predict(source_mat)
    aligned_list = zip(source_pred, target_mat)
    res = []
    for u, v in aligned_list:
        res.append(spatial.distance.cosine(u, v))
    res = np.array(res)
    print('result')
    print(res)
    print(np.mean(res))
    print(np.std(res))
    return model


def align_models_nn(source, target):
    model = align_space(source, target)
    return model


def align_models(source, target):
    w = align_space(source, target)
    new_source = copy.deepcopy(source)
    new_source.wv.vectors = np.matmul(new_source.wv.vectors, w)
    return new_source, target


if __name__ == '__main__':
    gg_model = KeyedVectors.load_word2vec_format('../models/embedding/GoogleNews-vectors-negative300.bin', binary=True)
    gh_model = Word2Vec.load('../models/embedding/github/word2vec_sg_0_size_300_mincount_5')
    align_models_nn(gg_model, gh_model.wv)
