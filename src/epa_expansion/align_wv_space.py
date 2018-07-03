# from align_models import align_space
import numpy as np
import copy
from nltk.corpus import stopwords
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec


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
    return source_mat, target_mat


def sgd_model():
    model = Sequential()
    model.add(Dense(300, kernel_initializer='uniform', input_dim=300))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model


def align_space(source, target):
    source_mat, target_mat = get_dataset(source, target)
    model = sgd_model()
    model.fit(source_mat, target_mat, epochs=10, batch_size=128)
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
