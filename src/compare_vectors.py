import copy
import numpy as np
from gensim.models.word2vec import Word2Vec


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == '__main__':
    tw_model_old = Word2Vec.load('../models/embedding/twitter/word2vec_base_300')
    tw_model_new = copy.deepcopy(tw_model_old)

    tw_w = np.load('../models/transform/tw_wk.npy')
    tw_vec = np.matmul(tw_model_old.wv.vectors, tw_w)

    tw_model_new.wv.vectors = tw_vec

    wk_model = Word2Vec.load('../models/embedding/wikitext/word2vec_base_300')

