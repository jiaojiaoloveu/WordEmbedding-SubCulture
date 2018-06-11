import copy
import numpy as np
import json
from gensim.models.word2vec import Word2Vec
from corpus_type import CorpusType


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == '__main__':
    model_base_path = '../models/embedding/%s/word2vec_base_300'
    tw_model_old = Word2Vec.load(model_base_path % CorpusType.TWITTER.value)
    tw_w = np.load('../models/transform/tw_wk.npy')
    tw_model = copy.deepcopy(tw_model_old)
    tw_model.wv.vectors = np.matmul(tw_model_old.wv.vectors, tw_w)
    wk_model = Word2Vec.load(model_base_path % CorpusType.WIKITEXT.value)

    with open('../result/wk_tw_gh_wordlist/n', 'r') as fp:
        wordlist_noun = json.load(fp)

    distance = {}
    for noun in wordlist_noun:
        new_dis = cosine(tw_model.wv[noun], wk_model.wv[noun])
        old_dis = cosine(tw_model_old[noun], wk_model.wv[noun])
        distance[noun] = [new_dis, old_dis]

    with open('../result/tw_wk', 'w') as fp:
        json.dump(distance, fp)
