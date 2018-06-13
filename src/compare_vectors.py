import copy
import numpy as np
import json
import os
import argparse
from gensim.models.word2vec import Word2Vec
from corpus_type import CorpusType
from collections import OrderedDict


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)).item()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="compare models")
    ap.add_argument("--model", required=True, type=str)
    args = vars(ap.parse_args())
    model_name = args.get("model")
    model_base_path = '../models/embedding/%s/%s'
    tw_model_old = Word2Vec.load(model_base_path % (CorpusType.TWITTER.value, model_name))
    tw_w = np.load('../models/transform/%s/tw_wk.npy' % model_name)
    tw_model = copy.deepcopy(tw_model_old)
    tw_model.wv.vectors = np.matmul(tw_model_old.wv.vectors, tw_w)

    gh_model_old = Word2Vec.load(model_base_path % (CorpusType.GITHUB.value, model_name))
    gh_w = np.load('../models/transform/%s/gh_wk.npy' % model_name)
    gh_model = copy.deepcopy(gh_model_old)
    gh_model.wv.vectors = np.matmul(gh_model_old.wv.vectors, gh_w)

    wk_model = Word2Vec.load(model_base_path % (CorpusType.WIKITEXT.value, model_name))

    pos_list = {'all', 'a', 'n', 'r', 'v', 'all'}

    tw_cmp_base_path = '../result/tw_wk/%s' % model_name
    os.makedirs(tw_cmp_base_path, exist_ok=True)
    gh_cmp_base_path = '../result/gh_wk/%s' % model_name
    os.makedirs(gh_cmp_base_path, exist_ok=True)
    al_cmp_base_path = '../result/tw_gh_wk/%s' % model_name
    os.makedirs(al_cmp_base_path, exist_ok=True)

    for pos in pos_list:
        with open('../result/wk_tw_gh_wordlist/%s/%s' % (model_name, pos), 'r') as fp:
            wordlist_noun = json.load(fp)

        tw_distance = {}
        gh_distance = {}
        al_distance = {}
        for noun in wordlist_noun:
            tw_new_dis = cosine(tw_model.wv[noun], wk_model.wv[noun])
            tw_old_dis = cosine(tw_model_old[noun], wk_model.wv[noun])
            tw_distance[noun] = [tw_new_dis, tw_old_dis]

            gh_new_dis = cosine(gh_model.wv[noun], wk_model.wv[noun])
            gh_old_dis = cosine(gh_model_old[noun], wk_model.wv[noun])
            gh_distance[noun] = [gh_new_dis, gh_old_dis]

            al_distance[noun] = [tw_new_dis, gh_new_dis]

        tw_distance_ordered = OrderedDict(sorted(tw_distance.items(), key=lambda it: it[1][0], reverse=True))
        gh_distance_ordered = OrderedDict(sorted(gh_distance.items(), key=lambda it: it[1][0], reverse=True))
        al_distance_ordered = OrderedDict(sorted(al_distance.items(),
                                                 key=lambda it: abs(it[1][0] - it[1][1]), reverse=True))

        with open(os.path.join(tw_cmp_base_path, pos), 'w') as fp:
            json.dump(tw_distance_ordered, fp)

        with open(os.path.join(gh_cmp_base_path, pos), 'w') as fp:
            json.dump(gh_distance_ordered, fp)

        with open(os.path.join(al_cmp_base_path, pos), 'w') as fp:
            json.dump(al_distance_ordered, fp)
