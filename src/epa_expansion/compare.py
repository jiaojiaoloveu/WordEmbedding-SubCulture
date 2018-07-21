import json
import numpy as np
from nltk.corpus import wordnet as wn
from sample_seeds import read_warriner_ratings
from collections import OrderedDict

tokens = ['male', 'female', 'adult', 'mom', 'father']


def overall_comp():
    common_tokens = set(github.keys()) & set(twitter.keys())
    dis_e, dis_p, dis_a, dis_ave = {}, {}, {}, {}
    for token in common_tokens:
        # diff = list(map(abs, github[token] - twitter[token]))
        diff = np.abs(np.array(github[token]) - np.array(twitter[token])).tolist()
        dis_e[token], dis_p[token], dis_a[token] = diff[0], diff[1], diff[2]
        dis_ave[token] = np.mean(diff).item()
    dis_e = OrderedDict(sorted(dis_e.items(), key=lambda t: t[1], reverse=True))
    dis_p = OrderedDict(sorted(dis_p.items(), key=lambda t: t[1], reverse=True))
    dis_a = OrderedDict(sorted(dis_a.items(), key=lambda t: t[1], reverse=True))
    dis_ave = sorted(dis_ave, key=dis_ave.get, reverse=True)
    with open('../result/epa_expansion/comparison_e', 'w') as fp:
        json.dump(dis_e, fp)
    with open('../result/epa_expansion/comparison_p', 'w') as fp:
        json.dump(dis_p, fp)
    with open('../result/epa_expansion/comparison_a', 'w') as fp:
        json.dump(dis_a, fp)
    with open('../result/epa_expansion/comparison_ave', 'w') as fp:
        json.dump(dis_ave, fp)


def get_tokenset(words_list):
    words_list_pos = {
        wn.VERB: [],
        wn.NOUN: [],
        wn.ADV: [],
        wn.ADJ: []
    }

    for word in words_list:
        for pos in words_list_pos.keys():
            if word in set(w.name().split('.', 1)[0] for w in wn.synsets(word, pos=pos)):
                words_list_pos[pos].append(word)

    return words_list_pos


def tokenset_comp():
    words_list_pos = get_tokenset(set(github.keys()) & set(twitter.keys()))
    for pos in words_list_pos.keys():
        tokenlist = words_list_pos[pos]
        github_list = np.array([github[t] for t in tokenlist])
        twitter_list = np.array([twitter[t] for t in tokenlist])
        print('===== %s =====' % pos)
        print(np.mean(github_list, axis=0))
        print(np.mean(np.abs(github_list), axis=0))
        print(np.std(github_list, axis=0))

        print(np.mean(twitter_list, axis=0))
        print(np.mean(np.abs(twitter_list), axis=0))
        print(np.std(twitter_list, axis=0))


if __name__ == '__main__':
    with open('../result/epa_expansion/nn_result_github_all', 'r') as fp:
        github = json.load(fp)
    with open('../result/epa_expansion/nn_result_twitter_all', 'r') as fp:
        twitter = json.load(fp)
    warriner = read_warriner_ratings()

    # for token in tokens:
    #     print('%s %s %s %s' % (token, github[token], twitter[token], warriner[token]))

    tokenset_comp()


