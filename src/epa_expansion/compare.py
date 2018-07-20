import json
import numpy as np

tokens = ['male', 'female', 'adult', 'mom', 'father']


if __name__ == '__main__':
    with open('../result/epa_expansion/nn_result_github_all', 'r') as fp:
        github = json.load(fp)
    with open('../result/epa_expansion/nn_result_twitter_all', 'r') as fp:
        twitter = json.load(fp)
    with open('../data/epa/Ratings_Warriner_et_al_epa', 'r') as fp:
        warriner = json.load(fp)
    for token in tokens:
        print('%s %s %s %s' % (token, github[token], twitter[token], warriner[token]))

    common_tokens = set(github.keys()) & set(twitter.keys())
    dis_e, dis_p, dis_a, dis_ave = {}, {}, {}, {}
    for token in common_tokens:
        # diff = list(map(abs, github[token] - twitter[token]))
        diff = np.abs(np.array(github[token]) - np.array(twitter[token])).tolist()
        dis_e[token], dis_p[token], dis_a[token] = diff[0], diff[1], diff[2]
        dis_ave[token] = np.mean(diff).item()
    dis_e = sorted(dis_e, key=dis_e.get, reverse=True)
    dis_p = sorted(dis_p, key=dis_p.get, reverse=True)
    dis_a = sorted(dis_a, key=dis_a.get, reverse=True)
    dis_ave = sorted(dis_ave, key=dis_ave.get, reverse=True)
    with open('../result/epa_expansion/comparison_e') as fp:
        json.dump(dis_e, fp)
    with open('../result/epa_expansion/comparison_p') as fp:
        json.dump(dis_p, fp)
    with open('../result/epa_expansion/comparison_a') as fp:
        json.dump(dis_a, fp)
    with open('../result/epa_expansion/comparison_ave') as fp:
        json.dump(dis_ave, fp)
