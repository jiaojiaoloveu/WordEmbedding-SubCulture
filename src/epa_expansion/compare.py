import json

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
