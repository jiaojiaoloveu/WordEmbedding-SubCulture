import json

tokens = ['male', 'female', 'adult', 'mom', 'father']


if __name__ == '__main__':
    with open('../../result/epa_expasion/nn_result_github_all', 'w') as fp:
        github = json.load(fp)
    with open('../../result/epa_expasion/nn_result_twitter_all', 'w') as fp:
        twitter = json.load(fp)
    for token in tokens:
        print('%s %s %s' % (token, github[token], twitter[token]))
