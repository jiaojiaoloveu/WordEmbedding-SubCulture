from align_wv_space import get_aligned_wv
from sample_seeds import read_warriner_ratings, read_bayesact_epa, get_rand_seeds
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from sample_seeds import __norm2uni
import numpy as np
import os
import json
import random
import argparse


word_dataset_base = '../result/epa_expansion'
os.makedirs(word_dataset_base, exist_ok=True)
google_model_path = '../models/embedding/GoogleNews-vectors-negative300.bin'
compare_model_path = '../models/embedding/%s/fasttext_sg_0_size_300_mincount_5'


def load_google_word_vectors(model_path):
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
    return word_vectors


def load_github_word_vectors(model_path):
    github_model = Word2Vec.load(model_path)
    return github_model


def get_tokens():
    tokens = ['good', 'nice', 'excellent', 'positive', 'warm', 'correct', 'superior',
              'bad', 'awful', 'nasty', 'negative', 'cold', 'wrong', 'inferior',
              'powerful', 'strong', 'potent', 'dominant', 'big', 'forceful', 'hard',
              'powerless', 'weak', 'impotent', 'small', 'incapable', 'hopeless', 'soft',
              'active', 'fast', 'noisy', 'lively', 'energetic', 'dynamic', 'quick', 'vital',
              'quiet', 'clam', 'inactive', 'slow', 'stagnant', 'inoperative', 'passive'
              ]
    return tokens


def get_rand_tokens():
    gg_model = load_google_word_vectors(google_model_path)
    tokens = list(random.sample(gg_model.vocab.keys(), 50))
    del gg_model
    return tokens


def get_token_wv(tokens):
    gg_model = load_google_word_vectors(google_model_path)
    wv = []
    for token in tokens:
        if token in gg_model.vocab.keys():
            wv.append(gg_model[token])
    del gg_model
    return np.array(wv)


def wv_map(method, culture):
    gg_model = load_google_word_vectors(google_model_path)
    gh_model = load_github_word_vectors(compare_model_path % culture)
    print('align wv space')
    tokens = get_tokens()
    # pred_tokens = list(set(gg_model.vocab.keys()) & set(gh_model.wv.vocab.keys()))
    dic, s_dic = get_aligned_wv(gh_model.wv, gg_model, tokens, method)
    # gh_model, gg_model = align_models(gh_model, gg_model)
    # print('align done')
    # for w in get_tokens():
    #     if w in gg_model.vocab.keys() and w in gh_model.wv.vocab.keys():
    #         gg = gg_model[w]
    #         gh = gh_model.wv[w]
    #         dic[w] = (gh, gg)

    # dic: word -> [wv1, wv2]

    del gg_model, gh_model
    return dic, s_dic, wv_map_epa(list(dic.keys()))


def wv_map_epa(tokens):
    word_epa_dataset = load_all()
    dic = {}
    for word in tokens:
        if word in word_epa_dataset:
            dic[word] = word_epa_dataset[word]
        else:
            dic[word] = [0, 0, 0]
    return dic


def load_train():
    with open(os.path.join(word_dataset_base, 'seed'), 'r') as fp:
        seed_words = json.load(fp)
    return seed_words


def load_test():
    with open(os.path.join(word_dataset_base, 'eval'), 'r') as fp:
        eval_words = json.load(fp)
    return eval_words


def load_all():
    return read_warriner_ratings()
    # return read_bayesact_epa()


def load_feature_label(suffix):
    feature = np.load(os.path.join(word_dataset_base, 'feature_' + suffix + '.npy'))
    label = np.load(os.path.join(word_dataset_base, 'label_' + suffix + '.npy'))
    return feature, label


def preprocess_data(word_epa_dataset, suffix):
    wv_feature = []
    epa_label = []
    google_model = load_google_word_vectors(google_model_path)
    google_vocab = set(google_model.vocab.keys())

    for word in word_epa_dataset.keys():
        if word not in google_vocab:
            continue
        feature = google_model[word]
        wv_feature.append(feature)

        label = word_epa_dataset[word]
        epa_label.append(label)

    wv_feature = np.array(wv_feature)
    epa_label = np.array(epa_label)
    print(wv_feature.shape)
    print(epa_label.shape)
    np.save(os.path.join(word_dataset_base, 'feature_' + suffix), wv_feature)
    np.save(os.path.join(word_dataset_base, 'label_' + suffix), epa_label)
    del google_model
    return wv_feature, epa_label


def generate_data(generate):
    if generate < 2:
        # random sample from datasets
        if generate == 0:
            feature, label = load_feature_label('all')
        else:
            feature, label = preprocess_data(load_all(), 'all')
        (items, dimensions) = feature.shape
        mask = np.random.random_sample(items)
        train_test_split = 0.7
        feature_train, label_train = feature[mask < train_test_split], label[mask < train_test_split]
        feature_test, label_test = feature[mask >= train_test_split], label[mask >= train_test_split]
    elif generate < 4:
        # select items with large EPA values as training
        if generate == 2:
            feature_train, label_train = load_feature_label('train')
            feature_test, label_test = load_feature_label('test')
        else:
            (seed_words, eval_words) = get_rand_seeds(seed_size=9000, eval_size=4000, threshold=2)
            feature_train, label_train = preprocess_data(seed_words, 'train')
            feature_test, label_test = preprocess_data(eval_words, 'test')
    else:
        print('generate = %s not supported' % generate)
        raise Exception('generate not supported yet')
    return feature_train, label_train, feature_test, label_test


if __name__ == '__main__':
    ap = argparse.ArgumentParser('wv map')
    ap.add_argument('--method', type=str, required=True)
    args = vars(ap.parse_args())
    wv_map(args.get('method'), culture='github')
