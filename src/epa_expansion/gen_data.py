from align_wv_space import get_aligned_wv
from sample_seeds import read_warriner_ratings, csv_path
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
import numpy as np
import os
import json


word_dataset_base = '../result/epa_expansion'
os.makedirs(word_dataset_base, exist_ok=True)


def load_google_word_vectors(model_path):
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
    return word_vectors


def load_github_word_vectors(model_path):
    github_model = Word2Vec.load(model_path)
    return github_model


def get_tokens():
    tokens = []
    # with open(os.path.join(word_dataset_base, 'wikitext-wordlist'), 'r') as fp:
    #     tokens = json.load(fp)
    # print('comparing %s' % len(tokens))
    tokens = ['good', 'nice', 'excellent', 'positive', 'warm', 'correct', 'superior',
              'bad', 'awful', 'nasty', 'negative', 'cold', 'wrong', 'inferior',
              'powerful', 'strong', 'potent', 'dominant', 'big', 'forceful', 'hard',
              'powerless', 'weak', 'impotent', 'small', 'incapable', 'hopeless', 'soft',
              'active', 'fast', 'noisy', 'lively', 'energetic', 'dynamic', 'quick', 'vital',
              'quiet', 'clam', 'inactive', 'slow', 'stagnant', 'inoperative', 'passive'
              ]
    return tokens


def wv_map():
    gg_model = load_google_word_vectors('../models/embedding/GoogleNews-vectors-negative300.bin')
    gh_model = load_github_word_vectors('../models/embedding/github/word2vec_sg_0_size_300_mincount_5')
    print('align wv space')
    tokens = get_tokens()
    dic = get_aligned_wv(gh_model.wv, gg_model, tokens)
    # gh_model, gg_model = align_models(gh_model, gg_model)
    # print('align done')
    # for w in get_tokens():
    #     if w in gg_model.vocab.keys() and w in gh_model.wv.vocab.keys():
    #         gg = gg_model[w]
    #         gh = gh_model.wv[w]
    #         dic[w] = (gg, gh)

    # dic: word -> [wv1, wv2]
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
    return read_warriner_ratings(csv_path)


def get_wv_space():
    google_news_model_path = '../models/embedding/GoogleNews-vectors-negative300.bin'
    model = load_google_word_vectors(google_news_model_path)
    return model


def load_feature_label(suffix):
    feature = np.load(os.path.join(word_dataset_base, 'feature_' + suffix + '.npy'))
    label = np.load(os.path.join(word_dataset_base, 'label_' + suffix + '.npy'))
    return feature, label


def preprocess_data(word_epa_dataset, suffix):
    wv_feature = []
    epa_label = []
    google_model = get_wv_space()
    google_vocab = set(google_model.vocab.keys())

    def epa2list(epa):
        return [epa['E'], epa['P'], epa['A']]

    for word in word_epa_dataset.keys():
        if word not in google_vocab:
            continue
        feature = google_model[word]
        wv_feature.append(feature)

        label = word_epa_dataset[word]
        epa_label.append(epa2list(label))

    wv_feature = np.array(wv_feature)
    epa_label = np.array(epa_label)
    print(wv_feature.shape)
    print(epa_label.shape)
    np.save(os.path.join(word_dataset_base, 'feature_' + suffix), wv_feature)
    np.save(os.path.join(word_dataset_base, 'label_' + suffix), epa_label)
    return wv_feature, epa_label


def generate_data(generate):
    if generate < 2:
        if generate == 0:
            feature, label = load_feature_label('all')
        else:
            feature, label = preprocess_data(load_all(), 'all')
        (items, dimensions) = feature.shape
        mask = np.random.random_sample(items)
        train_test_split = 0.4
        feature_train, label_train = feature[mask < train_test_split], label[mask < train_test_split]
        feature_test, label_test = feature[mask >= train_test_split], label[mask >= train_test_split]
    elif generate < 4:
        if generate == 2:
            feature_train, label_train = load_feature_label('train')
            feature_test, label_test = load_feature_label('test')
        else:
            feature_train, label_train = preprocess_data(load_train(), 'train')
            feature_test, label_test = preprocess_data(load_test(), 'test')
    else:
        print('generate = %s not supported' % generate)
        raise Exception('generate not supported yet')
    return feature_train, label_train, feature_test, label_test
