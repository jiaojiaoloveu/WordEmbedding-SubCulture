import json
import os
import random
import argparse
import numpy as np
import sample_seeds
from labels import LabelSpace
from labels import Configs
from gensim.models import KeyedVectors
from nltk.corpus import wordnet as wn


word_dataset_base = '../result/semi-supervised'


def load_word_vectors(model_path):
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
    return word_vectors


def mean_absolute_error(it, real_label, predict_label, log_mask, eval_num):
    assert real_label.shape == predict_label.shape
    mae = np.sum(np.absolute(real_label - predict_label), axis=0) / eval_num

    with open(os.path.join(word_dataset_base, 'log'), 'a') as fp:
        out = [
            'iteration #%s/%s' % (it, Configs.iterations),
            'real',
            str(real_label[log_mask]),
            'predict',
            str(predict_label[log_mask]),
            'mae',
            mae
        ]
        fp.writelines('%s\n' % line for line in out)
    return mae


def log_data(token_words, seed_words, eval_words, weight_matrix):
    os.makedirs(word_dataset_base, exist_ok=True)
    with open(os.path.join(word_dataset_base, 'token'), 'w') as fp:
        json.dump(token_words, fp)
    with open(os.path.join(word_dataset_base, 'seed'), 'w') as fp:
        json.dump(seed_words, fp)
    with open(os.path.join(word_dataset_base, 'eval'), 'w') as fp:
        json.dump(eval_words, fp)
    np.save(os.path.join(word_dataset_base, 'matrix'), weight_matrix)


def reload_data():
    with open(os.path.join(word_dataset_base, 'token'), 'r') as fp:
        token_words = json.load(fp)
    with open(os.path.join(word_dataset_base, 'seed'), 'r') as fp:
        seed_words = json.load(fp)
    with open(os.path.join(word_dataset_base, 'eval'), 'r') as fp:
        eval_words = json.load(fp)
    weight_matrix = np.load(os.path.join(word_dataset_base, 'matrix.npy'))
    return token_words, seed_words, eval_words, weight_matrix


def generate():
    token_num = args.get('train')
    seed_num = args.get('seed')
    eval_num = args.get('eval')
    threshold = args.get('threshold')

    # seed_words and eval_words as dictionary of word:epa
    (seed_words, eval_words) = sample_seeds.get_rand_seeds(seed_num, eval_num, threshold)

    # get token_num tokens from epa wordset synsets
    token_words = set()
    token_words_buff = set(list(seed_words.keys()) + list(eval_words.keys()))
    while len(token_words) + len(token_words_buff) < token_num:
        token_words_syn = list()
        for token in token_words_buff:
            for syn in wn.synsets(token):
                token_words_syn.extend(syn.lemma_names())
        token_words.update(token_words_buff)
        token_words_buff = set(token_words_syn)
    token_words.update(token_words_buff)

    # get token_num tokens from whole word vector space at random
    google_news_model_path = '../models/embedding/GoogleNews-vectors-negative300.bin'
    google_news_model = load_word_vectors(google_news_model_path)

    all_token_words = set(google_news_model.vocab.keys())
    all_alphabets_token = [token for token in all_token_words if token.isalpha()]
    token_words_rand = set(random.sample(all_alphabets_token, token_num))

    # join together as a list
    # and make sure tokens are defined in high space
    token_words.update(token_words_rand)
    token_words = list(token_words & all_token_words)
    token_num = len(token_words)

    weight_matrix = np.zeros((token_num, token_num), dtype=np.double)

    for ind in range(0, token_num - 1):
        # fully connected graph
        # weight between nodes positive
        # distance = 1 - cosine-dis
        weight_matrix[ind, ind + 1:] = 1 - google_news_model.distances(token_words[ind], token_words[ind + 1:])
    del google_news_model

    log_data(token_words, seed_words, eval_words, weight_matrix)
    return token_words, seed_words, eval_words, weight_matrix


def train():
    print('start training')
    token_words, seed_words, eval_words, weight_matrix = reload_data()
    token_num = len(token_words)

    print('calculate matrix')
    print('1')
    weight_matrix = weight_matrix + weight_matrix.transpose()
    print('2')
    degree_matrix = np.sum(weight_matrix, axis=1)
    print('3')
    inverse_degree_matrix = 1 / degree_matrix
    print('4')
    laplacian_matrix = weight_matrix * np.reshape(inverse_degree_matrix, (token_num, 1))

    print('generate eval mat')
    token_label = np.zeros((token_num, LabelSpace.Dimension), dtype=np.double)
    eval_label = np.array(token_label)

    for ind in range(0, token_num):
        word = token_words[ind]
        if word in seed_words.keys():
            token_label[ind] = [seed_words[word][LabelSpace.E],
                                seed_words[word][LabelSpace.P],
                                seed_words[word][LabelSpace.A]]
        if word in eval_words.keys():
            eval_label[ind] = [eval_words[word][LabelSpace.E],
                               eval_words[word][LabelSpace.P],
                               eval_words[word][LabelSpace.A]]

    label_mask = np.any(token_label, axis=1)
    label_mask_inv = np.logical_not(label_mask)
    label_mask_all = (1 - Configs.alpha) * label_mask + label_mask_inv

    eval_mask = np.any(eval_label, axis = 1)
    eval_num = np.sum(eval_mask)
    log_window_size = 20
    log_mask = np.random.rand(eval_num) < (1.0 * log_window_size / eval_num)
    print(log_mask)

    mean_absolute_error(-1, eval_label, token_label, log_mask, eval_num)
    original_token_label = np.array(token_label)
    for it in range(0, Configs.iterations):
        print('round %s/%s' % (it, Configs.iterations))
        transient_token_label = np.matmul(laplacian_matrix, token_label)
        token_label = transient_token_label * np.reshape(label_mask_all, (token_num, 1)) + \
                      Configs.alpha * original_token_label
        mean_absolute_error(it, eval_label[eval_mask], token_label[eval_mask], log_mask, eval_num)


if __name__ == '__main__':
    ap = argparse.ArgumentParser("semi-supervised training using graph")
    ap.add_argument('--train', type=int, required=False, default=5000)
    ap.add_argument('--seed', type=int, required=False, default=50)
    ap.add_argument('--eval', type=int, required=False, default=500)
    ap.add_argument('--threshold', type=float, required=False, default=2.5)
    ap.add_argument('--generate', type=int, required=False, default=0)
    ap.add_argument('--alpha', type=float, required=False)
    ap.add_argument('--iteration', type=int, required=False)
    args = vars(ap.parse_args())
    if args.get("alpha") is not None:
        Configs.alpha = args.get("alpha")
    if args.get("iteration") is not None:
        Configs.iterations = args.get("iteration")

    if args.get("generate") == 0:
        generate()
    train()
