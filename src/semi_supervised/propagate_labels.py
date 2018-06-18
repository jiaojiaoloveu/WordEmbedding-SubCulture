import json
import os
import random
import argparse
import numpy as np
import sample_seeds
from labels import LabelSpace
from labels import Configs
from gensim.models import KeyedVectors


def load_word_vectors(model_path):
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
    return word_vectors


def mean_absolute_error(real_label, predict_label):
    print(real_label.shape)
    print(predict_label.shape)
    assert real_label.shape == predict_label.shape
    (token_size, feature_size) = real_label.shape

    mask = np.any(real_label, axis=1)
    error_mat = (real_label - predict_label) * np.reshape(mask, (token_size, 1))
    mae = np.sum(error_mat, axis=0) / np.sum(mask)
    print(mae)
    return mae


def log_data(token_words, seed_words, eval_words):
    word_dataset_base = '../result/semi-supervised'
    os.makedirs(word_dataset_base, exist_ok=True)
    with open(os.path.join(word_dataset_base, 'token'), 'w') as fp:
        json.dump(token_words, fp)
    with open(os.path.join(word_dataset_base, 'seed'), 'w') as fp:
        json.dump(seed_words, fp)
    with open(os.path.join(word_dataset_base, 'eval'), 'w') as fp:
        json.dump(eval_words, fp)


def main():
    ap = argparse.ArgumentParser("semi-supervised training using graph")
    ap.add_argument('--train', type=int, required=False, default=5000)
    ap.add_argument('--seed', type=int, required=False, default=50)
    ap.add_argument('--eval', type=int, required=False, default=500)
    ap.add_argument('--threshold', type=float, required=False, default=2.5)
    args = vars(ap.parse_args())
    token_num = args.get('train')
    seed_num = args.get('seed')
    eval_num = args.get('eval')
    threshold = args.get('threshold')

    google_news_model_path = '../models/embedding/GoogleNews-vectors-negative300.bin'
    google_news_model = load_word_vectors(google_news_model_path)

    all_token_words = list(google_news_model.vocab.keys())
    all_alphabets_token = [token for token in all_token_words if token.isalpha()]
    token_words = set(random.sample(all_alphabets_token, token_num))

    (seed_words, eval_words) = sample_seeds.get_rand_seeds(seed_num, eval_num, threshold)

    for token in seed_words.keys():
        if token in all_token_words:
            token_words.add(token)
        else:
            print('%s not in 3B' % token)
    for token in eval_words.keys():
        if token in all_token_words:
            token_words.add(token)
        else:
            print('%s not in 3B' % token)

    token_words = list(token_words)
    token_num = len(token_words)

    log_data(token_words, seed_words, eval_words)

    weight_matrix = np.zeros((token_num, token_num), dtype=np.double)

    for ind in range(0, token_num - 1):
        # fully connected graph
        # weight between nodes positive
        # distance = 1 - cosine-dis
        weight_matrix[ind, ind + 1:] = 2 - google_news_model.distances(token_words[ind], token_words[ind + 1:])

    del google_news_model

    weight_matrix = weight_matrix + weight_matrix.transpose()
    degree_matrix = np.eye(token_num) * np.sum(weight_matrix, axis=1)
    inverse_degree_matrix = np.linalg.inv(degree_matrix)
    laplacian_matrix = np.matmul(inverse_degree_matrix, weight_matrix)

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

    original_token_label = np.array(token_label)
    for it in range(0, Configs.iterations):
        print('iteration # %s / %s' % (it, Configs.iterations))
        transient_token_label = np.matmul(laplacian_matrix, token_label)
        token_label = Configs.alpha * transient_token_label + (1 - Configs.alpha) * original_token_label
        mean_absolute_error(eval_label, token_label)


if __name__ == '__main__':
    main()
