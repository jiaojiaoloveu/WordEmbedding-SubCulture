import json
import os
import argparse
import numpy as np
import sample_seeds
from labels import LabelSpace
from labels import Configs
from gensim.models import KeyedVectors


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
    seed_num = args.get('seed')
    eval_num = args.get('eval')
    threshold = args.get('threshold')

    # seed_words and eval_words as dictionary of word:epa
    (seed_words, eval_words) = sample_seeds.get_rand_seeds(seed_num, eval_num, threshold)
    token_words = set(list(seed_words.keys()) + list(eval_words.keys()))

    with open(os.path.join(word_dataset_base, 'twitter-wordlist'), 'r') as fp:
        corpus_words = set(json.load(fp))
    token_words.update(corpus_words)

    google_news_model_path = '../models/embedding/GoogleNews-vectors-negative300.bin'
    google_news_model = load_word_vectors(google_news_model_path)
    all_token_words = set(google_news_model.vocab.keys())
    token_words = list(token_words & all_token_words)
    token_num = len(token_words)

    print('%s/%s seeds in token words' % (len(set(token_words) & set(seed_words.keys())), seed_num))
    print('%s/%s eval in token words' % (len(set(token_words) & set(eval_words.keys())), eval_num))

    weight_matrix = np.zeros((token_num, token_num), dtype=np.double)

    for ind in range(0, token_num - 1):
        # fully connected graph
        # weight between nodes positive
        # distance = 1 - cosine-dis
        distance_matrix = google_news_model.distances(token_words[ind], token_words[ind + 1:])
        weight_matrix[ind, ind + 1:] = distance_matrix
    del google_news_model

    log_data(token_words, seed_words, eval_words, weight_matrix)
    return token_words, seed_words, eval_words, weight_matrix


def train():
    print('start training')
    token_words, seed_words, eval_words, weight_matrix = reload_data()
    token_num = len(token_words)

    print('calculate matrix')
    weight_matrix = weight_matrix + weight_matrix.transpose()
    degree_matrix = np.sum(weight_matrix, axis=1)
    inverse_degree_matrix = 1 / degree_matrix
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

    eval_mask = np.any(eval_label, axis=1)
    eval_num = np.sum(eval_mask)
    log_window_size = 20
    log_mask = np.random.rand(eval_num) < (1.0 * log_window_size / eval_num)

    mean_absolute_error(-1, eval_label[eval_mask], token_label[eval_mask], log_mask, eval_num)
    original_token_label = np.array(token_label)
    for it in range(0, Configs.iterations):
        print('round %s/%s' % (it, Configs.iterations))
        transient_token_label = np.matmul(laplacian_matrix, token_label)
        token_label = transient_token_label * np.reshape(label_mask_all, (token_num, 1)) + \
                      Configs.alpha * original_token_label
        mean_absolute_error(it, eval_label[eval_mask], token_label[eval_mask], log_mask, eval_num)


if __name__ == '__main__':
    ap = argparse.ArgumentParser("semi-supervised training using graph")
    ap.add_argument('--seed', type=int, required=False, default=1000)
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

    log_path = os.path.join(word_dataset_base, 'log')
    if os.path.exists(log_path):
        os.remove(log_path)
    train()
