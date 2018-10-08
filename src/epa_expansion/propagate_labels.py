import json
import os
import argparse
import numpy as np
import time
import csv
import random
from labels import LabelSpace
from labels import Configs
from scipy import spatial
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from scipy.stats.stats import pearsonr
from sample_seeds import __norm2uni, __uni2norm, get_rand_seeds


word_dataset_base = '../result/epa_expansion/graph'
os.makedirs(word_dataset_base, exist_ok=True)


def load_google_word_vectors(model_path):
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
    return word_vectors


def load_github_word_vectors(model_path):
    github_model = Word2Vec.load(model_path)
    return github_model


def log_json(path, arr):
    with open(path, 'w') as fp:
        json.dump(arr, fp)


def log_np(path, arr):
    np.save(path, arr)

# def log_data(token_words, comparing_words, seed_words, eval_words, token_label, eval_label, weight_matrix):
#     os.makedirs(word_dataset_base, exist_ok=True)
#     log_json(os.path.join(word_dataset_base, 'token'), token_words)
#     log_json(os.path.join(word_dataset_base, 'compare'), comparing_words)
#     log_json(os.path.join(word_dataset_base, 'seed'), seed_words)
#     log_json(os.path.join(word_dataset_base, 'eval'), eval_words)
#     log_np(os.path.join(word_dataset_base, 'token_label'), token_label)
#     log_np(os.path.join(word_dataset_base, 'eval_label'), eval_label)
#     log_np(os.path.join(word_dataset_base, 'matrix'), weight_matrix)
#
#
# def reload_data():
#     with open(os.path.join(word_dataset_base, 'token'), 'r') as fp:
#         token_words = json.load(fp)
#     with open(os.path.join(word_dataset_base, 'compare'), 'r') as fp:
#         compare_words = json.load(fp)
#     with open(os.path.join(word_dataset_base, 'seed'), 'r') as fp:
#         seed_words = json.load(fp)
#     with open(os.path.join(word_dataset_base, 'eval'), 'r') as fp:
#         eval_words = json.load(fp)
#     token_label = np.load(os.path.join(word_dataset_base, 'token_label.npy'))
#     eval_label = np.load(os.path.join(word_dataset_base, 'eval_label.npy'))
#     weight_matrix = np.load(os.path.join(word_dataset_base, 'matrix.npy'))
#     return token_words, compare_words, seed_words, eval_words, token_label, eval_label, weight_matrix


def get_github_distance(w1, wlist, wvocab):
    distance_list = []
#     for w2 in wlist:
#         distance_list.append(1 - spatial.distance.cosine(w1, wvocab[w2]))
#     return distance_list
    w2 = np.array([wvocab[w] for w in wlist])
    norm = np.linalg.norm(w1)
    all_norms = np.linalg.norm(w2, axis=1)
    dot_products = np.dot(w2, w1)
    distances = 1 - dot_products / (norm * all_norms)
    return distances


def generate():
    # seed_words and eval_words as dictionary of word:epa
    (seed_words, eval_words) = get_rand_seeds(Configs.seed, Configs.eval, Configs.epa)

    token_words = set(list(seed_words.keys()) + list(eval_words.keys()))

    # append wikitext words to enlarge the network size
    with open('../result/epa_expansion/wikitext-wordlist', 'r') as fp:
        corpus_words = set(json.load(fp))
    token_words.update(corpus_words)

    # use trained model to calculate distance
    google_news_model_path = '../models/embedding/GoogleNews-vectors-negative300.bin'
    google_news_model = load_google_word_vectors(google_news_model_path)
    # use token word in the wv space
    all_token_words = set(google_news_model.vocab.keys())
    token_words = list(token_words & all_token_words)
    token_num = len(token_words)

    # training matrix
    train_label = np.zeros((token_num, LabelSpace.Dimension), dtype=np.double)
    # eval matrix
    eval_label = np.array(train_label)

    # update label info
    seeds_in_token, eval_in_token = 0, 0
    for ind in range(0, token_num):
        word = token_words[ind]
        if word in seed_words.keys():
            train_label[ind] = seed_words[word]
            seeds_in_token += 1
        if word in eval_words.keys():
            eval_label[ind] = eval_words[word]
            eval_in_token += 1

    print('%s/%s seeds in token words' % (seeds_in_token, Configs.seed))
    print('%s/%s eval in token words' % (eval_in_token, Configs.eval))
    print('token number %s' % token_num)

    start_time = time.time()
    # update weight info
    weight_matrix = np.zeros((token_num, token_num), dtype=np.double)
    for ind in range(0, token_num - 1):
        # fully connected graph
        # weight between nodes positive
        # distance = 1 - cosine-dis
        distance_matrix = google_news_model.distances(token_words[ind], token_words[ind + 1: token_num])
        weight_matrix[ind, ind + 1: token_num] = distance_matrix
    del google_news_model

    print('time cost %s' % (time.time() - start_time))

    file_path = os.path.join(word_dataset_base, 'seed_%s_eval_%s_epa_%s' % (Configs.seed, Configs.eval, Configs.epa))
    os.makedirs(file_path, exist_ok=True)

    log_json(os.path.join(file_path, 'token'), token_words)
    log_json(os.path.join(file_path, 'seed'), seed_words)
    log_json(os.path.join(file_path, 'eval'), eval_words)
    log_np(os.path.join(file_path, 'train_label'), train_label)
    log_np(os.path.join(file_path, 'eval_label'), eval_label)
    log_np(os.path.join(file_path, 'matrix'), weight_matrix)


def generate2():
    github_label = {}
    with open('../data/GitHub_Aggregated.csv') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            concept = row['Concept']
            e, p, a = float(row['Evaluation_mean']), float(row['Potency_mean']), float(row['Activity_mean'])
            github_label[concept] = [round(d, 3) for d in [e, p, a]]
    word_seeds = list(github_label.keys())
    word_seeds_train = random.sample(word_seeds, int(0.7 * len(word_seeds)))
    words_seeds_test = [w for w in word_seeds if w not in word_seeds_train]
    seed_words = {k: github_label[k] for k in word_seeds_train}
    eval_words = {k: github_label[k] for k in words_seeds_test}

    Configs.seed = len(word_seeds_train)
    Configs.eval = len(words_seeds_test)

    token_words = set(list(seed_words.keys()) + list(eval_words.keys()))

    # append wikitext words to enlarge the network size
    with open('../result/epa_expansion/wikitext-wordlist', 'r') as fp:
        corpus_words = set(json.load(fp))
    token_words.update(corpus_words)

    # use trained model to calculate distance
    github_model_path = '../models/embedding/github_aligned/word2vec_sg_0_size_300_mincount_5'
    github_model = load_github_word_vectors(github_model_path)
    all_token_words = set(github_model.wv.vocab.keys())
    token_words = list(token_words & all_token_words)
    token_num = len(token_words)

    # training matrix
    train_label = np.zeros((token_num, LabelSpace.Dimension), dtype=np.double)
    # eval matrix
    eval_label = np.array(train_label)

    # update label info
    seeds_in_token, eval_in_token = 0, 0
    for ind in range(0, token_num):
        word = token_words[ind]
        if word in seed_words.keys():
            train_label[ind] = seed_words[word]
            seeds_in_token += 1
        if word in eval_words.keys():
            eval_label[ind] = eval_words[word]
            eval_in_token += 1

    print('%s/%s seeds in token words' % (seeds_in_token, Configs.seed))
    print('%s/%s eval in token words' % (eval_in_token, Configs.eval))
    print('token number %s' % token_num)

    start_time = time.time()
    # update weight info
    weight_matrix = np.zeros((token_num, token_num), dtype=np.double)
    for ind in range(0, token_num - 1):
        # fully connected graph
        # weight between nodes positive
        # distance = 1 - cosine-dis
        distance_matrix = github_model.wv.distances(token_words[ind], token_words[ind + 1: token_num])
        weight_matrix[ind, ind + 1: token_num] = distance_matrix
    del github_model

    print('time cost %s' % (time.time() - start_time))

    file_path = os.path.join(word_dataset_base, 'github_seed_%s_eval_%s_epa_%s' % (Configs.seed, Configs.eval, Configs.epa))
    os.makedirs(file_path, exist_ok=True)

    log_json(os.path.join(file_path, 'token'), token_words)
    log_json(os.path.join(file_path, 'seed'), seed_words)
    log_json(os.path.join(file_path, 'eval'), eval_words)
    log_np(os.path.join(file_path, 'train_label'), train_label)
    log_np(os.path.join(file_path, 'eval_label'), eval_label)
    log_np(os.path.join(file_path, 'matrix'), weight_matrix)


def generate_github():
    aligned_github_model_path = '../models/embedding/github_aligned/word2vec_sg_0_size_300_mincount_5'
    github_model = load_github_word_vectors(aligned_github_model_path)
    github_token_words = list(github_model.wv.vocab.keys())
    github_token_num = len(github_token_words)

    start_time = time.time()
    # update weight info
    github_weight_matrix = np.zeros((github_token_num, github_token_num), dtype=np.double)
    for ind in range(0, github_token_num - 1):
        distance_matrix = github_model.wv.distances(github_token_words[ind], github_token_words[ind + 1: github_token_num])
        github_weight_matrix[ind, ind + 1: github_token_num] = distance_matrix
    print('time cost %s' % (time.time() - start_time))
    del github_model

    log_json(os.path.join(word_dataset_base, 'gh_token'), github_token_words)
    log_np(os.path.join(word_dataset_base, 'gh_matrix'), github_weight_matrix)


def train():
    print('start training')

    file_path = os.path.join(word_dataset_base, 'github_seed_%s_eval_%s_epa_%s' % (Configs.seed, Configs.eval, Configs.epa))
    with open(os.path.join(file_path, 'token'), 'r') as fp:
        token_words = json.load(fp)
    train_label = np.load(os.path.join(file_path, 'train_label.npy'))
    eval_label = np.load(os.path.join(file_path, 'eval_label.npy'))
    weight_matrix = np.load(os.path.join(file_path, 'matrix.npy'))

    train_label_mask = np.any(train_label, axis=1)
    eval_label_mask = np.any(eval_label, axis=1)

    # uniform labels
    if Configs.uni:
        train_label[train_label_mask] = __norm2uni(train_label[train_label_mask])
        eval_label[eval_label_mask] = __norm2uni(eval_label[eval_label_mask])

    token_num = len(token_words)

    print('calculate matrix')
    weight_matrix = weight_matrix + np.transpose(weight_matrix)
    weight_matrix_mask = weight_matrix < Configs.enn
    np.fill_diagonal(weight_matrix_mask, False)
    weight_matrix = np.exp(weight_matrix * -Configs.exp) * weight_matrix_mask
    degree_matrix = np.sum(weight_matrix, axis=1)
    inverse_degree_matrix = np.divide(1, degree_matrix, where=degree_matrix != 0)
    laplacian_matrix = weight_matrix * np.reshape(inverse_degree_matrix, (token_num, 1))
    # save lap mat to local
    np.save(os.path.join(word_dataset_base, 'lap'), laplacian_matrix)

    print('generate eval mat')
    label_mask = np.array(train_label_mask)
    label_mask_inv = np.logical_not(label_mask)
    label_mask_all = (1 - Configs.alpha) * label_mask + label_mask_inv

    def log_item(it, pred, eval):
        return {
            'it': it,
            'mae': np.mean(np.abs(pred - eval), axis=0).tolist(),
            'rsme': np.sqrt(np.mean((pred - eval) ** 2, axis=0)).tolist(),
            'corr': [pearsonr(pred[:, 0], eval[:, 0]),
                     pearsonr(pred[:, 1], eval[:, 1]),
                     pearsonr(pred[:, 2], eval[:, 2])
                     ]
        }

    # logging_info = list()
    # logging_info.append(log_item(-1, eval_label[eval_label_mask], train_label[eval_label_mask]))

    original_train_label = np.array(train_label)

    for it in range(0, Configs.iterations):
        if it % 5 == 0:
            print('round %s/%s' % (it, Configs.iterations))
        transient_token_label = np.matmul(laplacian_matrix, train_label)
        train_label = transient_token_label * np.reshape(label_mask_all, (token_num, 1)) + \
                      Configs.alpha * original_train_label * np.reshape(label_mask, (token_num, 1))
        # logging_info.append(log_item(it, eval_label[eval_label_mask], train_label[eval_label_mask]))

    if Configs.uni:
        train_label_mask_new = np.any(train_label, axis=1)
        train_label[train_label_mask_new] = __uni2norm(train_label[train_label_mask_new])
        eval_label[eval_label_mask] = __uni2norm(eval_label[eval_label_mask])
        # logging_info.append(log_item(Configs.iterations + 1,
        #                              eval_label[eval_label_mask], train_label[eval_label_mask]))

    return log_item(Configs.iterations, eval_label[eval_label_mask], train_label[eval_label_mask])

    # result_file_path = os.path.join(file_path,
    #                                 'it_%s_alpha_%s_enn_%s_exp_%s_uni_%s' %
    #                                 (Configs.iterations, Configs.alpha, Configs.enn, Configs.exp, int(Configs.uni))
    #                                 )
    # os.makedirs(result_file_path, exist_ok=True)

    # log_json(logging_info, os.path.join(result_file_path, 'log'))
    # log_json(os.path.join(result_file_path, 'lexicon'),
    #          list(zip(token_words, train_label.tolist())))
    # np.save(os.path.join(result_file_path, 'train_label_expanded'), train_label)


# def predict():
#     with open(os.path.join(word_dataset_base, 'token'), 'r') as fp:
#         token_words = json.load(fp)
#     with open(os.path.join(word_dataset_base, 'seed'), 'r') as fp:
#         seed_words = json.load(fp)
#     with open(os.path.join(word_dataset_base, 'eval'), 'r') as fp:
#         eval_words = json.load(fp)
#     with open(os.path.join(word_dataset_base, 'gh_token'), 'r') as fp:
#         github_token_words = json.load(fp)
#     train_label = np.load(os.path.join(word_dataset_base, 'train_label.npy'))
#     train_label_2 = np.load(os.path.join(word_dataset_base, 'train_label_2.npy'))
#     eval_label = np.load(os.path.join(word_dataset_base, 'eval_label.npy'))
#     weight_matrix = np.load(os.path.join(word_dataset_base, 'matrix.npy'))
#     github_weight_matrix = np.load(os.path.join(word_dataset_base, 'gh_matrix.npy'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser("semi-supervised training using graph")
    ap.add_argument('--generate', type=int, required=True)

    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--eval', type=int, required=True)
    ap.add_argument('--epa', type=float, required=True)

    ap.add_argument('--exp', type=float, required=True)
    ap.add_argument('--enn', type=float, required=True)

    ap.add_argument('--iteration', type=int, required=True)

    ap.add_argument('--alpha', type=float, required=True)

    ap.add_argument('--uni', type=int, required=True)

    args = vars(ap.parse_args())

    Configs.alpha = args.get("alpha")
    Configs.iterations = args.get("iteration")
    Configs.enn = args.get('enn')
    Configs.exp = args.get('exp')
    Configs.seed = args.get('seed')
    Configs.eval = args.get('eval')
    Configs.epa = args.get('epa')
    Configs.uni = (args.get('uni') == 1)

    mae = train()
    print(mae)


    # logging = []
    # for enn in [0.4, 0.5, 0.6, 0.7, 0.8]:
    #     for exp in [0.5, 1, 2]:
    #         Configs.enn = enn
    #         Configs.exp = exp
    #         metrics = train()
    #         logging.append({
    #             'enn': enn,
    #             'exp': exp,
    #             'metrics': metrics
    #         })

    # with open(os.path.join(word_dataset_base, 'result_grid_search_seed_8500_eval_1000_epa_1.0'), 'w') as fp:
    #     json.dump(logging, fp)

    # for alpha in [0.2, 0.5, 0.8, 1]:
    #     Configs.alpha = alpha
    #     metrics = train()
    #     logging.append({
    #         'alpha': alpha,
    #         'metrics': metrics
    #     })

    # with open(os.path.join(word_dataset_base, 'result_alpha_seed_8500_eval_1000_epa_1.0'), 'w') as fp:
    #     json.dump(logging, fp)

    # for it in [10, 30, 50, 100, 200]:
    #     Configs.iterations = it
    #     metrics = train()
    #     logging.append({
    #         'it': it,
    #         'metrics': metrics
    #     })
    # with open(os.path.join(word_dataset_base, 'result_iteration_seed_8500_eval_1000_epa_1.0'), 'w') as fp:
    #     json.dump(logging, fp)

    # for uni in [False, True]:
    #     Configs.uni = uni
    #     for seed in range(8500, 499, -1000):
    #         Configs.seed = seed
    #         metrics = train()
    #         logging.append({
    #             'seed': seed,
    #             'uni': uni,
    #             'metrics': metrics
    #         })
    # with open(os.path.join(word_dataset_base, 'result_seed_uni'), 'w') as fp:
    #     json.dump(logging, fp)

    # for epa in range(30, -1, -5):
    #     Configs.seed = 600
    #     Configs.epa = epa * 0.1
    #     Configs.eval = 1000
    #     generate()

    # for uni in [False, True]:
    #     Configs.uni = uni
    #     for epa in range(30, -1, -5):
    #         Configs.epa = 0.1 * epa
    #         Configs.seed = 600
    #         metrics = train()
    #         logging.append({
    #             'uniform': uni,
    #             'epa': 0.1 * epa,
    #             'mae': metrics
    #         })
    # with open(os.path.join(word_dataset_base, 'result_epa_uni'), 'w') as fp:
    #    json.dump(logging, fp)


    # if args.get("generate") == 1:
    #     # for epa in range(30, -1, -5):
    #     #     Configs.seed = 600
    #     #     Configs.epa = epa * 0.1
    #     #     Configs.eval = 1000
    #         # generate()
    #         # train()

    #     for seed in range(8500, 499, -1000):
    #         Configs.epa = 1.0
    #         Configs.seed = seed
    #         Configs.eval = 1000
    #         generate()
            # train()

    # train()
