import json
import os
import argparse
import numpy as np
import time
from labels import LabelSpace
from labels import Configs
from scipy import spatial
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from scipy.stats.stats import pearsonr
from gen_data import word_dataset_base, load_github_word_vectors, load_google_word_vectors
from gen_data import get_tokens
from sample_seeds import __norm2uni, __uni2norm, get_rand_seeds


log_name = 'log_exp%s_enn%s_it%s'


def mean_absolute_error(it, real_label, predict_label, log_mask, eval_num):
    assert real_label.shape == predict_label.shape

    real_label_ori = __uni2norm(real_label)
    predict_label_ori = __uni2norm(predict_label)

    mae = np.sum(np.absolute(real_label - predict_label), axis=0) / eval_num
    mae_ori = np.sum(np.absolute(real_label_ori - predict_label_ori), axis=0) / eval_num

    with open(os.path.join(word_dataset_base, log_name), 'a') as fp:
        out = [
            'iteration #%s/%s' % (it, Configs.iterations),
            'real',
            str(real_label[log_mask]),
            'predict',
            str(predict_label[log_mask]),
            'real_ori',
            str(real_label_ori[log_mask]),
            'predict_ori',
            str(predict_label_ori[log_mask]),
            'mae',
            mae,
            'mae',
            mae_ori,
            'corr',
            str(pearsonr(real_label[:, 0], predict_label[:, 0])),
            str(pearsonr(real_label[:, 1], predict_label[:, 1])),
            str(pearsonr(real_label[:, 2], predict_label[:, 2])),
            'corr_ori',
            str(pearsonr(real_label_ori[:, 0], predict_label_ori[:, 0])),
            str(pearsonr(real_label_ori[:, 1], predict_label_ori[:, 1])),
            str(pearsonr(real_label_ori[:, 2], predict_label_ori[:, 2])),
        ]
        fp.writelines('%s\n' % line for line in out)
    return mae


def log_json(path, arr):
    with open(path, 'w') as fp:
        json.dump(arr, fp)


def log_np(path, arr):
    np.save(path, arr)


def log_data(token_words, comparing_words, seed_words, eval_words, token_label, eval_label, weight_matrix):
    os.makedirs(word_dataset_base, exist_ok=True)
    log_json(os.path.join(word_dataset_base, 'token'), token_words)
    log_json(os.path.join(word_dataset_base, 'compare'), comparing_words)
    log_json(os.path.join(word_dataset_base, 'seed'), seed_words)
    log_json(os.path.join(word_dataset_base, 'eval'), eval_words)
    log_np(os.path.join(word_dataset_base, 'token_label'), token_label)
    log_np(os.path.join(word_dataset_base, 'eval_label'), eval_label)
    log_np(os.path.join(word_dataset_base, 'matrix'), weight_matrix)


def reload_data():
    with open(os.path.join(word_dataset_base, 'token'), 'r') as fp:
        token_words = json.load(fp)
    with open(os.path.join(word_dataset_base, 'compare'), 'r') as fp:
        compare_words = json.load(fp)
    with open(os.path.join(word_dataset_base, 'seed'), 'r') as fp:
        seed_words = json.load(fp)
    with open(os.path.join(word_dataset_base, 'eval'), 'r') as fp:
        eval_words = json.load(fp)
    token_label = np.load(os.path.join(word_dataset_base, 'token_label.npy'))
    eval_label = np.load(os.path.join(word_dataset_base, 'eval_label.npy'))
    weight_matrix = np.load(os.path.join(word_dataset_base, 'matrix.npy'))
    return token_words, compare_words, seed_words, eval_words, token_label, eval_label, weight_matrix


def get_comparing_tokens():
    # dic: str -> np.ndarray
    github_model_path = '../models/embedding/github/word2vec_sg_0_size_300_mincount_5'
    github_model = load_github_word_vectors(github_model_path)
    google_news_model_path = '../models/embedding/GoogleNews-vectors-negative300.bin'
    google_news_model = load_google_word_vectors(google_news_model_path)
    github_voc = {}
    tokens_set = set(get_tokens())
    tokens_list = list(set(google_news_model.vocab.keys()) & set(github_model.wv.vocab.keys()) & tokens_set)
    for token in tokens_list:
        github_voc[token] = github_model.wv[token]
    del github_model
    del google_news_model
    return github_voc


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
    # github words word:wv
    comparing_words = get_comparing_tokens()
    token_words = set(list(seed_words.keys()) + list(eval_words.keys()) + list(comparing_words.keys()))

    # append wikitext words to network
    with open(os.path.join(word_dataset_base, 'wikitext-wordlist'), 'r') as fp:
        corpus_words = set(json.load(fp))
    token_words.update(corpus_words)

    # use trained model to calculate distance
    google_news_model_path = '../models/embedding/GoogleNews-vectors-negative300.bin'
    google_news_model = load_google_word_vectors(google_news_model_path)
    all_token_words = set(google_news_model.vocab.keys())
    token_words = list(token_words & all_token_words)
    sub_token_num = len(token_words)

    # add github words to network
    token_words.extend(list(comparing_words.keys()))
    token_num = len(token_words)

    # objective matrix
    token_label = np.zeros((token_num, LabelSpace.Dimension), dtype=np.double)
    eval_label = np.array(token_label)

    # update label info
    seeds_in_token, eval_in_token = 0, 0
    for ind in range(0, sub_token_num):
        word = token_words[ind]
        if word in seed_words.keys():
            token_label[ind] = [seed_words[word][LabelSpace.E],
                                seed_words[word][LabelSpace.P],
                                seed_words[word][LabelSpace.A]]
            seeds_in_token += 1
        if word in eval_words.keys():
            eval_label[ind] = [eval_words[word][LabelSpace.E],
                               eval_words[word][LabelSpace.P],
                               eval_words[word][LabelSpace.A]]
            eval_in_token += 1

    print('%s/%s seeds in token words' % (seeds_in_token, Configs.seed))
    print('%s/%s eval in token words' % (eval_in_token, Configs.eval))

    print('sub token number %s' % sub_token_num)
    print('token number %s' % token_num)

    time_arr1 = []
    time_arr2 = []

    # update weight info
    weight_matrix = np.zeros((token_num, token_num), dtype=np.double)
    for ind in range(0, sub_token_num - 1):
        # fully connected graph
        # weight between nodes positive
        # distance = 1 - cosine-dis
        time1 = time.time()
        distance_matrix = google_news_model.distances(token_words[ind], token_words[ind + 1: sub_token_num])
        weight_matrix[ind, ind + 1: sub_token_num] = distance_matrix
        time2 = time.time()
        weight_matrix[ind, sub_token_num: token_num] = get_github_distance(
            google_news_model.wv[token_words[ind]],
            token_words[sub_token_num: token_num],
            comparing_words)
        time3 = time.time()
        time_arr1.append((time2 - time1) / (sub_token_num - ind - 1))
        time_arr2.append((time3 - time2) / (token_num - sub_token_num))
    print(np.mean(time_arr1))
    print(np.mean(time_arr2))

    for ind in range(sub_token_num, token_num - 1):
        if ind % 10 == 0:
            print('second half %s / %s' % (ind, token_num))
        weight_matrix[ind, ind + 1: token_num] = get_github_distance(
            comparing_words[token_words[ind]],
            token_words[ind + 1: token_num],
            comparing_words)

    del google_news_model

    log_data(token_words, list(comparing_words.keys()), seed_words, eval_words, token_label, eval_label, weight_matrix)


def train():
    print('start training')
    token_words, compare_words, seed_words, eval_words, token_label, eval_label, weight_matrix = reload_data()

    token_label_mask = np.any(token_label, axis=1)
    token_label_ori = token_label[token_label_mask]

    eval_label_mask = np.any(eval_label, axis=1)
    eval_label_ori = eval_label[eval_label_mask]

    print(token_label[token_label_mask])

    token_label[token_label_mask] = __norm2uni(token_label_ori)
    eval_label[eval_label_mask] = __norm2uni(eval_label_ori)

    token_num = len(token_words)

    print('calculate matrix')
    weight_matrix = weight_matrix + np.transpose(weight_matrix)
    weight_matrix_mask = weight_matrix < Configs.enn
    np.fill_diagonal(weight_matrix_mask, False)
    weight_matrix = np.exp(weight_matrix * -Configs.exp) * weight_matrix_mask
    degree_matrix = np.sum(weight_matrix, axis=1)
    inverse_degree_matrix = np.divide(1, degree_matrix, where=degree_matrix != 0)
    laplacian_matrix = weight_matrix * np.reshape(inverse_degree_matrix, (token_num, 1))

    np.save(os.path.join(word_dataset_base, 'lap'), laplacian_matrix)

    print('generate eval mat')

    print('seed word in token dic %s/%s', (len(set(seed_words.keys()) & set(token_words)), len(seed_words.keys())))
    print('eval word in token dic %s/%s', (len(set(eval_words.keys()) & set(token_words)), len(eval_words.keys())))

    label_mask = np.array(token_label_mask)
    label_mask_inv = np.logical_not(label_mask)
    label_mask_all = (1 - Configs.alpha) * label_mask + label_mask_inv

    eval_mask = np.array(eval_label_mask)
    eval_num = np.sum(eval_mask)
    log_window_size = 20
    log_mask = np.random.rand(eval_num) < (1.0 * log_window_size / eval_num)

    mean_absolute_error(-1, eval_label[eval_mask], token_label[eval_mask], log_mask, eval_num)
    original_token_label = np.array(token_label)
    for it in range(0, Configs.iterations):
        if it % 10 == 0:
            print('round %s/%s' % (it, Configs.iterations))
        transient_token_label = np.matmul(laplacian_matrix, token_label)
        token_label = transient_token_label * np.reshape(label_mask_all, (token_num, 1)) + \
                      Configs.alpha * original_token_label * np.reshape(label_mask, (token_num, 1))
        mean_absolute_error(it, eval_label[eval_mask], token_label[eval_mask], log_mask, eval_num)

    token_label = __uni2norm(token_label)
    np.save(os.path.join(word_dataset_base, 'token_label_pre'), token_label)

    predict_label = list(zip(token_words, token_label.tolist()))
    with open(os.path.join(word_dataset_base, 'result'), 'w') as fp:
        json.dump(predict_label, fp)
    result = {}
    for (word, label) in predict_label:
        if word in compare_words:
            if word in result.keys():
                result[word].append(label)
            else:
                result[word] = [label]
    print(result)


if __name__ == '__main__':
    ap = argparse.ArgumentParser("semi-supervised training using graph")
    ap.add_argument('--seed', type=int, required=False)
    ap.add_argument('--eval', type=int, required=False)
    ap.add_argument('--epa', type=float, required=False)
    ap.add_argument('--generate', type=int, required=False, default=0)
    ap.add_argument('--alpha', type=float, required=False)
    ap.add_argument('--iteration', type=int, required=False)
    ap.add_argument('--enn', type=float, required=False)
    ap.add_argument('--exp', type=float, required=False)
    args = vars(ap.parse_args())
    if args.get("alpha") is not None:
        Configs.alpha = args.get("alpha")

    if args.get("iteration") is not None:
        Configs.iterations = args.get("iteration")

    if args.get('enn') is not None:
        Configs.enn = args.get('enn')

    if args.get('exp') is not None:
        Configs.exp = args.get('exp')

    if args.get('seed') is not None:
        Configs.seed = args.get('seed')

    if args.get('eval') is not None:
        Configs.eval = args.get('eval')

    if args.get('epa') is not None:
        Configs.epa = args.get('epa')

    if args.get("generate") == 0:
        generate()

    log_name = log_name % (Configs.exp, Configs.enn, Configs.iterations)

    log_path = os.path.join(word_dataset_base, log_name)
    if os.path.exists(log_path):
        os.remove(log_path)

    train()
