import csv
import json
import random
import os
import numpy as np
from scipy import special
from labels import WarrinerColumn
from labels import LabelSpace


bayesact_dat_folder = '../data/epa/bayesact'
warinner_csv_path = '../data/epa/Ratings_Warriner_et_al.csv'

label_mean = np.array([0.19571685413316248, 0.5452512966060962, -0.6727038433386933])
label_std = np.array([1.5079544406893441, 1.2972247817961575, 1.2448124114055585])


def _fixed_seeds():
    voc = ['good', 'nice', 'excellent', 'positive', 'warm', 'correct', 'superior',
           'bad', 'awful', 'nasty', 'negative', 'cold', 'wrong', 'inferior',
           'powerful', 'strong', 'potent', 'dominant', 'big', 'forceful', 'hard',
           'powerless', 'weak', 'impotent', 'small', 'incapable', 'hopeless', 'soft',
           'active', 'fast', 'noisy', 'lively', 'energetic', 'dynamic', 'quick', 'vital',
           'quiet', 'clam', 'inactive', 'slow', 'stagnant', 'inoperative', 'passive'
           ]
    return voc


def __rand_eval_wordlist(vocabulary, word_seeds, eval_size):
    eval_poll = []
    for word in vocabulary.keys():
        if word not in word_seeds:
            eval_poll.append(word)
    eval_words = random.sample(eval_poll, eval_size)
    return eval_words


def __get_fixed_seeds(vocabulary, eval_size):
    word_seeds = _fixed_seeds()
    eval_words = __rand_eval_wordlist(vocabulary, word_seeds, eval_size)
    return (__get_mapping_epa(vocabulary, word_seeds),
            __get_mapping_epa(vocabulary, eval_words))


def __get_rand_seeds(vocabulary, seed_size, eval_size, threshold):
    seeds_poll = [[], [], []]
    for word in vocabulary.keys():
        epa = vocabulary[word]
        for axis in range(0, 3):
            if abs(epa[axis]) > threshold:
                seeds_poll[axis].append(word)
    word_seeds = []
    print('eval size %s' % seed_size)
    for axis in range(0, 3):
        print('axis %s size %s' % (axis, len(seeds_poll[axis])))
        word_seeds.extend(random.sample(seeds_poll[axis], int(seed_size / 3)))
        print('current size %s' % len(word_seeds))
    word_seeds.extend(_fixed_seeds())
    word_seeds = list(set(word_seeds))
    eval_words = __rand_eval_wordlist(vocabulary, word_seeds, eval_size)
    return (__get_mapping_epa(vocabulary, word_seeds),
            __get_mapping_epa(vocabulary, eval_words))


def __get_mapping_epa(vocabulary, word_seeds):
    return dict((w, vocabulary[w]) for w in set(word_seeds) & vocabulary.keys())


def __scale_vad_to_epa(vocab_vad):
    vocab_epa = {}
    vad = np.array(list(vocab_vad.values()))
    vad_max, vad_min = np.max(vad, axis=0), np.min(vad, axis=0)
    for word in vocab_vad.keys():
        vocab_epa[word] = ((vocab_vad[word] - vad_min) / (vad_max - vad_min) * 4.3 * 2 - 4.3).tolist()
    # for word in vad.keys():
    #     vad = vad[word]
    #     epa = {}
    #     for axis in vad.keys():
    #         epa[LabelSpace.get_epa(axis)] = __max_min_scaling(vad[axis],
    #                                                           max_min_board[axis][WarrinerColumn.Max],
    #                                                           max_min_board[axis][WarrinerColumn.Min],
    #                                                           LabelSpace.Max,
    #                                                           LabelSpace.Min)
    #     vocab_epa[word] = epa
    return vocab_epa


def __norm2uni(x, mu=label_mean, sigma=label_std):
    # map from (-4,4) to (-1,1)
    print('norm2uni called')
    y = (x - mu) / sigma
    return special.erfc(-y / np.sqrt(2)) - 1


def __uni2norm(x, mu=label_mean, sigma=label_std):
    print('uni2norm called')
    x = np.clip(x, -1, 1)
    y = -np.sqrt(2) * special.erfcinv(1 + x)
    return y * sigma + mu


def read_bayesact_epa():
    vocabulary_epa = {}
    for (dirpath, dirnames, filenames) in os.walk(bayesact_dat_folder):
        for names in filenames:
            with open(os.path.join(dirpath, names)) as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    word = row[0]
                    epa = [float(row[1]), float(row[2]), float(row[3])]
                    vocabulary_epa[word] = epa
    return vocabulary_epa


def read_warriner_ratings():
    # read as vad and scale to epa
    # return as {"word": [E, P, A]}
    vocabulary_vad = {}
    with open(warinner_csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            word = row[WarrinerColumn.Word]
            if not word.isalpha():
                continue
            vad = [float(row[WarrinerColumn.V]), float(row[WarrinerColumn.D]), float(row[WarrinerColumn.A])]
            vocabulary_vad[word] = vad
    return __scale_vad_to_epa(vocabulary_vad)


def get_rand_seeds(seed_size=5000, eval_size=8000, threshold=2.0):
    # return (dic, dic)
    # dic = (w, [e,p,a])
    voc = read_warriner_ratings()
    return __get_rand_seeds(voc, seed_size, eval_size, threshold)


def get_fixed_seeds(eval_size=500):
    voc = read_warriner_ratings()
    return __get_fixed_seeds(voc, eval_size)


if __name__ == '__main__':
    # get_rand_seeds()
    # get_fixed_seeds()
    res = read_warriner_ratings()
    new_csv_path = '../data/epa/Ratings_Warriner_et_al_epa_test'
    with open(new_csv_path, 'w') as fp:
        json.dump(res, fp)
    # print(len(read_bayesact_epa().keys()))
