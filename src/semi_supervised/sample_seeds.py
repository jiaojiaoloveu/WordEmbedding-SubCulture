import csv
import random
from labels import WarrinerColumn
from labels import LabelSpace


def rand_eval_wordlist(vocabulary, word_seeds, eval_size):
    eval_poll = []
    for word in vocabulary.keys():
        if word not in word_seeds:
            eval_poll.append(word)
    eval_words = random.sample(eval_poll, eval_size)
    return eval_words


def get_fixed_seeds(vocabulary, eval_size=500):
    word_seeds = ['good', 'nice', 'excellent', 'positive', 'warm', 'correct', 'superior',
                  'bad', 'awful', 'nasty', 'negative', 'cold', 'wrong', 'inferior',
                  'powerful', 'strong', 'potent', 'dominant', 'big', 'forceful', 'hard',
                  'powerless', 'weak', 'impotent', 'small', 'incapable', 'hopeless', 'soft',
                  'active', 'fast', 'noisy', 'lively', 'energetic', 'dynamic', 'quick', 'vital',
                  'quiet', 'clam', 'inactive', 'slow', 'stagnant', 'inoperative', 'passive'
                  ]
    eval_words = rand_eval_wordlist(vocabulary, word_seeds, eval_size)
    return (get_mapping_epa(vocabulary, word_seeds),
            get_mapping_epa(vocabulary, eval_words))


def get_rand_seeds(vocabulary, seed_size=30, eval_size=500, threshold=2.5):
    seeds_poll = []
    for word in vocabulary.keys():
        epa = vocabulary[word]
        for axis in epa.keys():
            if abs(epa[axis]) > threshold:
                seeds_poll.append(word)
                break
    word_seeds = random.sample(seeds_poll, seed_size)
    eval_words = rand_eval_wordlist(vocabulary, word_seeds, eval_size)
    return (get_mapping_epa(vocabulary, word_seeds),
            get_mapping_epa(vocabulary, eval_words))


def get_mapping_epa(vocabulary, word_seeds):
    word_epa = {}
    for word in word_seeds:
        if word in vocabulary.keys():
            word_epa[word] = vocabulary[word]
    return word_epa


def max_min_scaling(x, maxA, minA, maxB, minB):
    return 1.0 * (x - minA) / (maxA - minA) * (maxB - minB) + minB


def scale_vad_to_epa(vocabulary_vad, max_min_board):
    vocabulary_epa = {}
    for word in vocabulary_vad.keys():
        vad = vocabulary_vad[word]
        epa = {}
        for axis in vad.keys():
            epa[LabelSpace.get_epa(axis)] = max_min_scaling(vad[axis],
                                                            max_min_board[axis][WarrinerColumn.Max],
                                                            max_min_board[axis][WarrinerColumn.Min],
                                                            LabelSpace.Max,
                                                            LabelSpace.Min)
        vocabulary_epa[word] = epa
    return vocabulary_epa


def read_warriner_ratings(path):
    # read as vad and scale to epa
    # return as {"word": {'E': 0, 'P': 0, 'A':0}}
    vocabulary_vad = {}
    max_min_board = {
        LabelSpace.V: WarrinerColumn.get_min_max_dic(),
        LabelSpace.A: WarrinerColumn.get_min_max_dic(),
        LabelSpace.D: WarrinerColumn.get_min_max_dic()
    }
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            word = row[WarrinerColumn.Word]
            vad = {LabelSpace.V: float(row[WarrinerColumn.V]),
                   LabelSpace.A: float(row[WarrinerColumn.A]),
                   LabelSpace.D: float(row[WarrinerColumn.D])
                   }
            vocabulary_vad[word] = vad
            for axis in vad.keys():
                if max_min_board[axis][WarrinerColumn.Min] > vad[axis]:
                    max_min_board[axis][WarrinerColumn.Min] = vad[axis]
                if max_min_board[axis][WarrinerColumn.Max] < vad[axis]:
                    max_min_board[axis][WarrinerColumn.Max] = vad[axis]

    return scale_vad_to_epa(vocabulary_vad, max_min_board)


def main():
    csv_path = '../data/epa/Ratings_Warriner_et_al.csv'
    voc = read_warriner_ratings(csv_path)
    sample = get_rand_seeds(voc)
    # sample = get_fixed_seeds(voc)
    return sample


if __name__ == '__main__':
    main()