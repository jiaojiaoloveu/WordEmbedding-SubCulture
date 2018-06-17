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


def main():
    google_news_model_path = '../models/embedding/GoogleNews-vectors-negative300.bin'
    google_news_model = load_word_vectors(google_news_model_path)

    (token_num, feature_size) = google_news_model.vectors.shape
    weight_matrix = np.zeros((token_num, token_num), dtype=np.double)

    token_list = list(google_news_model.vocab.keys())

    for ind in range(0, len(token_list) - 1):
        # fully connected graph
        # weight between nodes positive
        # distance = 1 - cosine-dis
        weight_matrix[ind, ind + 1:] = 2 - google_news_model.distances(token_list[ind], token_list[ind + 1])
    weight_matrix = weight_matrix + weight_matrix.transpose()
    degree_matrix = np.eye(token_num) * np.sum(weight_matrix, axis=1)
    inverse_degree_matrix = np.linalg.inv(degree_matrix)
    laplacian_matrix = np.matmul(inverse_degree_matrix, weight_matrix)

    token_label = np.zeros((token_num, feature_size), dtype=np.double)
    eval_label = np.array(token_label)

    (seed_words, eval_words) = sample_seeds.main()
    for ind in range(0, len(token_list)):
        word = token_list[ind]
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
