from gensim.models.word2vec import Word2Vec
from read_data import CorpusType
import numpy as np
from tempfile import TemporaryFile


# from https://stackoverflow.com/questions/21030391/how-to-normalize-array-numpy
def normalized(a, axis=-1, order=2):
    """Utility function to normalize the rows of a numpy array."""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def get_dictionary(wv1, wv2):
    wv1_set = set(wv1)
    wv2_set = set(wv2)
    return list(wv1_set & wv2_set)


def make_training_datasets(source_model, target_model, overlap_dictionary):
    source_matrix = []
    target_matrix = []
    for word in overlap_dictionary:
        source_matrix.append(source_model.wv[word])
        target_matrix.append(target_model.wv[word])
    return np.array(source_matrix), np.array(target_matrix)


def align_space(source_model, target_model):
    overlap_dictionary = get_dictionary(source_model.wv.vocab.keys(), target_model.wv.vocab.keys())
    source_dataset, target_dataset = make_training_datasets(source_model, target_model, overlap_dictionary)
    product = np.matmul(source_dataset.transpose(), target_dataset)
    U, s, V = np.linalg.svd(product)
    return np.matmul(U, V)


if __name__ == '__main__':
    model_base = '../models/embedding/%s/%s'
    model_name = 'word2vec_base_300'
    tw_model = Word2Vec.load(model_base % (CorpusType.TWITTER, model_name))
    # gh_model = Word2Vec.load(model_base % (CorpusType.GITHUB, model_name))
    wk_model = Word2Vec.load(model_base % (CorpusType.WIKITEXT, model_name))
    w = align_space(tw_model, wk_model)
    outfile = TemporaryFile()
    np.save(outfile, w)
