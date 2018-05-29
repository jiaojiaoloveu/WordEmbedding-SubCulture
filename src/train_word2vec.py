from read_data import read_all_wordlist
from read_data import CorpusType
from gensim.models.word2vec import Word2Vec


def train_word_vectors(sentences, path):
    model = Word2Vec(sentences, size=300)
    model.save(path)
    return model


if __name__ == '__main__':
    corpus_type = CorpusType.WIKIPEDIA.value

    token_matrix = read_all_wordlist('../data/%s-wordlist-all' % corpus_type)
    train_word_vectors(sentences=token_matrix, path=('../models/embedding/%s/word2vec_base_300' % corpus_type))

