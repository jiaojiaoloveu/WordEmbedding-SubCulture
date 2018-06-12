import os
import argparse
from read_data import read_all_wordlist
from corpus_type import CorpusType
from gensim.models.word2vec import Word2Vec


def train_word_vectors(sentences, path, sg, size, mincount):
    print(os.path.dirname(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model = Word2Vec(sentences, sg=sg, size=size, min_count=mincount)
    model.save(path)
    return model


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="train models")
    ap.add_argument("--sg", required=False, type=int, help="use sg(1) or cbow(0)", default=0)
    ap.add_argument("--size", required=False, type=int, help="feature size", default=300)
    ap.add_argument("--mincount", required=False, type=int, help="minimum word frequency", default=300)
    args = vars(ap.parse_args())
    sg = args.get("sg")
    size = args.get("size")
    mincount = args.get("mincount")

    corpus_type = CorpusType.GITHUB.value

    token_matrix = read_all_wordlist('../data/%s-wordlist-all' % corpus_type)
    train_word_vectors(sentences=token_matrix, path=('../models/embedding/%s/word2vec_sg_%s_size_%s' %
                                                     (corpus_type, sg, size)), sg=sg, size=size, mincount=mincount)

