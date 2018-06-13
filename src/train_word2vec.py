import os
import argparse
from read_data import read_all_wordlist
from corpus_type import CorpusType
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText


def train_word_vectors(modeltype, sentences, path, sg, size, mincount):
    print(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if modeltype == 'word2vec':
        model = Word2Vec(sentences, sg=sg, size=size, min_count=mincount)
        model.save(path)
    elif modeltype == 'fasttest':
        model = FastText(sentences, sg=sg, size=size, min_count=mincount)
        model.save(path)
    else:
        raise Exception("%s is not valid model type" % modeltype)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="train models")
    ap.add_argument("--sg", required=False, type=int, help="use sg(1) or cbow(0)", default=0)
    ap.add_argument("--size", required=False, type=int, help="feature size", default=300)
    ap.add_argument("--mincount", required=False, type=int, help="minimum word frequency", default=5)
    ap.add_argument("--model", required=True, type=str, help="type of model to use(word2vec or fasttest)")
    args = vars(ap.parse_args())
    sg = args.get("sg")
    size = args.get("size")
    mincount = args.get("mincount")
    modeltype = args.get("model")

    corpus_type = CorpusType.GITHUB.value

    token_matrix = read_all_wordlist('../data/%s-wordlist-all' % corpus_type)
    train_word_vectors(modeltype=modeltype,
                       sentences=token_matrix,
                       path=('../models/embedding/%s/%s_sg_%s_size_%s_mincount_%s' %
                             (corpus_type, modeltype, sg, size, mincount)),
                       sg=sg, size=size, mincount=mincount)

