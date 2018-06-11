from nltk.corpus import wordnet as wn
from gensim.models.word2vec import Word2Vec
from corpus_type import CorpusType
import pickle
import os

if __name__ == '__main__':
    base_model_path = '../models/embedding/%s/word2vec_base_300'
    wk_model = Word2Vec.load(base_model_path % CorpusType.WIKITEXT.value)
    tw_model = Word2Vec.load(base_model_path % CorpusType.TWITTER.value)
    gh_model = Word2Vec.load(base_model_path % CorpusType.GITHUB.value)
    words_list = list(set(wk_model.wv.vocab.keys()) & set(tw_model.wv.vocab.keys()) & set(gh_model.wv.vocab.keys()))

    base_wordlist_path = '../data/wk_tw_gh_wordlist'
    os.makedirs(base_wordlist_path, exist_ok=True)
    with open(os.path.join(base_wordlist_path, 'all'), 'wb') as fp:
        pickle.dump(words_list, fp)

    words_list_pos = {
        wn.VERB: [],
        wn.NOUN: [],
        wn.ADV: [],
        wn.ADJ: []
    }

    for word in words_list:
        for pos in words_list_pos.keys():
            if word in set(w.name().split('.', 1)[0] for w in wn.synsets(word, pos=pos)):
                words_list_pos[pos].append(word)

    for pos in words_list_pos.keys():
        with open(os.path.join(base_wordlist_path, pos), 'wb') as fp:
            pickle.dump(words_list_pos[pos], fp)
