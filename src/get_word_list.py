from nltk.corpus import wordnet as wn
from utils import CorpusType
from utils import load_model
import json
import os
import argparse

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="get word list")
    ap.add_argument('--model', type=str, required=True)
    args = vars(ap.parse_args())
    model_name = args.get("model")
    base_model_path = '../models/embedding/%s/%s'
    wk_model = load_model(base_model_path % (CorpusType.WIKITEXT.value, model_name))
    tw_model = load_model(base_model_path % (CorpusType.TWITTER.value, model_name))
    gh_model = load_model(base_model_path % (CorpusType.GITHUB.value, model_name))

    words_list = list(set(wk_model.wv.vocab.keys()) & set(tw_model.wv.vocab.keys()) & set(gh_model.wv.vocab.keys()))

    base_wordlist_path = '../result/wk_tw_gh_wordlist/%s' % model_name
    os.makedirs(base_wordlist_path, exist_ok=True)
    with open(os.path.join(base_wordlist_path, 'all'), 'w') as fp:
        json.dump(words_list, fp)

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
        with open(os.path.join(base_wordlist_path, pos), 'w') as fp:
            json.dump(words_list_pos[pos], fp)
