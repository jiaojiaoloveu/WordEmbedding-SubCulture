from enum import Enum
import os
from gensim.models.fasttext import FastText
from gensim.models.word2vec import Word2Vec


class CorpusType(Enum):
    GITHUB = 'github'
    WIKIPEDIA = 'wikipedia'
    SEPHORA = 'sephora'
    COHA = 'coha'
    TWITTER = 'twitter'
    WIKITEXT = 'wikitext'
    ONEBILLION = '1-billion'


def decide_model_type(model_name=""):
    return model_name.split("_", 1)[0]


def load_model(model_path):
    model_type = decide_model_type(os.path.basename(model_path))
    if model_type == 'word2vec':
        model = Word2Vec.load(model_path)
    elif model_type == 'fasttext':
        model = FastText.load(model_path)
    else:
        model = []
    return model
