from enum import Enum


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
