from enum import Enum


class CorpusType(Enum):
    GITHUB = 'github'
    WIKIPEDIA = 'wikipedia'
    SEPHORA = 'sephora'
    COHA = 'coha'
    TWITTER = 'twitter'
    WIKITEXT = 'wikitext'
    ONEBILLION = '1-billion'