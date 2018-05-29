import os
import re
import pickle
import csv
from enum import Enum
from bs4 import BeautifulSoup
from tinydb import TinyDB
from nltk.tokenize import word_tokenize
from nltk.corpus import twitter_samples
from nltk.twitter.common import json2csv
from langdetect import detect, lang_detect_exception


class CorpusType(Enum):
    GITHUB = 'github'
    WIKIPEDIA = 'wikipedia'
    SEPHORA = 'sephora'
    COHA = 'coha'
    TWITTER = 'twitter'


def clean_and_tokenize(sentence):
    sentence = re.sub("(https?://[^ ]+)", " ", sentence).lower()
    text = BeautifulSoup(sentence, "lxml").get_text()
    text = re.sub("[^a-zA-Z]", " ", text)
    return word_tokenize(text)


def read_single_repo(path):
    word_list = []
    if os.path.isfile(path):
        db = TinyDB(path)
        try:
            for entry in db:
                for value in entry['issues']:
                    if value['title'] is not None and value['body'] is not None:
                        word_list.append(clean_and_tokenize(value['title'] + " " + value['body']))
                    elif value['body'] is not None:
                        word_list.append(clean_and_tokenize(value['body']))
                    elif value['title'] is not None:
                        word_list.append(clean_and_tokenize(value['title']))

                for value in entry['issue_comments']:
                    if value['body'] is not None:
                        word_list.append(clean_and_tokenize(value['body']))

                for value in entry['pull_requests']:
                    if value['title'] is not None and value['body'] is not None:
                        word_list.append(clean_and_tokenize(value['title'] + " " + value['body']))
                    elif value['body'] is not None:
                        word_list.append(clean_and_tokenize(value['body']))
                    elif value['title'] is not None:
                        word_list.append(clean_and_tokenize(value['title']))

                for value in entry['review_comments']:
                    if value['body'] is not None:
                        word_list.append(clean_and_tokenize(value['body']))

                for value in entry['commits']:
                    if value['commit']['message'] is not None:
                        word_list.append(clean_and_tokenize(value['commit']['message']))

                for value in entry['commit_comments']:
                    if value['body'] is not None:
                        word_list.append(clean_and_tokenize(value['body']))
        except:
            print(path)
    return word_list


def read_single_wiki(path):
    word_list = []
    if os.path.isfile(path):
        db = TinyDB(path)
        try:
            for entry in db:
                print("Entry name: %s" % entry['title'])
                word_list.append(clean_and_tokenize(entry["content"]))
        except:
            print(path)
    return word_list


def read_single_coha(path):
    word_list = []
    if os.path.isfile(path):
        with open(path, 'r') as fp:
            context = fp.read()
            word_list.append(clean_and_tokenize(context))
    return word_list


def read_single_sephora(path):
    word_list = []
    if os.path.isfile(path):
        file_type = os.path.basename(path).split('.')[0].split('_')[1]
        with open(path, 'r') as fp:
            reader = csv.DictReader(fp)
            if file_type == 'review':
                for row in reader:
                    word_list.append(clean_and_tokenize(row['review_title'] + ' ' + row['review_text']))
            elif file_type == 'product':
                for row in reader:
                    word_list.append(row['name'] + ' ' + row['detail_text'])
    return word_list


def read_single_tweets(path):
    word_list = []
    if os.path.isfile(path):
        input_tweets = twitter_samples.abspath(os.path.abspath(path))
        output_tweets = os.path.join(os.path.dirname(path) + '_text', os.path.basename(path) + '.csv')
        os.makedirs(os.path.dirname(output_tweets), exist_ok=True)
        with open(input_tweets) as fp:
            json2csv(fp, output_tweets, ['text'])
        with open(output_tweets, 'r') as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                try:
                    tweet = row['text']
                    if detect(tweet) == 'en':
                        word_list.append(clean_and_tokenize(tweet))
                except lang_detect_exception.LangDetectException:
                    continue
    return word_list


def read_all_files(path, corpus_type):
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            print("Processing %s" % root)
            for file in files:
                file_full_path = os.path.join(root, file)
                dump_file_path = os.path.join('../data/%s-wordlist-all' % corpus_type.value,
                                              os.path.relpath(file_full_path, path))
                os.makedirs(os.path.dirname(dump_file_path), exist_ok=True)
                if corpus_type == CorpusType.GITHUB:
                    word_list = read_single_repo(path=file_full_path)
                elif corpus_type == CorpusType.WIKIPEDIA:
                    word_list = read_single_wiki(path=file_full_path)
                elif corpus_type == CorpusType.COHA:
                    word_list = read_single_coha(path=file_full_path)
                elif corpus_type == CorpusType.SEPHORA:
                    word_list = read_single_sephora(path=file_full_path)
                elif corpus_type == CorpusType.TWITTER:
                    word_list = read_single_tweets(path=file_full_path)
                else:
                    raise Exception("wrong file type")
                with open(dump_file_path, 'wb') as fp:
                    pickle.dump(word_list, fp)


def read_all_wordlist(path):
    word_matrix = []
    print('Path %s' % path)
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                file_full_path = os.path.join(root, file)
                print('Parsing %s' % file_full_path)
                with open(file_full_path, 'rb') as fp:
                    word_list = pickle.load(fp)
                    word_matrix.extend(word_list)
    return word_matrix


if __name__ == '__main__':

    global_corpus_type = CorpusType.GITHUB

    corpus_path = {CorpusType.GITHUB: '../data/github',
                   CorpusType.WIKIPEDIA: '../data/wikipedia/content',
                   CorpusType.COHA: '../data/coha',
                   CorpusType.SEPHORA: '../data/sephora',
                   CorpusType.TWITTER: '../data/twitter/part-00'
                   }

    read_all_files(path=corpus_path.get(global_corpus_type), corpus_type=global_corpus_type)
