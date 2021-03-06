import csv
import json
import numpy as np
from svo import SVO
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from epa_expansion.align_wv_space import get_aligned_wv


data_epa_path = '../data/NH_dataset/NewsHeadlines_EPA.csv'
data_epa_svo_pred_path = '../data/NH_dataset/NewsHeadlines_SVO_pred.csv'

data_valence_path = '../data/NH_dataset/NewsHeadlines_Valence.csv'
google_news_model_path = '../models/embedding/GoogleNews-vectors-negative300.bin'
github_model_path = '../models/embedding/github_aligned/word2vec_sg_0_size_300_mincount_20'



def get_word_vector(tokens):
    wv_model = KeyedVectors.load_word2vec_format(google_news_model_path, binary=True)
    wv = list()
    for line in tokens:
        wv_svo = list()
        for w in line:
            if w in wv_model.vocab.keys():
                wv_svo.append(wv_model[w])
            else:
                wv_svo.append(np.zeros(300))
        wv.append(wv_svo)
        # wv.append([wv_model[w] if w in wv_model.vocab.keys() else print(w) for w in line])
    del wv_model
    return wv


def get_comp_word_vector(tokens):
    wv = list()
    model = Word2Vec.load(github_model_path)
    for line in tokens:
        wv_svo = list()
        for w in line:
            if w in model.wv.vocab.keys():
                wv_svo.append(model.wv[w])
            else:
                wv_svo.append(np.zeros(300))
        wv.append(wv_svo)
    return wv


def get_pred_svo(svo, sentence):
    print("======")
    print(sentence)
    root_tree = svo.get_parse_tree(sentence)
    svo_dic = svo.process_parse_tree(next(root_tree))
    print(svo_dic)
    return svo_dic


def read_epa(use_pred):
    # return as np arr
    # #,NewsHeadline,Subject,Verb,Object,E_e,E_p,E_a,S_e,S_p,S_a,V_e,V_p,V_a,O_e,O_p,O_a
    svo = list()
    epa = list()
    svo_pred = load_predict_svo()
    sentences = dict()
    with open(data_epa_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sent = row['NewsHeadline']
            if use_pred and sent in svo_pred.keys() and len(svo_pred[sent]) > 0:
                key = [svo_pred[sent]['subject'][0], svo_pred[sent]['predicate'][0], svo_pred[sent]['object'][0]]
            else:
                key = [row['Subject'], row['Verb'], row['Object']]
            sentences[str(key)] = sent
            svo.append(key)
            event_epa = [row['E_e'], row['E_p'], row['E_a']]
            subject_epa = [row['S_e'], row['S_p'], row['S_a']]
            verb_epa = [row['V_e'], row['V_p'], row['V_a']]
            object_epa = [row['O_e'], row['O_p'], row['O_a']]
            epa.append([event_epa, subject_epa, verb_epa, object_epa])
    with open('../result/state_prediction/news_sent_svo', 'w') as fp:
        json.dump(sentences, fp)
    # svo_wv = np.array(get_word_vector(svo))
    # svo_mask = np.all(np.all(svo_wv, axis=2), axis=1)
    # svo_wv = svo_wv[svo_mask]
    # svo, epa = np.array(svo)[svo_mask], np.array(epa)[svo_mask].astype(np.float)
    # print('svo shape %s' % str(svo.shape))
    # print('svo wv shape %s' % str(svo_wv.shape))
    # print('epa shape %s' % str(epa.shape))
    # return svo, svo_wv, epa


def read_valence():
    # # ,NewsHeadline,Valence
    headline = list()
    valence = list()
    with open(data_valence_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            headline.append(row['NewsHeadline'])
            valence.append(row['Valence'])
    headline, valence = np.array(headline), np.array(valence).astype(np.float)
    print('headline shape %s' % str(headline.shape))
    print('valence shape %s' % str(valence.shape))
    return headline, valence


def predict_svo():
    svo_pred = dict()
    svo_splitter = SVO()
    with open(data_epa_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sent = row['NewsHeadline']
            svo_pred[sent] = get_pred_svo(svo_splitter, sent)
    print(svo_pred)
    with open(data_epa_svo_pred_path, 'w') as fp:
        json.dump(svo_pred, fp)


def load_predict_svo():
    with open(data_epa_svo_pred_path, 'r') as fp:
        pred_svo = json.load(fp)
    return pred_svo


if __name__ == '__main__':
    read_epa(False)
    # read_valence()
    # predict_svo()
