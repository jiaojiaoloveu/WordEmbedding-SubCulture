import json
import csv
import numpy as np
from svo import SVO
from read_news_headline import get_pred_svo

github_comments_path = '../data/Github_comments/emotions_pull_request_status_from_mechnical_turk.txt'
github_comments_svo_pred_path = '../data/Github_comments/emotions_pull_request_status_svo_pred.txt'

basic_sentiment = ['aggressive', 'angry', 'calm', 'careless',
                   'cautious', 'defensive', 'happy', 'nervous',
                   'sorry', 'thanks'
                   ]


def read_gh_comments():
    # pull_request_id,id,comment_id,body,
    # Thanks,Sorry,Calm,Nervous,Careless,Cautious,Agressive,Defensive,Happy,Angry,pull_request_status
    with open(github_comments_svo_pred_path, 'r') as fp:
        svo_pred = json.load(fp)
    svo = list()
    senti_vec = list()
    with open(github_comments_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sent = row['body']
            if sent in svo_pred.keys() and len(svo_pred[sent]) > 0:
                for item in svo_pred[sent]:
                    if len(item) > 0:
                        svo.append([item['subject'][0].lower(), item['predicate'][0].lower(), item['object'][0].lower()])
                        senti_vec.append([row[sent.capitalize()] for sent in basic_sentiment])
    # svo = np.array(svo)
    # senti_vec = np.array(senti_vec)
    # print(svo.shape)
    # print(senti_vec.shape)
    # print(svo)
    return svo, senti_vec


def pred_github_svo():
    svo_pred = dict()
    svo_splitter = SVO()
    with open(github_comments_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            svo_lists = list()
            sentence = row['body']
            sentences = svo_splitter.sentence_split(sentence)
            for sent in sentences:
                svo_lists.append(get_pred_svo(svo_splitter, sent))
            svo_pred[sentence] = svo_lists
    print(svo_pred)
    with open(github_comments_svo_pred_path, 'w') as fp:
        json.dump(svo_pred, fp)


if __name__ == '__main__':
    # pred_github_svo()
    svo, senti = read_gh_comments()

