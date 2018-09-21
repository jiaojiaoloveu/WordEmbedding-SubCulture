import numpy as np
import json


basic_sentiment = ['aggressive', 'angry', 'calm', 'careless',
                   'cautious', 'defensive', 'happy', 'nervous',
                   'sorry', 'thanks'
                   ]


base_svo_path = '../result/state_prediction/svo_%s'
base_epa_path = '../result/epa_expansion/nn_result_%s_all'


def get_senti_epa(name):
    with open(base_epa_path % name, 'r') as fp:
        epa = json.load(fp)
    senti_epa = {}
    for senti in basic_sentiment:
        if senti in epa.keys():
            senti_epa[senti] = np.array(epa[senti])
    print(senti_epa)
    return senti_epa


def get_closest_senti(senti_epa, epa):
    # return list of tuples [('senti': score), ...]
    senti_dis = {}
    for senti in senti_epa:
        distance = np.linalg.norm(senti_epa[senti] - epa)
        senti_dis[senti] = distance
    return sorted(senti_dis.items(), key=lambda kv: kv[1])


def main():
    all = {'epa': 'google', 'general': 'google',
           'github': 'github', 'twitter': 'twitter'}
    for svo_type in all.keys():
        epa_type = all[svo_type]
        print('%s %s' % (svo_type, epa_type))
        with open(base_svo_path % svo_type, 'r') as fp:
            svo_epa = dict((str(item[0]), item[1]) for item in json.load(fp))
        senti_epa = get_senti_epa(name=epa_type)
        svo_senti = {}
        for svo in svo_epa:
            svo_senti[svo] = []
            for epa in svo_epa[svo]:
                svo_senti[svo].append(get_closest_senti(senti_epa, np.array(epa)))
        with open('../result/state_prediction/svo_senti_%s' % svo_type, 'w') as fp:
            json.dump(svo_senti, fp)


def main2():
    svo_epa_sent = {}
    with open('../result/state_prediction/github_comment', 'r') as fp:
        svo_epa_senti = dict((str(item[0]), [item[1]]) for item in json.load(fp))


if __name__ == '__main__':
    main()

