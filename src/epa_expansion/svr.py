from sklearn.svm import SVR
from neural_network import generate_data
from sample_seeds import __uni2norm, __norm2uni
import os
import json
import numpy as np
import argparse


word_dataset_base = '../result/epa_expansion/svr'
os.makedirs(word_dataset_base, exist_ok=True)


def train(seed_size, eval_size, epa, uniform, epsilon=0.05):
    feature_train, label_train, feature_test, label_test = generate_data(2, seed_size, eval_size, epa)

    clf = SVR(kernel='rbf', epsilon=epsilon, C=10)
    if uniform:
        label_train = __norm2uni(label_train)
        label_test = __norm2uni(label_test)
    label_pred = []
    for axis in range(0, 3):
        label_train_axis = label_train[:, axis]
        print('start training')
        clf.fit(feature_train, label_train_axis)
        label_pred.append(clf.predict(feature_test))
    label_pred = np.transpose(label_pred)
    mae = np.mean(np.abs(label_pred - label_test), axis=0)
    rsme = np.sqrt(np.mean((label_pred - label_test) ** 2, axis=0))
    if uniform:
        label_test = __uni2norm(label_test)
        label_pred = __uni2norm(label_pred)
        mae_ori = np.mean(np.abs(label_pred - label_test), axis=0)
        rsme_ori = np.sqrt(np.mean((label_pred - label_test) ** 2, axis=0))
        return np.array([mae, rsme, mae_ori, rsme_ori]).tolist()
    else:
        return np.array([mae, rsme]).tolist()
    #     label_space = []
    #     for w in wv:
    #         label_space.append(clf.predict(wv[w]))
    #     label_space = np.array(label_space)
    #     print('label space shape')
    #     print(label_space.shape)
    #     gh_label_pred.append(label_space[:, 0])
    #     gg_label_pred.append(label_space[:, 1])
    # gh_label_pred = np.transpose(gh_label_pred)
    # gg_label_pred = np.transpose(gg_label_pred)
    # print('label pred shape')
    # print(gh_label_pred.shape)
    # print(gh_label_pred)
    # print(gg_label_pred.shape)
    # print(gg_label_pred)
    # if uniform:
    #     gh_label_pred = __uni2norm(gh_label_pred)
    #     gg_label_pred = __uni2norm(gg_label_pred)
    # print('after uni2norm')
    # print(gh_label_pred)
    # print(gg_label_pred)
    # print('google')
    # print(np.mean(gg_label_pred, axis=0))
    # print(np.mean(np.abs(gg_label_pred), axis=0))
    # print(np.std(gg_label_pred, axis=0))
    # print('github')
    # print(np.mean(gh_label_pred, axis=0))
    # print(np.mean(np.abs(gh_label_pred), axis=0))
    # print(np.std(gh_label_pred, axis=0))


def main():
    seed_size = args.get('seed')
    eval_size = args.get('eval')
    epa = args.get('epa')
    uniform = (args.get('uniform') == 1)
    logging = []

    # for uniform in [True, False]:
    #     for epa in range(30, -1, -5):
    #         metrics = train(600, 1000, 0.1 * epa, uniform)
    #         logging.append({
    #             'seed': 600,
    #             'eval': 1000,
    #             'epa': 0.1 * epa,
    #             'uniform': int(uniform),
    #             'mae': metrics
    #         })
    #     for seed in range(500, 5001, 500):
    #         metrics = train(seed, 8000, 2.0, uniform)
    #         logging.append({
    #             'seed': seed,
    #             'eval': 8000,
    #             'epa': 2,
    #             'uniform': int(uniform),
    #             'mae': metrics
    #         })
    # for epsilon in [0.01, 0.05, 0.1, 0.2, 0.5]:
    #     metrics = train(8500, 1000, 1.0, False, epsilon)
    #     logging.append({
    #         'epsilon': epsilon,
    #         'metrics': metrics
    #     })
    # with open(os.path.join(word_dataset_base, 'result_grid_search_seed_8500_eval_1000_epa_1.0'), 'w') as fp:
    #     json.dump(logging, fp)

    # for uni in [False, True]:
    #     for seed in range(8500, 499, -1000):
    #         metrics = train(seed, 1000, 1.0, uni, 0.05)
    #         logging.append({
    #             'uniform': uni,
    #             'seed': seed,
    #             'mae': metrics
    #         })
    # with open(os.path.join(word_dataset_base, 'result_seed_uni'), 'w') as fp:
    #     json.dump(logging, fp)

    for uni in [False, True]:
        for epa in range(30, -1, -5):
            metrics = train(600, 1000, 0.1 * epa, uni, 0.05)
            logging.append({
                'uniform': uni,
                'epa': 0.1 * epa,
                'mae': metrics
            })
    with open(os.path.join(word_dataset_base, 'result_epa_uni'), 'w') as fp:
        json.dump(logging, fp)


if __name__ == '__main__':
    ap = argparse.ArgumentParser('keras deep learning method')
    ap.add_argument('--generate', type=int, required=False)
    ap.add_argument('--uniform', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--eval', type=int, required=True)
    ap.add_argument('--epa', type=float, required=True)
    args = vars(ap.parse_args())

    main()

