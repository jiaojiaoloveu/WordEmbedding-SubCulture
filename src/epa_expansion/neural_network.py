import json
import os
import time
import argparse
import numpy as np
from sample_seeds import __uni2norm
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Activation, Dropout, Embedding
from keras.layers import GlobalMaxPooling1D, MaxPooling1D
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
from gen_data import wv_map, generate_data, get_tokens, get_rand_tokens, get_token_wv
from sample_seeds import __uni2norm, __norm2uni
from align_wv_space import __comparison

word_dataset_base = '../result/epa_expansion/nn'
os.makedirs(word_dataset_base, exist_ok=True)


def baseline_model(dtype='lr', uniform=False):
    print(dtype)
    model = Sequential()
    if dtype == 'lr':
        model.add(Dense(128, input_dim=300, kernel_initializer='normal', activation='tanh'))
        model.add(Dense(32, kernel_initializer='normal', activation='tanh'))
        model.add(Dense(3, kernel_initializer='normal'))
        # model.add(Activation('sigmoid'))
        if uniform:
            model.add(Activation('tanh'))
    if dtype == 'lr2':
        model.add(Dense(128, input_dim=300, kernel_initializer='normal', activation='tanh'))
        model.add(Dense(32, kernel_initializer='normal', activation='tanh'))
        model.add(Dense(3, kernel_initializer='normal'))
        # model.add(Activation('sigmoid'))
        if uniform:
            model.add(Activation('tanh'))
    elif dtype == 'cnn':
        model.add(Conv1D(32, 10, padding='valid', activation='tanh', strides=1, input_shape=(300, 1)))
        # model.add(MaxPooling1D(pool_size=5))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(16))
        model.add(Dropout(0.2))
        model.add(Activation('tanh'))
        model.add(Dense(3))
    elif dtype == 'cnn2':
        model.add(Conv1D(4, 10, padding='valid', activation='tanh', strides=1, input_shape=(300, 1)))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(3))

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def kfold_test(feature, label, dtype, uniform, epoch, batch_size):
    seed = 10
    np.random.seed(seed)
    estimator = KerasRegressor(build_fn=baseline_model, dtype=dtype, uniform=uniform,
                               epochs=epoch, batch_size=batch_size, verbose=0)
    kfold = KFold(n_splits=5, random_state=seed)
    results = cross_val_score(estimator, feature, label, cv=kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    estimator.fit(feature, label)
    return estimator


def fit_model(feature_train, label_train, feature_test, label_test, dtype, uniform, epochs, batch_size):
    if uniform:
        label_train = __norm2uni(label_train)
        label_test = __norm2uni(label_test)
    if 'cnn' in dtype:
        # channel last
        feature_train = np.reshape(feature_train, feature_train.shape + (1, ))
        feature_test = np.reshape(feature_test, feature_test.shape + (1,))

    # model = baseline_model(dtype=dtype, uniform=uniform)
    # print('start training %s %s' % (str(feature_train.shape), str(label_train.shape)))
    # model.fit(feature_train, label_train, epochs=epochs, batch_size=batch_size)
    # print('start evaluating %s %s' % (str(feature_test.shape), str(label_test.shape)))
    # score = model.evaluate(feature_test, label_test, batch_size=batch_size)
    # print(score)

    model = kfold_test(feature_train, label_train, dtype, uniform, epochs, batch_size)
    label_pred = model.predict(feature_test)
    mae = np.mean(np.abs(label_pred - label_test), axis=0)
    print('mae %s' % mae)
    rsme = np.sqrt(np.mean((label_pred - label_test) ** 2, axis=0))
    print('rsme %s' % rsme)

    if uniform:
        label_test = __uni2norm(label_test)
        label_pred = __uni2norm(label_pred)
        mae_ori = np.mean(np.abs(label_pred - label_test), axis=0)
        print('mae ori %s' % mae_ori)
        rsme_ori = np.sqrt(np.mean((label_pred - label_test) ** 2, axis=0))
        print('rsme ori %s' % rsme_ori)
        return model, [mae, rsme, mae_ori, rsme_ori]
    else:
        return model, [mae, rsme]


def train(generate, seed_size, eval_size, epa, epochs, batch_size):
    dtype = args.get('model')
    uniform = (args.get('uniform') == 1)
    print('type %s uniform %s' % (dtype, uniform))
    print('seed %s eval %s epa %s epoch %s batch %s' % (seed_size, eval_size, epa, epochs, batch_size))

    feature_train, label_train, feature_test, label_test = generate_data(generate, seed_size, eval_size, epa)

    model, mae = fit_model(feature_train, label_train, feature_test, label_test,
                           dtype, uniform, epochs, batch_size)

    return model, mae


def expansion(model, dic, culture):
    tokens = dic.keys()
    token_wv = np.array([dic[w] for w in tokens])
    token_epa = model.predict(token_wv, batch_size=5)

    with open(os.path.join(word_dataset_base, 'nn_result_%s_all' % culture), 'w') as fp:
        res = dict(list(zip(tokens, token_epa.tolist())))
        json.dump(res, fp)


def validate(model):
    extreme_tokens_wv = get_token_wv(get_tokens())
    neutral_tokens_wv = get_token_wv(get_rand_tokens())
    extreme_pred = model.predict(extreme_tokens_wv, batch_size=5)
    neutral_pred = model.predict(neutral_tokens_wv, batch_size=5)
    print('extreme')
    print(np.mean(extreme_pred, axis=0))
    print(np.mean(np.abs(extreme_pred), axis=0))
    print(np.std(extreme_pred, axis=0))

    print('neutral')
    print(np.mean(neutral_pred, axis=0))
    print(np.mean(np.abs(neutral_pred), axis=0))
    print(np.std(neutral_pred, axis=0))


# def evaluate(model, culture):
#     dic, s_dic, epa = wv_map(method=align, culture=culture)
#     print('evaluate tokens size on two corpus %s' % len(dic.keys()))
#     w_eval = []
#     gg_eval = []
#     gh_eval = []
#     epa_eval = []
#     for w in dic:
#         w_eval.append(w)
#         gh_eval.append(dic[w][0])
#         gg_eval.append(dic[w][1])
#         epa_eval.append(epa[w])
#     gh_eval = np.array(gh_eval)
#     gg_eval = np.array(gg_eval)
#     epa_eval = np.array(epa_eval)
#     gh_pred = model.predict(gh_eval, batch_size=5)
#     gg_pred = model.predict(gg_eval, batch_size=5)
#
#     if uniform:
#         gh_pred = __uni2norm(gh_pred)
#         gg_pred = __uni2norm(gg_pred)
#
#     print('nn eval epa')
#
#     epa_mask = np.any(epa_eval, axis=1)
#     print('number of words in epa datasets')
#     print(np.sum(epa_mask))
#
#     print('diff gg && %s' % culture)
#     __comparison(gg_pred[epa_mask], gh_pred[epa_mask])
#
#     print('diff gg && epa')
#     __comparison(gg_pred[epa_mask], epa_eval[epa_mask])
#
#     print('diff %s && epa' % culture)
#     __comparison(gh_pred[epa_mask], epa_eval[epa_mask])
#
#     print('google')
#     print(np.mean(gg_pred, axis=0))
#     print(np.mean(np.abs(gg_pred), axis=0))
#     print(np.std(gg_pred, axis=0))
#
#     print(culture)
#     print(np.mean(gh_pred, axis=0))
#     print(np.mean(np.abs(gh_pred), axis=0))
#     print(np.std(gh_pred, axis=0))
#
#     with open(os.path.join(word_dataset_base, 'nn_result_%s_google_%s' % (dtype, culture)), 'w') as fp:
#         res = list(zip(w_eval, epa_eval.tolist(), gg_pred.tolist(), gh_pred.tolist()))
#         res = dict((line[0], line[1:]) for line in res)
#         json.dump(res, fp)

#     return s_dic


def main():
    logging = []
    for epoch in range(5, 120, 20):
        for batch in range(10, 120, 20):
            for epa in range(30, -1, -5):
                # generate_data(3, 600, 1000, 0.1 * epa)
                model, metrics = train(2, 600, 1000, 0.1 * epa, epoch, batch)
                logging.append({
                    'epoch': epoch,
                    'batch': batch,
                    'seed': 600,
                    'eval': 1000,
                    'epa': epa,
                    'mae': metrics
                })

            for seed in range(500, 5001, 500):
                # generate_data(3, 5000, 8000, 2)
                model, metrics = train(2, seed, 8000, 2, epoch, batch)
                logging.append({
                    'epoch': epoch,
                    'batch': batch,
                    'seed': seed,
                    'eval': 8000,
                    'epa': 2,
                    'mae': metrics
                })
    with open(os.path.join(word_dataset_base, 'result'), 'w') as fp:
        json.dump(logging, fp)


if __name__ == '__main__':
    ap = argparse.ArgumentParser('keras deep learning method')
    ap.add_argument('--generate', type=int, required=True)
    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--uniform', type=int, required=True)
    ap.add_argument('--align', type=str, required=True)

    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--eval', type=int, required=True)
    ap.add_argument('--epa', type=float, required=True)

    ap.add_argument('--epoch', type=int, required=True)
    ap.add_argument('--batch', type=int, required=True)

    args = vars(ap.parse_args())

    gen = args.get('generate')
    align = args.get('align')

    main()

    # for epa in range(30, -1, -5):
    #     generate_data(3, 600, 1000, 0.1 * epa)

    # for seed in range(500, 5000, 500):
    #     generate_data(3, seed, 8000, 2.0)
