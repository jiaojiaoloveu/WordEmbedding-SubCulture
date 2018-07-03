import json
import os
import argparse
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Activation, Dropout, Embedding
from keras.layers import GlobalMaxPooling1D, MaxPooling1D
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
from propagate_labels import load_google_word_vectors, word_dataset_base
from sample_seeds import csv_path, read_warriner_ratings
from gen_data import wv_map


def load_train():
    with open(os.path.join(word_dataset_base, 'seed'), 'r') as fp:
        seed_words = json.load(fp)
    return seed_words


def load_test():
    with open(os.path.join(word_dataset_base, 'eval'), 'r') as fp:
        eval_words = json.load(fp)
    return eval_words


def load_all():
    return read_warriner_ratings(csv_path)


def get_wv_space():
    google_news_model_path = '../models/embedding/GoogleNews-vectors-negative300.bin'
    model = load_google_word_vectors(google_news_model_path)
    return model


def load_feature_label(suffix):
    feature = np.load(os.path.join(word_dataset_base, 'feature_' + suffix + '.npy'))
    label = np.load(os.path.join(word_dataset_base, 'label_' + suffix + '.npy'))
    return feature, label


def preprocess_data(word_epa_dataset, suffix):
    wv_feature = []
    epa_label = []
    google_model = get_wv_space()
    google_vocab = set(google_model.vocab.keys())

    def epa2list(epa):
        return [epa['E'], epa['P'], epa['A']]

    for word in word_epa_dataset.keys():
        if word not in google_vocab:
            continue
        feature = google_model[word]
        wv_feature.append(feature)

        label = word_epa_dataset[word]
        epa_label.append(epa2list(label))

    wv_feature = np.array(wv_feature)
    epa_label = np.array(epa_label)
    print(wv_feature.shape)
    print(epa_label.shape)
    np.save(os.path.join(word_dataset_base, 'feature_' + suffix), wv_feature)
    np.save(os.path.join(word_dataset_base, 'label_' + suffix), epa_label)
    return wv_feature, epa_label


def baseline_model(dtype):
    print(dtype)
    model = Sequential()
    if dtype == 'lr':
        model.add(Dense(32, input_dim=300, kernel_initializer='normal', activation='relu'))
        model.add(Dense(3, kernel_initializer='normal'))
    elif dtype == 'cnn':
        model.add(Conv1D(32, 10, padding='valid', activation='relu', strides=1))
        # model.add(MaxPooling1D(pool_size=5))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(16))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(3))
    elif dtype == 'cnn2':
        model.add(Conv1D(4, 10, padding='valid', activation='relu', strides=1))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(3))

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    return model


#def kfold_est(feature, label):
#    seed = 10
#    np.random.seed(seed)
#    estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
#    kfold = KFold(n_splits=10, random_state=seed)
#    results = cross_val_score(estimator, feature, label, cv=kfold)
#    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


def fit_model(feature_train, label_train, feature_test, label_test, dtype):
    if 'cnn' in dtype:
        # channel last
        feature_train = np.reshape(feature_train, feature_train.shape + (1, ))
        feature_test = np.reshape(feature_test, feature_test.shape + (1,))
    model = baseline_model(dtype=dtype)
    print('start training %s %s' % (str(feature_train.shape), str(label_train.shape)))
    model.fit(feature_train, label_train, epochs=10, batch_size=128)
    print('start evaluating %s %s' % (str(feature_test.shape), str(label_test.shape)))
    score = model.evaluate(feature_test, label_test, batch_size=128)
    print(score)
    return model


def generate_data(generate):
    if generate < 2:
        if generate == 0:
            feature, label = load_feature_label('all')
        else:
            feature, label = preprocess_data(load_all(), 'all')
        (items, dimensions) = feature.shape
        mask = np.random.random_sample(items)
        train_test_split = 0.4
        feature_train, label_train = feature[mask < train_test_split], label[mask < train_test_split]
        feature_test, label_test = feature[mask >= train_test_split], label[mask >= train_test_split]
    elif generate < 4:
        if generate == 2:
            feature_train, label_train = load_feature_label('train')
            feature_test, label_test = load_feature_label('test')
        else:
            feature_train, label_train = preprocess_data(load_train(), 'train')
            feature_test, label_test = preprocess_data(load_test(), 'test')
    else:
        print('generate = %s not supported' % generate)
        raise Exception('generate not supported yet')
    return feature_train, label_train, feature_test, label_test


def train():
    generate = args.get('generate')
    model = args.get('model')
    feature_train, label_train, feature_test, label_test = generate_data(generate)
    model = fit_model(feature_train, label_train, feature_test, label_test, model)
    dic = wv_map()
    gg_eval = []
    gh_eval = []
    for w in dic:
        gg_eval.append(dic[w][0])
        gh_eval.append(dic[w][1])
    gg_eval = np.array(gg_eval)
    gh_eval = np.array(gh_eval)
    gg_pred = model.predict(gg_eval, batch_size=5)
    gh_pred = model.predict(gh_eval, batch_size=5)
    res = np.abs(gg_pred - gh_pred)
    print('nn eval epa')
    print(np.mean(res, axis=0))
    print(np.std(res, axis=0))


if __name__ == '__main__':
    ap = argparse.ArgumentParser('keras deep learning method')
    ap.add_argument('--generate', type=int, required=True)
    ap.add_argument('--model', type=str, required=True)
    args = vars(ap.parse_args())
    train()
