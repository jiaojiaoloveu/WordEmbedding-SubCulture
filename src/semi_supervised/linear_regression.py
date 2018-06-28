import json
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
from propagate_labels import load_word_vectors, word_dataset_base
from sample_seeds import csv_path, read_warriner_ratings


def load_train():
    with open(os.path.join(word_dataset_base, 'seed'), 'w') as fp:
        seed_words = json.load(fp)
    return seed_words


def load_test():
    with open(os.path.join(word_dataset_base, 'eval'), 'w') as fp:
        eval_words = json.load(fp)
    return eval_words


def load_all():
    return read_warriner_ratings(csv_path)


def get_wv_space():
    google_news_model_path = '../models/embedding/GoogleNews-vectors-negative300.bin'
    model = load_word_vectors(google_news_model_path)
    return model


def preprocess_data():
    wv_feature = []
    epa_label = []
    word_epa_dataset = load_all()
    google_model = get_wv_space()

    def epa2list(epa):
        return [epa['E'], epa['P'], epa['A']]

    for word in word_epa_dataset.keys():
        feature = google_model[word]
        wv_feature.append(feature)

        label = word_epa_dataset[word]
        epa_label.append(epa2list(label))

    wv_feature = np.array(wv_feature)
    epa_label = np.array(epa_label)
    print(wv_feature.shape)
    print(epa_label.shape)
    np.save(os.path.join(word_dataset_base, 'feature'), wv_feature)
    np.save(os.path.join(word_dataset_base, 'label'), epa_label)
    return wv_feature, epa_label


def baseline_model():
    model = Sequential()
    model.add(Dense(32, input_dim=300, kernel_initializer='normal', activation='relu'))
    model.add(Dense(3, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def kfold_est(feature, label):
    seed = 10
    np.random.seed(seed)
    estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
    kfold = KFold(n_splits=10, random_state=seed)
    results = cross_val_score(estimator, feature, label, cv=kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


def linear_regression():
    feature, label = preprocess_data()
    kfold_est(feature, label)


if __name__ == '__main__':
    linear_regression()
