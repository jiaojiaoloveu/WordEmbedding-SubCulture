import json
import os
import argparse
import numpy as np
from sample_seeds import __uni2norm
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Activation, Dropout, Embedding
from keras.layers import GlobalMaxPooling1D, MaxPooling1D
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
from gen_data import wv_map, generate_data


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
    label_pred = model.predict(feature_test)
    mae = np.mean(np.abs(label_pred - label_test), axis=0)
    print('mae %s' % mae)
    return model


def train():
    generate = args.get('generate')
    model = args.get('model')
    uniform = args.get('uniform')
    feature_train, label_train, feature_test, label_test = generate_data(generate, uniform == 0)
    model = fit_model(feature_train, label_train, feature_test, label_test, model)
    dic = wv_map()
    gg_eval = []
    gh_eval = []
    for w in dic:
        gh_eval.append(dic[w][0])
        gg_eval.append(dic[w][1])
    gh_eval = np.array(gh_eval)
    gg_eval = np.array(gg_eval)
    gh_pred = model.predict(gh_eval, batch_size=5)
    gg_pred = model.predict(gg_eval, batch_size=5)
    if uniform:
        gh_pred = __uni2norm(gh_pred)
        gg_pred = __uni2norm(gg_pred)
    print(gh_pred)
    print(gg_pred)
    print('nn eval epa')

    print('google')
    print(np.mean(gg_pred, axis=0))
    print(np.mean(np.abs(gg_pred), axis=0))
    print(np.std(gg_pred, axis=0))

    print('github')
    print(np.mean(gh_pred, axis=0))
    print(np.mean(np.abs(gh_pred), axis=0))
    print(np.std(gh_pred, axis=0))


if __name__ == '__main__':
    ap = argparse.ArgumentParser('keras deep learning method')
    ap.add_argument('--generate', type=int, required=True)
    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--uniform', type=int, required=True)
    args = vars(ap.parse_args())
    train()
