from read_news_headline import read_epa, get_comp_word_vector
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
import json


def baseline_model():
    model = Sequential()
    model.add(LSTM(500, dropout=0.2, recurrent_dropout=0.2, input_shape=(3, 300)))
    model.add(Dense(300, activation='tanh'))
    model.add(Dense(3))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    return model


def kfold_test(feature, label, epoch, batch_size):
    seed = 10
    np.random.seed(seed)
    estimator = KerasRegressor(build_fn=baseline_model, epochs=epoch, batch_size=batch_size, verbose=0)
    kfold = KFold(n_splits=5, random_state=seed)
    results = cross_val_score(estimator, feature, label, cv=kfold)
    print(results)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    estimator.fit(feature, label)
    return estimator


def train():
    svo, svo_wv, epa = read_epa()
    print('svo shape %s' % str(svo.shape))
    print('svo wv shape %s' % str(svo_wv.shape))
    print('epa shape %s' % str(epa.shape))

    models = []

    for axis in range(0, 4):
        label = epa[:, axis, :]
        model = kfold_test(svo_wv, label, epoch=10, batch_size=50)
        pred = model.predict(svo_wv)
        mae = np.mean(np.abs(pred - label), axis=0)
        print('axis %s, mae %s' % (axis, mae))
        models.append(model)

    with open('../result/state_prediction/epa', 'w') as fp:
        json.dump(epa.tolist(), fp)

    evaluate(models, svo, epa, svo_wv, 'general')
    evaluate(models, svo, epa, np.array(get_comp_word_vector(svo, 'github')), 'github')
    evaluate(models, svo, epa, np.array(get_comp_word_vector(svo, 'twitter')), 'twitter')


def evaluate(model_list, svo, epa, wv, name):
    wv_mask = np.all(np.all(wv, axis=2), axis=1)
    wv = wv[wv_mask]
    for axis in range(0, 4):
        model = model_list[axis]
        pred_epa = model.predict(wv)
        print(np.mean(pred_epa, axis=0))
        with open('../result/state_prediction/%s_%s' % (name, axis), 'w') as fp:
            json.dump(pred_epa.tolist(), fp)


if __name__ == '__main__':
    train()
