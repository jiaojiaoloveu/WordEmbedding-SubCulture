from read_news_headline import read_epa
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
import numpy as np


def baseline_model():
    model = Sequential()
    model.add(LSTM(300, dropout=0.2, recurrent_dropout=0.2, input_shape=(3, 300)))
    model.add(LSTM(300))
    model.add(LSTM(3, return_sequences=True))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def train():
    svo, svo_wv, epa = read_epa()
    print('svo shape %s' % str(svo.shape))
    print('svo wv shape %s' % str(svo_wv.shape))
    print('epa shape %s' % str(epa.shape))
    model = baseline_model()
    model.fit(svo_wv, epa[:, 0, :], epochs=10, batch_size=50)


if __name__ == '__main__':
    train()
