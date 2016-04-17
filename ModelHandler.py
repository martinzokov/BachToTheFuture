import warnings

from keras.callbacks import EarlyStopping, Callback
from keras.optimizers import RMSprop, Adam
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.models import Sequential
import numpy as np


class Model(object):

    def __init__(self, neurons=100, dropout=0.3, learning_rate=0.001, optimizer='adam', desired_loss=0.3):
        self.neurons = neurons
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.optimizer_obj = None
        self.desired_loss = desired_loss
        self.ONE_HOT_VECTOR_LENGTH = 129

        self.model = self.create_model()

    def train_model(self, training_data, epochs, save_weigths=False, save_file='weights.hdf5'):
        print len(training_data)
        stop_callback = EarlyStoppingSpecificValue(monitor='loss', desired_val=self.desired_loss)
        for i, sample in enumerate(training_data):
            print i+1, " out of ", len(training_data)
            self.model.fit(sample[0], sample[1], nb_epoch=epochs, callbacks=[stop_callback])
            if save_weigths:
                self.model.save_weights(save_file, overwrite=True)
        # TO TRY! step by step training through a song

    def load_weights(self, save_file='weights.hdf5'):
        self.model.load_weights(save_file)

    def generate(self, input_sequence):
        return self.model.predict(input_sequence)

    def create_model(self):
        model = Sequential()
        model.add(LSTM(input_dim=self.ONE_HOT_VECTOR_LENGTH, output_dim=self.neurons, return_sequences=True))
        model.add(Dropout(self.dropout))
        model.add(LSTM(input_dim=self.neurons, output_dim=self.neurons, return_sequences=False))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.ONE_HOT_VECTOR_LENGTH))
        model.add(Activation('softmax'))
        if self.optimizer == 'adam':
            self.optimizer_obj = Adam(lr=self.learning_rate)
        if self.optimizer == 'rmsprop':
            self.optimizer_obj = RMSprop(lr=self.learning_rate)

        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer_obj)
        return model


class EarlyStoppingSpecificValue(EarlyStopping):
    def __init__(self, monitor='val_loss', desired_val=0., patience=0, verbose=0, mode='auto'):
        super(Callback, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.desired_val = desired_val

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode), RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Early stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)

        if self.monitor_op(self.desired_val, current):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print('Epoch %05d: early stopping' % (epoch))
                self.model.stop_training = True
            self.wait += 1