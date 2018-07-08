import csv
import warnings

from keras.callbacks import EarlyStopping, Callback
from keras.optimizers import RMSprop, Adam
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.models import Sequential, load_model, model_from_json
import numpy as np


class Model(object):
    """
    The Model object is used to create a new Sequential() model from Keras' implementation.
    """

    def __init__(self, neurons=100, dropout=0.3, learning_rate=0.001, optimizer='adam', desired_loss=0.3):
        """
        Constructor for the model that will be used.

        :param neurons: number of neurons in the hidden layer
        :param dropout: probability that a neuron will be disabled; uses values between 0 and 1
        :param learning_rate: learning rate for the chosen optimizer
        :param optimizer: the name of the optimizer - currently supported are 'rmsprop' and 'adam'
        :param desired_loss: if while training this value for loss is reached, training stops
        """
        self.neurons = neurons
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.optimizer_obj = None
        self.desired_loss = desired_loss
        self.ONE_HOT_VECTOR_LENGTH = 129

        self.model = self.create_model()

    def train_model(self, training_data, epochs, save_weigths=False, save_file='weights.h5'):
        """
        Uses the created model and training_data to optimize the weights.

        :param training_data: an array of training samples. Each element has two parts - an array of training examples
        in the first position and an array of their expected outputs in the second position.
        :param epochs: number of epochs for which the neural network will be trained.
        :param save_weigths: determines if the trained model's weights will be saved
        :param save_file: if save_weights is True, weights are saved in this file.
        """
        print(len(training_data))
        stop_callback = EarlyStoppingSpecificValue(monitor='loss', desired_val=self.desired_loss)
        loss_history = LossHistory()
        for i, sample in enumerate(training_data):
            print(i+1, " out of ", len(training_data))
            self.model.fit(sample[0], sample[1], epochs=epochs, callbacks=[loss_history])
            if save_weigths:
                self.model.save_weights(save_file)
        loss_file = open('loss_history.csv', 'wb')
        loss_file.write(','.join([str(elem) for elem in loss_history.losses]))

    def load_weights(self, save_file='weights.h5'):
        """
        Read from a file information about previously saved weights.
        :param save_file: name and path for the save file.
        """
        self.model.load_weights(save_file)

    def generate(self, input_sequence):
        """
        Takes in a 3D input (samples, time steps, features) and predicts the output.

        :param input_sequence: a 3D array of time steps and notes in 1-of-n representation.
        :return: vector with probability distribution for each class (note).
        """
        return self.model.predict(input_sequence)

    def create_model(self):
            """
            Creates a Keras Sequential() LSTM Model object with the parameters that were passed to this class' constructor.

            :return: a working LSTM model
            """
            model = Sequential()
            model.add(LSTM(self.ONE_HOT_VECTOR_LENGTH, return_sequences=True))
            model.add(Dropout(self.dropout))
            model.add(LSTM(self.neurons, return_sequences=False))
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
    """
    A callback class to stop training early if the loss reaches a specific value.
    """
    def __init__(self, monitor='val_loss', desired_val=0., patience=0, verbose=0, mode='auto'):
        """
        Constructor for the callback.

        :param monitor: name of the parameter that is being monitored. Used only for 'loss'
        :param desired_val: the specific loss value on which training will stop.
        """
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
        """
        Called at the end of each epoch. Compares current loss value with the desired_loss parameter.
        """
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


class LossHistory(Callback):
    """
    Used to record the history of loss values.
    """
    def on_train_begin(self, logs={}):
        """
        Creates the array which holds loss values.
        :param logs: not used.
        """
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        """
        Appends new loss values after each training batch.

        :param batch: current batch
        :param logs: logs with information about tracked parameters.
        """
        self.losses.append(logs.get('loss'))
