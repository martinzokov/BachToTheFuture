import numpy as np
import os
from DataHandler import DataHandler
from ModelHandler import Model

training_samples = []
data_handler = DataHandler(t_steps=4, t_step_length=0.25)

for midi_file in os.listdir('midi_data/'):
    print 'processing:', midi_file
    notes = data_handler.get_np_notes('midi_data/'+midi_file)
    training_samples.append(notes)
    print notes[0].shape
    if len(notes[0]) == 0:
        print 'delete ', midi_file
        print len(notes[0].shape), len(notes[1])

print 'Data loaded'
neurons = 100
dropout = 0.3
l_rate = 0.0025
epochs = 1600
file_name = str(neurons) + "n_"+str(dropout)+"do_"+str(l_rate)+"lr_"+str(epochs)+"epo_"+str(len(training_samples))+"samps_adam.hdf5"

model = Model(neurons=neurons, dropout=dropout, learning_rate=l_rate, optimizer='adam', desired_loss=0.3)

model.train_model(training_samples, epochs=epochs, save_weigths=True, save_file=file_name)