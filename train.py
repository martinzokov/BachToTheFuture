import numpy as np
import os
from DataHandler import DataHandler
from ModelHandler import Model

training_samples = []
settings = DataHandler.get_config_params()

data_handler = DataHandler(t_steps=settings["t_steps"], t_step_length=settings["t_step_length"])

print 'Loading data:'
for midi_file in os.listdir(settings["data_dir"]):
    print 'processing:', midi_file
    notes = data_handler.get_np_notes(settings["data_dir"] + midi_file)
    training_samples.append(notes)
    print notes[0].shape
    if len(notes[0]) == 0:
        print 'delete ', midi_file
        print len(notes[0].shape), len(notes[1])

print 'Data loaded'
neurons = settings["neurons"]
dropout = settings["dropout"]
l_rate = settings["learning_rate"]
epochs = settings["epochs"]
file_name = str(neurons) + "n_" + str(dropout) + "do_" + str(l_rate) + "lr_" + str(epochs) + "epochs_" + \
            str(len(training_samples)) + "samps_adam.hdf5"

print 'Building model:'
model = Model(neurons=neurons, dropout=dropout, learning_rate=l_rate, optimizer='adam', desired_loss=0.3)

print 'Training:'
model.train_model(training_samples, epochs=epochs, save_weigths=True, save_file=file_name)
