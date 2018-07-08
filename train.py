import os

from DataHandler import DataHandler
from ModelHandler import Model

"""
This script is used to train a neural network on MIDI music files. Weights can then be saved and used in the
generate.py script. The data_dir configuration parameter sets the working directory from which MIDI files
will be extracted. The scripts uses a DataHandler object to parse files and a Model object from ModelHandler.py
to create a neural network.
"""
training_samples = []
settings = DataHandler.get_config_params()

data_handler = DataHandler(t_steps=settings["t_steps"], t_step_length=settings["t_step_length"])

print('Loading data:')
for midi_file in os.listdir(settings["data_dir"]):
    print('processing:', midi_file)
    notes = data_handler.get_np_notes(settings["data_dir"] + midi_file)
    training_samples.append(notes)
    print(notes[0].shape)
    if len(notes[0]) == 0:
        print('delete ', midi_file)
        print(len(notes[0].shape), len(notes[1]))

print('Data loaded')
neurons = settings["neurons"]
dropout = settings["dropout"]
l_rate = settings["learning_rate"]
epochs = settings["epochs"]
optimizer = settings["optimizer"]
file_name = str(neurons) + "n_" + str(dropout) + "do_" + str(l_rate) + "lr_" + str(epochs) + "epochs_" + \
            str(len(training_samples)) + "samps_"+optimizer+".h5"

print('Building model:')
model = Model(neurons=neurons, dropout=dropout, learning_rate=l_rate, optimizer=optimizer, desired_loss=0.2)

print('Training:')
model.train_model(training_samples, epochs=epochs, save_weigths=True, save_file=settings["weights_dir"] + file_name)
