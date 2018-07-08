import numpy as np

from DataHandler import DataHandler
from ModelHandler import Model
"""
The script used to generate new music. It creates a model and loads a weights file. Then it gets a seed
from the DataHandler object and starts th generation process.
"""
settings = DataHandler.get_config_params()
data_handler = DataHandler()

print('Building model')
neurons = settings["neurons"]
dropout = settings["dropout"]
l_rate = settings["learning_rate"]
epochs = settings["epochs"]
optimizer = settings["optimizer"]
model = Model(neurons=neurons, dropout=dropout, learning_rate=l_rate, optimizer=optimizer, desired_loss=0.3)

print('Loading weights')
data_handler.get_weights(model, settings)


def sample(prob_distribution, temperature=1.0):
    """
    A function to sample an index from the array of probabilities.

    :param prob_distribution: an array with probabilities for each class (note).
    :param temperature: denominator parameter used to divide the natural log of each element. Helps in transforming
    values in the probability array

    :return: the index of the element which was drawn from the array of probabilities.
    """
    prob_distribution = np.log(prob_distribution[0]) / temperature
    prob_distribution = np.exp(prob_distribution) / np.sum(np.exp(prob_distribution))
    return np.argmax(np.random.multinomial(1, prob_distribution, 1))

print('Generating')
generated = data_handler.get_seed()
print(generated)
for i in range(0, settings["gen_length"]):
    # next_note = model.generate(data_handler.sequence_to_one_hot(generated[i-5:i-1]))
    next_note = model.generate(data_handler.sequence_to_one_hot(generated))
    temp = np.random.uniform(0.5, 1.5)
    generated.append(sample(next_note, temp))
print(generated)

data_handler.save_to_midi(generated, settings["save_dir"])
