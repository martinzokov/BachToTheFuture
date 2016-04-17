import random

from ModelHandler import ModelFactory
from DataHandler import DataHandlerFactory
import numpy as np

data_handler = DataHandlerFactory.create_handler("one_hot", t_steps=5, t_step_length=0.25)
model = ModelFactory.create_model("one_hot", neurons=100, dropout=0.4)

model.load_weights('100n_0.3do_0.002lr_1600epo_15samps_adam.hdf5')


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a[0]) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


seed = np.zeros((1, 4, 129), dtype=np.bool)
generated = [70, 70, 70, 70, 68]
for i in range(5, 250):
    print 'generating: ', i
    # next_note = model.generate(data_handler.sequence_to_one_hot(generated[i-5:i-1]))
    next_note = model.generate(data_handler.sequence_to_one_hot(generated))
    temp = np.random.uniform(0.5, 1.5)
    print temp
    generated.append(sample(next_note, temp))
print generated
# data_handler.create_note_stream(generated).show()
data_handler.save_to_midi(generated)