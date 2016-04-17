import uuid
from fractions import Fraction
from itertools import groupby

from music21 import converter
from music21.chord import Chord
from music21.note import Rest, GeneralNote, Note
import music21.midi.translate as m21_translate
import numpy as np
import time
from music21.stream import Stream


class DataHandler(object):
    def __init__(self, t_steps, t_step_length):
        self.t_steps = t_steps
        self.t_step_length = t_step_length

    def get_midi_representation(self, note):
        if isinstance(note, Rest):
            return 128
        if isinstance(note, Chord):
            return note.pitches[0].midi
        return note.pitch.midi

    def get_note_objects(self, midi_stream):
        note_objects = []
        for track in midi_stream:
            for note in track:
                if isinstance(note, GeneralNote):
                    note_objects.append(note)

        return note_objects

    def pad_steps(self, steps_array):
        for step in range(0, self.t_steps - len(steps_array)):
            steps_array.append([128])

    def get_note_duration(self, note):
        if isinstance(note.duration.quarterLength, Fraction):
            duration = float(note.duration.quarterLength.numerator) / note.duration.quarterLength.denominator
        else:
            duration = note.duration.quarterLength
        return duration

    def get_note_rep_array(self, file_name, one_hot):
        midi_stream = converter.parse(file_name)
        note_list_arr = []

        for note in self.get_note_objects(midi_stream):

            duration = self.get_note_duration(note)

            if duration < self.t_step_length:
                steps = 1
            else:
                steps = int(duration / self.t_step_length)

            temp_note = [self.get_midi_representation(note)]
            for num_steps in range(steps):
                if one_hot:
                    note_list_arr.append(self.to_onehot_1d(temp_note))
                else:
                    note_list_arr.append(temp_note)
        return note_list_arr

    def to_onehot_1d(self, int_note):
        one_hot = np.zeros(129, dtype=np.bool)
        one_hot[int_note] = 1
        return one_hot

    def to_onehot(self, int_note):
        one_hot = np.zeros((1, 1, 129), dtype=np.bool)
        one_hot[0, 0, int_note] = 1
        return one_hot

    def get_np_notes(self, midi_file):
        pass  # implemented in subclasses

    def get_notes_from_sequence(self, notes_np):
        pass  # implemented in subclasses

    def count_note_events(self, sequence):
        return [(k, sum(1 for i in g)) for k, g in groupby(sequence)]

    def create_note_stream(self, notes_sequence):
        notes_arr = self.get_notes_from_sequence(notes_sequence)
        stream = Stream()
        for note in notes_arr:
            stream.append(note)
        return stream

    def get_seed(self):
        pass  # better seed options

    def save_to_midi(self, stream):
        midi_data = m21_translate.streamToMidiFile(self.create_note_stream(stream))
        midi_data.open('generated/' + str(uuid.uuid4()) + '.midi', attrib='wb')
        midi_data.write()

class OneHotDataHandler(DataHandler):
    """ A data handler for One-Hot encoding """

    def __init__(self, t_steps, t_step_length):
        super(OneHotDataHandler, self).__init__(t_steps, t_step_length)
        self.ONE_HOT_VECTOR_LENGTH = 129

    def parse_midi_stream(self, file_name):
        note_list_arr = self.get_note_rep_array(file_name, one_hot=True)
        notes_input = []
        notes_output = []
        for t in range(0, len(note_list_arr) - self.t_steps, 1):
            notes_input.append(note_list_arr[t:t + self.t_steps])
            notes_output.append(note_list_arr[t + self.t_steps])

        return notes_input, notes_output

    def get_np_notes2(self, midi_file):
        notes_input, notes_output = self.parse_midi_stream(midi_file)
        num_samples = 50
        np_notes_in = np.zeros((num_samples, self.t_steps, self.ONE_HOT_VECTOR_LENGTH), dtype=np.int32)
        np_notes_out = np.zeros((num_samples, self.ONE_HOT_VECTOR_LENGTH), dtype=np.int32)
        samples_list = []
        for j in range(len(notes_input)/num_samples):

            for i in range(num_samples):
                if len(notes_input[i]) < self.t_steps:
                    self.pad_steps(notes_input[i])
                np_notes_in[i] = notes_input[(num_samples*j) + i]
                np_notes_out[i] = notes_output[(num_samples*j) + i]
            samples_list.append((np_notes_in, np_notes_out))

        return samples_list

    def get_np_notes(self, midi_file):
        notes_input, notes_output = self.parse_midi_stream(midi_file)
        np_notes_in = np.zeros((len(notes_input), self.t_steps, self.ONE_HOT_VECTOR_LENGTH), dtype=np.int32)
        np_notes_out = np.zeros((len(notes_output), self.ONE_HOT_VECTOR_LENGTH), dtype=np.int32)

        for i in range(len(notes_input)):
            if len(notes_input[i]) < self.t_steps:
                self.pad_steps(notes_input[i])
            np_notes_in[i] = notes_input[i]
            np_notes_out[i] = notes_output[i]

        return np_notes_in, np_notes_out

    def get_notes_from_sequence(self, note_sequence):
        note_objects = []

        notes = self.count_note_events(note_sequence)

        for pair in notes:
            if pair[0] == 128:
                note_objects.append(Rest(quarterLength=pair[1] * self.t_step_length))
            else:
                note_objects.append(Note(pair[0], quarterLength=pair[1] * self.t_step_length))

        return note_objects

    def sequence_to_one_hot(self, note_sequence):
        np_notes_in = np.zeros((1, 0, self.ONE_HOT_VECTOR_LENGTH), dtype=np.bool)
        for note in note_sequence:
            np_notes_in = np.append(np_notes_in, self.to_onehot(note), axis=1)
        return np_notes_in



class RawDataHandler(DataHandler):
    """ A data handler for raw value parsing """
    def __init__(self, t_steps, t_step_length):
        super(RawDataHandler, self).__init__(t_steps, t_step_length)

    def parse_midi_stream(self, file_name):
        note_list_arr = self.get_note_rep_array(file_name, one_hot=False)
        notes_input = []
        notes_output = []
        expected_output = self.shift_for_training(note_list_arr, 1)

        for t in range(0, len(note_list_arr) - self.t_steps, 1):
            notes_input.append(note_list_arr[t:t + self.t_steps])
            notes_output.append(expected_output[t + self.t_steps])

        # return note_list_arr, expected_output
        return notes_input, notes_output

    def shift_for_training(self, sequence, shift_amount):
        return sequence[shift_amount:] + sequence[:shift_amount]

    def get_np_notes(self, midi_file, mode=1):
        if mode == 1:
            notes_input, notes_output = self.parse_midi_stream(midi_file)
            np_notes_in = np.zeros((1, len(notes_input), 1), dtype=np.int32)
            np_notes_out = np.zeros((1, len(notes_output), 1), dtype=np.int32)
            for i in range(len(notes_input)):
                np_notes_in[0, i] = notes_input[i]
                np_notes_out[0, i] = notes_output[i]
        if mode == 2:
            notes_input, notes_output = self.parse_midi_stream(midi_file)
            np_notes_in = np.zeros((len(notes_input), self.t_steps, 1), dtype=np.int32)
            np_notes_out = np.zeros((len(notes_output), 1), dtype=np.int32)

            for i in range(len(notes_input)):
                if len(notes_input[i]) < self.t_steps:
                    self.pad_steps(notes_input[i])
                np_notes_in[i] = notes_input[i]
                np_notes_out[i] = notes_output[i]
        return np_notes_in, np_notes_out

    def flatten_3d_to_2d(self, numpy_3d):
        return numpy_3d.reshape((len(numpy_3d) * len(numpy_3d[0]), 1))

    def get_notes_from_sequence(self, notes_np):
        note_objects = []
        #flat = np.ceil(self.flatten_3d_to_2d(notes_np))
        notes = self.count_note_events(notes_np)

        for pair in notes:
            if pair[0] > 100:
                note_objects.append(Rest(quarterLength=pair[1] * self.t_step_length))
            else:
                note_objects.append(Note(pair[0], quarterLength=pair[1] * self.t_step_length))

        return note_objects

    def sequence_to_3d(self, note_sequence):
        np_notes_in = np.asarray(note_sequence).reshape((1, len(note_sequence), 1))
        return np_notes_in


class DataHandlerFactory(object):
    def create_handler(handler_type, t_steps, t_step_length):
        if handler_type == "one_hot":
            return OneHotDataHandler(t_steps, t_step_length)
        if handler_type == "raw":
            return RawDataHandler(t_steps, t_step_length)
        assert 0, "Bad handler creation: " + handler_type

    create_handler = staticmethod(create_handler)
