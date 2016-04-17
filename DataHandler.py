import uuid
from fractions import Fraction
from itertools import groupby
from music21 import converter
from music21.chord import Chord
from music21.note import Rest, GeneralNote, Note
import music21.midi.translate as m21_translate
import numpy as np
from music21.stream import Stream


class DataHandler(object):
    def __init__(self, t_steps, t_step_length):
        self.T_STEPS = t_steps
        self.T_STEP_LENGTH = t_step_length
        self.ONE_HOT_VECTOR_LENGTH = 129

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
        for step in range(0, self.T_STEPS - len(steps_array)):
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

            if duration < self.T_STEP_LENGTH:
                steps = 1
            else:
                steps = int(duration / self.T_STEP_LENGTH)

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

    def parse_midi_stream(self, file_name):
        note_list_arr = self.get_note_rep_array(file_name, one_hot=True)
        notes_input = []
        notes_output = []
        for t in range(0, len(note_list_arr) - self.T_STEPS, 1):
            notes_input.append(note_list_arr[t:t + self.T_STEPS])
            notes_output.append(note_list_arr[t + self.T_STEPS])

        return notes_input, notes_output

    def get_np_notes2(self, midi_file):
        notes_input, notes_output = self.parse_midi_stream(midi_file)
        num_samples = 50
        np_notes_in = np.zeros((num_samples, self.T_STEPS, self.ONE_HOT_VECTOR_LENGTH), dtype=np.int32)
        np_notes_out = np.zeros((num_samples, self.ONE_HOT_VECTOR_LENGTH), dtype=np.int32)
        samples_list = []
        for j in range(len(notes_input)/num_samples):

            for i in range(num_samples):
                if len(notes_input[i]) < self.T_STEPS:
                    self.pad_steps(notes_input[i])
                np_notes_in[i] = notes_input[(num_samples*j) + i]
                np_notes_out[i] = notes_output[(num_samples*j) + i]
            samples_list.append((np_notes_in, np_notes_out))

        return samples_list

    def get_np_notes(self, midi_file):
        notes_input, notes_output = self.parse_midi_stream(midi_file)
        np_notes_in = np.zeros((len(notes_input), self.T_STEPS, self.ONE_HOT_VECTOR_LENGTH), dtype=np.int32)
        np_notes_out = np.zeros((len(notes_output), self.ONE_HOT_VECTOR_LENGTH), dtype=np.int32)

        for i in range(len(notes_input)):
            if len(notes_input[i]) < self.T_STEPS:
                self.pad_steps(notes_input[i])
            np_notes_in[i] = notes_input[i]
            np_notes_out[i] = notes_output[i]

        return np_notes_in, np_notes_out

    def get_notes_from_sequence(self, note_sequence):
        note_objects = []

        notes = self.count_note_events(note_sequence)

        for pair in notes:
            if pair[0] == 128:
                note_objects.append(Rest(quarterLength=pair[1] * self.T_STEP_LENGTH))
            else:
                note_objects.append(Note(pair[0], quarterLength=pair[1] * self.T_STEP_LENGTH))

        return note_objects

    def sequence_to_one_hot(self, note_sequence):
        np_notes_in = np.zeros((1, 0, self.ONE_HOT_VECTOR_LENGTH), dtype=np.bool)
        for note in note_sequence:
            np_notes_in = np.append(np_notes_in, self.to_onehot(note), axis=1)
        return np_notes_in

