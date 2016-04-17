import ConfigParser
import uuid
from fractions import Fraction
from itertools import groupby
from music21 import converter
from music21.chord import Chord
from music21.note import Rest, GeneralNote, Note
import music21.midi.translate as m21_translate
import music21.scale as scale
import numpy as np
from music21.stream import Stream


class DataHandler(object):
    def __init__(self, t_steps=4, t_step_length=0.25):
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
        settings = DataHandler.get_config_params()
        seed = []
        mode = settings["seed_mode"]
        if mode == 'from_existing':
            seed_source = self.get_note_rep_array(settings["data_dir"] + settings["seed_source"], False)
            for i in range(settings["seed_size"]):
                seed.append(seed_source[i][0])
        if mode == 'from_scale':
            seed_scale = settings["seed_scale"]
            if seed_scale == 'major':
                sc = scale.MajorScale()
                seed = self.get_notes_from_scale(scale_obj=sc, length=settings["seed_size"])
            if seed_scale == 'minor':
                sc = scale.MinorScale()
                seed = self.get_notes_from_scale(scale_obj=sc, length=settings["seed_size"])
            if seed_scale == 'harmonic_minor':
                sc = scale.HarmonicMinorScale()
                seed = self.get_notes_from_scale(scale_obj=sc, length=settings["seed_size"])
        return seed

    def get_notes_from_scale(self, scale_obj, length):
        seed_arr = []
        for note in scale_obj.pitches:
            seed_arr.append(note.midi)
        return np.random.choice(a=seed_arr, size=length).tolist()

    def save_to_midi(self, stream, location='generated/'):
        midi_data = m21_translate.streamToMidiFile(self.create_note_stream(stream))
        midi_data.open(location + str(uuid.uuid4()) + '.mid', attrib='wb')
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

    def get_config_params():
        config = ConfigParser.ConfigParser()
        config.read('config.ini')

        settings = {}
        settings["data_dir"] = config.get("training", "data_dir") or "./"

        settings["t_steps"] = int(config.get("training", "t_steps")) or 4
        settings["t_step_length"] = float(config.get("training", "t_step_length")) or 0.25

        settings["neurons"] = int(config.get("training", "neurons")) or 100
        settings["epochs"] = int(config.get("training", "epochs")) or 1500
        settings["dropout"] = float(config.get("training", "dropout")) or 0.3
        settings["learning_rate"] = float(config.get("training", "learning_rate")) or 0.001
        settings["optimizer"] = config.get("training", "optimizer") or 'adam'

        if settings["data_dir"][-1] != '/':
            settings["data_dir"] += '/'

        # generation settings
        settings["save_dir"] = config.get("generating", "save_dir") or "./"

        settings["seed_mode"] = config.get("generating", "seed_mode") or "from_existing"
        settings["seed_source"] = config.get("generating", "seed_source") or ""
        settings["gen_length"] = int(config.get("generating", "gen_length")) or 250
        settings["seed_size"] = int(config.get("generating", "seed_size")) or 4
        settings["seed_scale"] = config.get("generating", "seed_scale") or 'major'

        if settings["save_dir"][-1] != '/':
            settings["save_dir"] += '/'

        return settings

    get_config_params = staticmethod(get_config_params)

