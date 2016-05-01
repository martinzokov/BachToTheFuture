import ConfigParser
import os
import uuid
from fractions import Fraction
from itertools import groupby

import sys
from music21 import converter
from music21.chord import Chord
from music21.note import Rest, GeneralNote, Note
import music21.midi.translate as m21_translate
import music21.scale as scale
import numpy as np
from music21.stream import Stream


class DataHandler(object):
    """
    An object that can manipulate data for both feature extraction and song generation.
    """
    def __init__(self, t_steps=4, t_step_length=0.25):
        """
        Constructor for the DataHandler.

        :param t_steps: number of time steps per training sequence
        :param t_step_length: length of each time step segment.
        """
        self.T_STEPS = t_steps
        self.T_STEP_LENGTH = t_step_length
        self.ONE_HOT_VECTOR_LENGTH = 129

    def get_midi_representation(self, note):
        """
        Gets the midi representation for a Note object.

        :param note: a music21.GeneralNote
        :return: midi representation for a music21.note.Note object. If it is a music21.note.Rest, return a specific
        value outside the ones allowed in MIDI. If it is a music21.chord.Chord, return only the first (root) note.
        """
        if isinstance(note, Rest):
            return 128
        if isinstance(note, Chord):
            return note.pitches[0].midi
        return note.pitch.midi

    def get_note_objects(self, midi_stream):
        """
        Extracts all music21.note.GeneralNote (superclass for all note types) note objects.

        :param midi_stream: a music21.stream.Stream object which contains all MIDIEvents.
        :return: an array of all GeneralNote objects.
        """
        note_objects = []
        for track in midi_stream:
            for note in track:
                if isinstance(note, GeneralNote):
                    note_objects.append(note)

        return note_objects

    def pad_steps(self, steps_array):
        """
        If a training sequence has less than self.T_STEPS notes in it, it has to be padded because all sequences need
        to be the same length.

        :param steps_array: the sequence to be padded.
        """
        for step in range(0, self.T_STEPS - len(steps_array)):
            steps_array.append([128])

    def get_note_duration(self, note):
        """
        Determines the number of time steps that need to be added for a particular note. If the duration is a Fraction
        (i.e. when there is a triplet), the numerator is divided by the denominator.

        :param note: a music21.note.GeneralNote object
        :return: number of time steps needed to represent a note.
        """
        if isinstance(note.duration.quarterLength, Fraction):
            duration = float(note.duration.quarterLength.numerator) / note.duration.quarterLength.denominator
        else:
            duration = note.duration.quarterLength
        return duration

    def get_note_rep_array(self, file_name, one_hot):
        """
        Converts a MIDI file into a suitable representation for use in the neural network.

        :param file_name: path and file name for a MIDI file that will be used for training.
        :param one_hot: determines if the representation used for notes will be one-hot (1-of-n)
        :return: an array with the appropriate number of steps to represent each note.
        """
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
        """
        Creates a 1-of-n represented note in a numpy array.

        :param int_note: MIDI representation value for a note.
        :return: 1-of-n represented note
        """
        one_hot = np.zeros(129, dtype=np.bool)
        one_hot[int_note] = 1
        return one_hot

    def to_onehot(self, int_note):
        """
        Creates a 3D 1-of-n represented note in a numpy array.
        :param int_note:
        :return:
        """
        one_hot = np.zeros((1, 1, 129), dtype=np.bool)
        one_hot[0, 0, int_note] = 1
        return one_hot

    def count_note_events(self, sequence):
        """
        Counts the number of consecutive times that a note occurs in a sequence.

        :param sequence: an array of notes in MIDI representation.
        :return: an array of tuples (note_int, number of consecutive occurrences)
        """
        return [(k, sum(1 for i in g)) for k, g in groupby(sequence)]

    def create_note_stream(self, notes_sequence):
        """
        Creates a music21.stream.Stream object to which notes are added.

        :param notes_sequence: sequence of notes to add in a stream.
        :return: a Stream of Note objects.
        """
        notes_arr = self.get_notes_from_sequence(notes_sequence)
        stream = Stream()
        for note in notes_arr:
            stream.append(note)
        return stream

    def get_seed(self):
        """
        Generates a seed based on the configuration file. Four main types are used:
         'from_existing' - takes a seed from an existing MIDI file. A specific amount of time steps is extracted,
         based on a parameter from the config file.
         'from_scale' - creates a seed from a specific scale ('major', 'minor' or 'harmonic_minor'). It randomises
         which notes will be taken for the seed.
         'custom' - asks the user for a custom seed sequence. Notes have to be entered in a MIDI representation.
         'from_random_file' - takes a random file from the data_dir that is set in the configuration file.

        :return: a seed sequence
        """
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
        if mode == 'custom':
            seed_input = input("Please enter seed sequence (comma separated sequence of notes in MIDI representation):")
            seed = [int(i) for i in seed_input]
        if mode == 'from_random_file':
            files = os.listdir(settings["data_dir"])
            file_index = np.random.randint(0, len(files))
            seed_source = self.get_note_rep_array(settings["data_dir"] + files[file_index], False)
            for i in range(settings["seed_size"]):
                seed.append(seed_source[i][0])
        return seed

    def get_notes_from_scale(self, scale_obj, length):
        """
        Randmly picks notes from a scale.

        :param scale_obj: a music21.scale.Scale object.
        :param length: number of notes to be picked. Determined by seed length.
        :return: a list of random notes in a particular scale.
        """
        seed_arr = []
        for note in scale_obj.pitches:
            seed_arr.append(note.midi)
        return np.random.choice(a=seed_arr, size=length).tolist()

    def save_to_midi(self, stream, location='generated/'):
        """
        Creates a MIDI file from a music21.stream.Stream object.

        :param stream: a Stream that will be saved to file
        :param location: file path for the MIDI file.
        """
        midi_data = m21_translate.streamToMidiFile(self.create_note_stream(stream))
        midi_data.open(location + str(uuid.uuid4()) + '.mid', attrib='wb')
        midi_data.write()

    def parse_midi_stream(self, file_name):
        """
        Creates sequences two arrays - sequences which will be used as input and the expected output for each sequence.

        :param file_name: file from which to extract sequences.
        :return: input sequences and expected output. Both in 1-of-n representation.
        """
        note_list_arr = self.get_note_rep_array(file_name, one_hot=True)
        notes_input = []
        notes_output = []
        for t in range(0, len(note_list_arr) - self.T_STEPS, 1):
            notes_input.append(note_list_arr[t:t + self.T_STEPS])
            notes_output.append(note_list_arr[t + self.T_STEPS])

        return notes_input, notes_output

    def get_np_notes(self, midi_file):
        """
        Parses a MIDI file, extracts small sequences and the expected output for each, then converts them
        to numpy arrays.

        :param midi_file: filepath to a MIDI file
        :return: two numpy arrays of sequences and their expected output.
        """
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
        """
        Converts an array of MIDI represented notes to an array of music21.note objects.

        :param note_sequence: MIDI represented note sequence.
        :return: array of Note and Rest objects.
        """
        note_objects = []

        notes = self.count_note_events(note_sequence)

        for pair in notes:
            if pair[0] == 128:
                note_objects.append(Rest(quarterLength=pair[1] * self.T_STEP_LENGTH))
            else:
                note_objects.append(Note(pair[0], quarterLength=pair[1] * self.T_STEP_LENGTH))

        return note_objects

    def sequence_to_one_hot(self, note_sequence):
        """
        Given an array of MIDI represented notes, converts it to a 3D array with 1-of-n encoding of notes.

        :param note_sequence: sequence to convert.
        :return: 3D numpy array with the shape (samples, time steps, features /1-0f-n notes/).
        """
        np_notes_in = np.zeros((1, 0, self.ONE_HOT_VECTOR_LENGTH), dtype=np.bool)
        for note in note_sequence:
            np_notes_in = np.append(np_notes_in, self.to_onehot(note), axis=1)
        return np_notes_in

    def get_weights(self, model, settings):
        """
        Loads a weights file into a model. Can be a random weight or a specific file.

        :param model: the model to load weights into. NOTE: the model needs the same parameters as the model
        that was used to save the weights
        :param settings: settings from the configuration file.
        """
        if settings["weights_mode"] == 'random':
            weights = os.listdir(settings["weights_dir"])
            if len(weights) == 0:
                print "No weights files in "+settings["weights_dir"]
                sys.exit()
            weight_index = np.random.randint(0, len(weights)-1)
            model.load_weights(settings["weights_dir"]+weights[weight_index])
        if settings["weights_mode"] == 'custom':
            model.load_weights(settings["weights_dir"]+settings["weights_file"])

    def get_config_params():
        """
        Parses a configuration file with settings for the neural network, training and generation scripts.
        This is a static method.

        :return: a dictionary of key:value pairs with settings.
        """
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

        settings["weights_mode"] = config.get("generating", "weights_mode") or 'random'
        settings["weights_dir"] = config.get("generating", "weights_dir") or 'weights'
        settings["weights_file"] = config.get("generating", "weights_file") or ''

        if settings["save_dir"][-1] != '/':
            settings["save_dir"] += '/'
        if settings["weights_dir"][-1] != '/':
            settings["weights_dir"] += '/'

        return settings

    get_config_params = staticmethod(get_config_params)

