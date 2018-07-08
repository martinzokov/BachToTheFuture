# Bach To The Future

This project was my dissertation thesis for my last year at uni. Originally it was trained on music from J.S. Bach (hence the name of the project).

The following Python dependencies are required to run - music21, keras, numpy, h5py

## 1. How it works
### Modelling music
Music is essentially a time-series of notes that change over time. This is why an LSTM ([Long Short-term Memory](http://www.bioinf.jku.at/publications/older/2604.pdf)) network was chosen. It has the ability to recognise patterns in time. Essentially, the two main components that had to be considered were how to represent notes and how to represent time so that the neural network can process compositions.

### Input
So how do we represent notes? The LSTM network processes MIDI files which makes things a bit easier, because it doesn't rely on sound waves. The way MIDI represents music is by encoding each note as an integer from 0 to 128. This is more than necessary for most music pieces (a grand piano has 88 keys). So using a 128 vector as input to the network is a good start. But what about rests? Music isn't always flowing from one note to another. A sense of rhythm is provided by the absence of notes and we have to account for that as well. The input vector was expanded to 129 whith the last dimension reserved for rests.

Representing time is quite tricky and there is some oversimplification involved. Music is written in a time signature ([a more detailed explanation](https://en.wikipedia.org/wiki/Time_signature)) in order to provide a feeling of rhythm. A typical signature like 4/4 has 4 quarter notes within a measure and so quarter notes are used a standard duration. In music21's representation a quarter note has a duration of 1.0 . In order to represent a song we need segments (or timesteps) of time with a specified length. 

For example, a music piece which consists of 4 quarter notes and a segment length of 1(quarter length duration) would be represented by 4 timesteps. Based on that, we can have shorter segments in order to represent shorter notes (like eight and sixteenth notes). A segment length of 0.25 would require 4 timesteps to represent a quarter note and 1 timestep to represent a sixteenth note.

To put these concepts together - a song is sliced into multiple timesteps which represent an equal amount of time. Within each timestep we have a 129-dimensional vector (128 possible notes and 1 rest state). Notes/rests are encoded as a 1-of-n vectors (or One-Hot encoded) and fed to the network as a series of timesteps.

### Training and output
To train a network on music data we need to look at notes sequentially and define it as a classification task. This means that at each training iteration the network is provided n timesteps to represent the context of the current note sequence. The note following that sequence is the expected output. By training over multiple songs, the network achieves an approximation of the particular style.

After training the network to have weights based on a specific style, we need an input seed. Within the project there are 4 ways of getting a seed - custom sequence of MIDI notes, random or specific file from the training set or a random sequence of notes within a musical scale. From that seed, the network can generate n timesteps (configurable) with notes and those are written to an output MIDI file at the end. At each generation step, the network outputs a probability for each of the 129 dimensions based on the notes that came before. The probabilities are then sampled to provide an output in the form of a 1-of-n representation. 

![Training sequence](https://i.imgur.com/FAHpr1W.png)

### Oversimplifications and potential future improvements
There are a few things that were deliberately oversimplified for this project. 

First, for every song in the training set only one instrument is taken. This is so because the harmony and interplay required for multiple instruments is a complex problem and even cutting-edge research struggles with it.

Another oversimplification is the lack of chords. At the moment, if the initial note parser finds a chord it only takes one note from it and ignores the rest.

Finally, if the same note is played multiple times consecutively, it is treated as one long note instead of separate instances of it.

All of these points are an excellent start for improvement!

## 2. How to run
There is a config file called config.ini which has a number of different parameters. Playing around with them would give different results and some settings might be better suited for certain styles. Each setting is described so you can experiment.

Simple way of running:
1. Configure the data_dir line in config.ini with the path to a set of MIDI files
2. Run `train.py` to train and save the learned weights

After that you can generate new MIDI files like so:
1. Configure output path in config.ini - there is a [generating] section for all related settings 
2. Configure the weights_file setting with the filename from the training step
3. Set the seed mode as you see fit
4. Run `generate.py`


## 3. Samples

Here are some samples I've generated with this using different data sets from Bach Centarl - http://www.bachcentral.com/

Sample 1:
[![Sample 1](https://img.youtube.com/vi/nigaxfN3v3w/0.jpg)](https://www.youtube.com/watch?v=nigaxfN3v3w)

Sample 2:
[![Sample 2](https://img.youtube.com/vi/nigaxfN3v3w/0.jpg)](https://www.youtube.com/watch?v=aHBQPFENRLQ)

Sample 3:
[![Sample 3](https://img.youtube.com/vi/nigaxfN3v3w/0.jpg)](https://www.youtube.com/watch?v=Bk0gf3U5K3c)