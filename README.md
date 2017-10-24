# Bach To The Future

This project was my dissertation thesis for my last year at uni. Originally it was trained on music from J.S. Bach (hence the name of the project).

It is an LSTM neural network which generates new MIDI files with music. It has to be trained with MIDI files first.

Simple way of running:
1. Configure config.ini where the data_dir is with the path to a some set of MIDI files
2. Tweak other parameters as you wish
3. Run train.py to train and save the learned weights
4. Run generate.py

The following Pythin packages are required to run - music21, Keras, numPy

Sometime in the near future this readme will be improved to include more detail about how the system works, but feel free to look around and try it out.
