import os
import librosa as lb
import numpy as np
import gammatone.gtgram as gtg

# ----- Audio data -----

# Step 1 : Read audio filename.wav to the variable A.

filepath = "/Phase_1/sound_files/part1_track1_dry.wav"
audiofile, fs = lb.load(os.getcwd() + filepath, sr=44100)

print(f"Sampling rate: {fs} Hz")
print("Array of the audio file: ",audiofile)
print("\n")

# Step 2: Decompose the audio in 28 frequency subbands using GTF.
# Define the filter bank parameters
num_subbands = 28
low_frequency = 50
high_frequency = 5000
center_frequencies = np.linspace(low_frequency, high_frequency, num_subbands)

# Gamma-tone filter bank
gtgram = gtg.gtgram(audiofile, fs, window_time=0.025, hop_time=0.01, channels=center_frequencies, f_min=low_frequency)

# Transpose the result to have subbands as rows and time frames as columns
gtgram = gtgram.T

print("Shape of the gamma-tone filtered subband signals:", gtgram.shape)