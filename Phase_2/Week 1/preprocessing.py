import os
import librosa as lb
import numpy as np

# ----- Audio data -----

# Step 1 : Read audio filename.wav to the variable A.

filepath = "/Phase_1/sound_files/part1_track1_dry.wav"
audiofile, fs = lb.load(os.getcwd() + filepath)

print("Array of the audio file: ",audiofile)
print("\n")

# Step 2: Decompose the audio in 28 frequency subbands using GTF.

cf_low, cf_high = (50, 5000) # set the low and high center frequency