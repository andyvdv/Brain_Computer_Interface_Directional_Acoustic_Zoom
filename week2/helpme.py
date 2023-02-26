import os, sys
sys.path.append(os.getcwd())
from helpers import create_micsigs, verify_correct_fs
from package.gui_utils import load_rirs
from music import *
import scipy.signal as ss
import scipy.linalg as LA
import numpy as np
import matplotlib.pyplot as plt

# Load the scenario
nmics = 5
acousticScenario = load_rirs(os.getcwd() + "/rirs/Week_2/90deg1source.pkl.gz")
verify_correct_fs(acousticScenario, 44100)
# Define the speech file
speechfilenames = ["speech1.wav"]
# speechfilenames = ["whitenoise_signal_1.wav"]
micsigs, speech, noise = create_micsigs(acousticScenario, nmics, speechfilenames, n_audio_sources=1, duration=10)

S, freqs_list = stack_stfts(micsigs, acousticScenario.fs, 1024, 512)
print(f"Stack of STFTs: S (M, N, T) has size {S.shape}")
max_bin = find_max_bin(S)
print(f"Frequency bin with highest average power is bin {max_bin}")
spectrum, angles = music_narrowband(S, nmics, 1, freqs_list, 0.05)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(angles, spectrum)
ax.set_xlabel("Theta (Degrees)")
plt.show()