import os, sys
sys.path.append(os.getcwd())
from helpers import create_micsigs, verify_correct_fs
from package.gui_utils import load_rirs
import scipy.signal as ss
import scipy.linalg as LA
import numpy as np
import matplotlib.pyplot as plt
from music import music_spectrum

# Load the scenario
nmics = 5
# acousticScenario = load_rirs(os.getcwd() + "/rirs/rirs_20230225_19h22m23s.pkl.gz")
# acousticScenario = load_rirs(os.getcwd() + "/rirs/rirs_20230225_19h22m40s.pkl.gz") 
acousticScenario = load_rirs(os.getcwd() + "/rirs/rirs_20230226_09h06m07s.pkl.gz")
verify_correct_fs(acousticScenario, 44100)
# Define the speech file
speechfilenames = ["speech1.wav"]
micsigs, speech, noise = create_micsigs(acousticScenario, nmics, speechfilenames, n_audio_sources=1, duration=10)
micsigs = np.array(micsigs)
# Assume micsigs is a numpy array of shape (num_mics, num_samples)
num_mics, num_samples = micsigs.shape

# Set STFT parameters
nfft = 1024   # FFT size
hop = nfft//2  # Hop size
win = 'hann'  # Window function
fs = acousticScenario.fs
# Compute STFT for each microphone signal
stfts = []
for i in range(num_mics):
    f, t, Zxx = ss.stft(micsigs[i], window=win, nperseg=nfft, noverlap=hop)
    stfts.append(Zxx)

# Convert STFTs to numpy array of shape (num_mics, freq_bins, time_frames)
stfts = np.array(stfts)
print(stfts.shape)
spectrum = music_spectrum(stfts, 3, fs)
print(spectrum)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(spectrum)
plt.show()