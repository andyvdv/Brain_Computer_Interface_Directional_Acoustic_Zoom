from brian2 import Hz, kHz
from brian2hears import Sound, erbspace, Gammatone, Filterbank
from scipy.signal import butter, filtfilt
import librosa as lb
import os, sys
sys.path.append(os.getcwd())
import numpy as np
from envelope import EnvelopeFromGammatoneFilterbank
import matplotlib.pyplot as plt
# ----- Audio data -----

method = "LR"

# Step 1 : Read audio filename.wav to the variable A.
filepath = "/sound_files/part1_track1_dry.wav"
audio_signal, sr = lb.load(os.getcwd() + filepath, sr=44100)

# Cut the audio signal to 5 seconds
audio_signal = audio_signal[:5*sr]

# Convert the numpy array to a Sound object
audio_sound = Sound(audio_signal, samplerate=sr*Hz)

num_filters = 28
center_freqs = np.linspace(50*Hz, 5*kHz, num_filters)
gammatone_filterbank = Gammatone(audio_sound, center_freqs)

subbands = gammatone_filterbank.process()

compressed_subbands = np.abs(subbands)**0.6

combined_envelope = np.sum(compressed_subbands, axis=1)

print(f"Shape of combined_envelope: {combined_envelope.shape}")

# Choose the frequency range based on the method
low_freq = 1
high_freq = 9  if method == "LR" else 32

# Design a Butterworth bandpass filter
order = 4
nyquist = 0.5 * sr
low = low_freq / nyquist
high = high_freq / nyquist
b, a = butter(order, [low, high], btype='band')

# Apply the filter
filtered_envelope = filtfilt(b, a, combined_envelope)

# Fix non-finite values before resampling
filtered_envelope = np.nan_to_num(filtered_envelope)

# Choose the downsampling rate based on the method
downsampling_rate = 20  if method == "LR" else 64

resampled_envelope = lb.resample(filtered_envelope, orig_sr=sr, target_sr=downsampling_rate)


# Set up the figure with subplots
fig, axs = plt.subplots(3, 2, figsize=(12, 12))

# Plot the original audio signal
axs[0, 0].plot(audio_signal)
axs[0, 0].set_title("Original Audio Signal")

# Plot the compressed subbands
for i, subband in enumerate(compressed_subbands):
    axs[0, 1].plot(subband, label=f"Subband {i+1}")
axs[0, 1].set_title("Compressed Subbands")

# Plot the combined envelope
axs[1, 0].plot(combined_envelope)
axs[1, 0].set_title("Combined Envelope")

# Plot the filtered envelope
axs[1, 1].plot(filtered_envelope)
axs[1, 1].set_title("Filtered Envelope")

# Plot the resampled envelope
axs[2, 0].plot(resampled_envelope)
axs[2, 0].set_title("Resampled Envelope")

# Hide the unused subplot
axs[2, 1].axis('off')

# Set the layout and display the plot
plt.tight_layout()
plt.show()