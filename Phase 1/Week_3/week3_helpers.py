import scipy.signal as ss
import soundfile as sf
import numpy as np
import math
import matplotlib.pyplot as plt
from Week_2.music import *
from package.gui_utils import load_rirs, plot_rirs
from package.general import listen_to_array

def create_micsigs_wk3(acousticScenario, nmics, speechfilenames, noisefilenames, duration=5):
    fs = acousticScenario.fs

    # 1) Loop over speech files
    speech_data = []
    for filename in speechfilenames:
        data, samplerate = sf.read(f'sound_files/{filename}')
        data = data[0:samplerate*duration]
        resampled = ss.resample(data, fs*duration)
        speech_data.append(resampled)

    # 2) Loop over noise files
    noise_data = []
    for filename in noisefilenames:
        data, samplerate = sf.read(f'sound_files/{filename}')
        data = data[0:samplerate*duration]
        resampled = ss.resample(data, fs*duration)
        noise_data.append(resampled)
        
    # 3) Retrieve computed RIRs
    mic_speech_recs = None
    mic_noise_recs = []
    mic_recs = None
    # Loop over speech sources
    for i in range(0, len(speech_data)):
        # Loop over microphones
        for j in range(0, nmics):
            speech_rir = acousticScenario.RIRsAudio[:, j, i]
            speech = speech_data[i]
            speech_rec = ss.fftconvolve(speech, speech_rir)
            mic_speech_recs = np.empty((len(speech_rec), nmics, len(speech_data))) if mic_speech_recs is None else mic_speech_recs
            mic_speech_recs[:, j, i] = speech_rec
            if len(noise_data) > 0:
                noise_rir = acousticScenario.RIRsNoise[:, j, 0]
                noise = noise_data[0]
                noise_rec = ss.fftconvolve(noise, noise_rir)
                mic_noise_recs.append(noise_rec)
                mic_rec = (np.array(speech_rec) + 0.01*np.array(noise_rec))
            else:
                mic_rec = speech_rec
            if mic_recs is None:
                mic_recs = np.empty((nmics, len(speech_data), len(mic_rec)))
            mic_recs[j][i] = mic_rec
    # Sum the recorded signals for each source per mic
    mics_total = np.sum(mic_recs, axis=(1))
    noise_matrix = np.empty(mic_speech_recs.shape)
    # Loop over audio sources
    for i in range(len(speech_data)):
        mic_0_power, vad = find_power(mic_speech_recs[:, 0, i])
        for j in range(nmics):
            noise_matrix[:, j, i] = get_noise(noise_matrix.shape[0], mic_0_power * 0.1, vad)
    mic_speech_recs_total = np.sum(mic_speech_recs, axis=(1))
    noise_matrix_total = np.sum(noise_matrix, axis=(1))
    # Calculate first mic received signal
    if len(speechfilenames) > 1:
        mic_0_rec = mic_speech_recs[:, 0, 0] + mic_speech_recs[:, 0, 1] + noise_matrix[:, 0, 0] + noise_matrix[:, 0, 1]
        mic_0_total_snr = calculate_snr(mic_speech_recs[:, 0, 0] + mic_speech_recs[:, 0, 1], noise_matrix[:, 0, 0] + noise_matrix[:, 0, 1])
        print(f"Total SNR in mic 0: {mic_0_total_snr}")
    # listen_to_array(mic_0_rec, 44100)
    return mics_total, mic_recs, mic_speech_recs, noise_matrix

def find_power(signal):
    vad = abs(signal[:]) > np.std(signal[:]) * 1e-3
    signal_nonzero = signal[vad==1]
    return np.var(signal_nonzero), vad

def get_noise(length, power, vad):
    noise = np.random.normal(0, np.sqrt(power), length)
    return noise

def calculate_snr(signal, noise):
    """
    Calculate the signal-to-noise ratio (SNR) between two signals.
    Assumes that the signals have the same length.

    Parameters:
        signal (array-like): The signal.
        noise (array-like): The noise.

    Returns:
        float: The SNR between the signal and the noise.
    """
    # Calculate the power of the signal
    signal_power = np.mean(np.abs(signal)**2)

    # Calculate the power of the noise
    noise_power = np.mean(np.abs(noise)**2)

    # Calculate the SNR
    snr = 10 * np.log10(signal_power / noise_power)

    return snr

def das_bf(data, DOA, num_mics, d, fs):
    c = 343
    # Generate microphone positions
    mic_pos = np.zeros((num_mics, 3))
    mic_pos[:, 0] = np.arange(num_mics) * d

    # Generate steering vector for DOA
    theta = np.radians(DOA)
    steer_vec = np.exp(-1j * 2 * np.pi * fs * mic_pos[:, 0] * np.sin(theta) / c)

    # Initialize delay-and-sum beamformer weights
    weights = np.ones((num_mics,)) / num_mics
    # DAS-BF
    # Apply beamformer weights
    bf_data = np.multiply(data, weights)
    # Apply delay to each microphone
    delays = np.zeros((num_mics,))
    for i in range(num_mics):
        delays[i] = -1 * mic_pos[i, 0] * np.sin(theta) / c
    # Create delay matrix
    delay_mat = np.zeros((num_mics, data.shape[0]))
    for i in range(num_mics):
        delay_mat[i, :] = np.roll(bf_data[:, i], int(delays[i] * fs))
    # Sum delayed signals
    output = np.sum(delay_mat, axis=0)
    return output

def create_micsigs_wk4(acousticScenario, nmics, speechfilenames, noisefilenames, duration=10, startfrom=0):
    fs = acousticScenario.fs

    # 1) Loop over speech files
    speech_data = []
    for filename in speechfilenames:
        data, samplerate = sf.read(f'sound_files/{filename}')
        data = data[startfrom:startfrom+samplerate*duration]
        resampled = ss.resample(data, fs*duration)
        speech_data.append(resampled)

    # 2) Loop over noise files
    noise_data = []
    for filename in noisefilenames:
        data, samplerate = sf.read(f'sound_files/{filename}')
        data = data[startfrom:startfrom+samplerate*duration]
        resampled = ss.resample(data, fs*duration)
        noise_data.append(resampled)
        
    # 3) Retrieve computed RIRs
    mic_speech_recs = None
    mic_noise_recs = []
    mic_recs = None
    # Loop over speech sources
    for i in range(0, len(speech_data)):
        # Loop over microphones
        for j in range(0, nmics):
            speech_rir = acousticScenario.RIRsAudio[:, j, i]
            speech = speech_data[i]
            speech_rec = ss.fftconvolve(speech, speech_rir)
            mic_speech_recs = np.empty((len(speech_rec), nmics, len(speech_data))) if mic_speech_recs is None else mic_speech_recs
            mic_speech_recs[:, j, i] = speech_rec
            if len(noise_data) > 0:
                noise_rir = acousticScenario.RIRsNoise[:, j, 0]
                noise = noise_data[0]
                noise_rec = ss.fftconvolve(noise, noise_rir)
                mic_noise_recs.append(noise_rec)
                mic_rec = (np.array(speech_rec) + 0.01*np.array(noise_rec))
            else:
                mic_rec = speech_rec
            if mic_recs is None:
                mic_recs = np.empty((nmics, len(speech_data), len(mic_rec)))
            mic_recs[j][i] = mic_rec
    # Sum the recorded signals for each source per mic
    mics_total = np.sum(mic_recs, axis=(1))
    noise_matrix = np.empty(mic_speech_recs.shape)
    # Loop over audio sources
    for i in range(len(speech_data)):
        mic_0_power, vad = find_power(mic_speech_recs[:, 0, i])
        for j in range(nmics):
            noise_matrix[:, j, i] = get_noise(noise_matrix.shape[0], mic_0_power * 0.1, vad)
    mic_speech_recs_total = np.sum(mic_speech_recs, axis=(1))
    noise_matrix_total = np.sum(noise_matrix, axis=(1))
    # Calculate first mic received signal
    if len(speechfilenames) > 1:
        mic_0_rec = mic_speech_recs[:, 0, 0] + mic_speech_recs[:, 0, 1] + noise_matrix[:, 0, 0] + noise_matrix[:, 0, 1]
        mic_0_total_snr = calculate_snr(mic_speech_recs[:, 0, 0] + mic_speech_recs[:, 0, 1], noise_matrix[:, 0, 0] + noise_matrix[:, 0, 1])
        print(f"Total SNR in mic 0: {mic_0_total_snr}")
    # listen_to_array(mic_0_rec, 44100)
    return mics_total, mic_recs, mic_speech_recs, noise_matrix