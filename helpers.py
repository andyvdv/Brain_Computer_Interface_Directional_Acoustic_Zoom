import scipy.signal as ss
import soundfile as sf
import numpy as np
import math
import matplotlib.pyplot as plt
from package.gui_utils import load_rirs, plot_rirs
from package.general import listen_to_array

def create_micsigs(acousticScenario, nmics, speechfilenames, noisefilenames=[], n_audio_sources=2, n_noise_source=1, duration=5):
    fs = acousticScenario.fs
    print(f"The sample rate is {fs} Hz")
    # Loop over speech files
    speech_recs = []
    for filename in speechfilenames:
        data, samplerate = sf.read(f'sound_files/{filename}')
        data = data[0:samplerate*duration]
        resampled = ss.resample(data, fs*duration)
        speech_recs.append(resampled)
        # listen_to_array(resampled, fs)
    # Loop over noise files
    noise_recs = []
    for filename in noisefilenames:
        data, samplerate = sf.read(f'sound_files/{filename}')
        data = data[0:samplerate*duration]
        resampled = ss.resample(data, fs*duration)
        noise_recs.append(resampled)
        
    # Retrieve computed RIRs
    mic_recs = []
    mic_speech_recs = []
    mic_noise_recs = []
    for i in range(0, len(speech_recs)):
        for j in range(0, nmics):
            speech_rir = acousticScenario.RIRsAudio[:, j, i]
            speech = speech_recs[i]
            speech_rec = ss.fftconvolve(speech, speech_rir)
            mic_speech_recs.append(speech_rec)
            if len(noise_recs) > 0:
                noise_rir = acousticScenario.RIRsNoise[:, j, 0]
                noise = noise_recs[0]
                noise_rec = ss.fftconvolve(noise, noise_rir)
                mic_noise_recs.append(noise_rec)
                mic_rec = (list) (np.array(speech_rec) + 0.01*np.array(noise_rec))
            else:
                mic_rec = speech_rec
            mic_recs.append(mic_rec)
    return mic_recs, speech_recs, noise_recs

def TDOA_corr(acousticScenario):
    # First do ground truth sample delay
    peak1 = np.argmax(acousticScenario.RIRsAudio[:, 0, 0])
    peak2 = np.argmax(acousticScenario.RIRsAudio[:, 1, 0])
    TDOA_gt = peak2 - peak1
    print(f"Ground truth TDOA: {TDOA_gt}")
    # Crosscorrelation
    mic, speech, noise = create_micsigs(acousticScenario=acousticScenario,
                                        nmics=2,
                                        speechfilenames=["part1_track1_dry.wav"], 
                                        noisefilenames=[])
    corr = ss.correlate(mic[0], mic[1])
    print(len(mic[0]))
    print(f"Crosscorrelation maximum value: {np.argmax(corr)}")
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    ax1.plot(corr)
    ax2.stem(len(mic[1]) - TDOA_gt - 1, 1)
    plt.show()
    difference = (len(mic[1]) - TDOA_gt) - np.argmax(corr) - 1
    print(f"DIFFERENCE: {difference} samples")
    return corr

def DOA_corr(acousticScenario):
    c = 340  # Speed of sound in air (m/s)
    mic, speech, noise = create_micsigs(acousticScenario=acousticScenario,
                                        nmics=2,
                                        speechfilenames=["whitenoise_signal_1.wav"], 
                                        noisefilenames=[])
    corr = ss.correlate(mic[1], mic[0])
    sample_delay = len(mic[0]) - np.argmax(corr) - 1
    dist =  sample_delay * 1/acousticScenario.fs * c    # (samples * seconds/sample * m/second)
    DOA = np.arccos(dist/0.1) * 180 / math.pi
    print(f"Sample delay is {sample_delay} samples")
    print(f"DOA is {DOA}")