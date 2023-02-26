import scipy.signal as ss
import soundfile as sf
import numpy as np
import math
import matplotlib.pyplot as plt
from package.gui_utils import load_rirs, plot_rirs
from package.general import listen_to_array

def create_micsigs(acousticScenario, nmics, speechfilenames, noisefilenames=[], n_audio_sources=2, n_noise_source=1, duration=5):
    fs = acousticScenario.fs
    # print(f"The sample rate is {fs} Hz")
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

def TDOA_corr(acousticScenario, nmics, mic1, mic2, audiosources):
    # First do ground truth sample delay
    peak1 = np.argmax(acousticScenario.RIRsAudio[:, mic1, 0])
    peak2 = np.argmax(acousticScenario.RIRsAudio[:, mic2, 0])
    TDOA_gt = peak2 - peak1
    # print(f"Ground truth TDOA: {TDOA_gt}")
    # Crosscorrelation
    mic, speech, noise = create_micsigs(acousticScenario=acousticScenario,
                                        nmics=nmics,
                                        speechfilenames=audiosources, 
                                        noisefilenames=[],
                                        duration = 2)
    mic1_tot = mic[mic1] + mic[mic1 + nmics]
    mic2_tot = mic[mic2] + mic[mic2 + nmics]
    corr = ss.correlate(mic1_tot, mic2_tot)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(5, 1, 1)
    # ax2 = fig.add_subplot(5, 1, 2, sharex=ax1)
    # ax3 = fig.add_subplot(5, 1, 3, sharex=ax1)
    # ax4 = fig.add_subplot(5, 1, 4, sharex=ax1)
    # ax5 = fig.add_subplot(5, 1, 5, sharex=ax1)
    # ax1.plot(mic1_tot)
    # ax2.plot(mic2_tot)
    # ax3.plot(speech[0])
    # ax4.plot(speech[1])
    # ax5.plot(corr)
    # plt.show()
    # difference = (len(mic[1]) - TDOA_gt) - np.argmax(corr) - 1
    # print(f"DIFFERENCE: {difference} samples")
    sample_diff1 = len(mic1_tot) - np.argmax(corr)
    corr[np.argmax(corr)] = 0
    sample_diff2 = len(mic1_tot) - np.argmax(corr)
    return [sample_diff1, sample_diff2]

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

def verify_correct_fs(acousticScenario, fs):
    if acousticScenario.fs != fs:
        raise Exception("Incorrect sample frequency in acoustic scenario!!!")
    return
