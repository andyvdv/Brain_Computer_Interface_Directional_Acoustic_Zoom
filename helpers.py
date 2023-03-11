import scipy.signal as ss
import soundfile as sf
import numpy as np
import math
import matplotlib.pyplot as plt
from package.gui_utils import load_rirs, plot_rirs
from package.general import listen_to_array

def create_micsigs(acousticScenario, nmics, speechfilenames, noisefilenames=[], duration=5):
    fs = acousticScenario.fs

    # 1) Loop over speech files
    speech_recs = []
    for filename in speechfilenames:
        data, samplerate = sf.read(f'sound_files/{filename}')
        data = data[0:samplerate*duration]
        resampled = ss.resample(data, fs*duration)
        speech_recs.append(resampled)

    # 2) Loop over noise files
    noise_recs = []
    for filename in noisefilenames:
        data, samplerate = sf.read(f'sound_files/{filename}')
        data = data[0:samplerate*duration]
        resampled = ss.resample(data, fs*duration)
        noise_recs.append(resampled)
        
    # 3) Retrieve computed RIRs
    mic_speech_recs = []
    mic_noise_recs = []
    mic_recs = None
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
                mic_rec = (np.array(speech_rec) + 0.01*np.array(noise_rec))
            else:
                mic_rec = speech_rec
            if mic_recs is None:
                mic_recs = np.empty((nmics, len(speech_recs), len(mic_rec)))
            mic_recs[j][i] = mic_rec
    # Sum the recorded signals for each source per mic
    mics_total = np.sum(mic_recs, axis=(1))
    return mics_total, mic_recs, speech_recs, noise_recs

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

def verify_parameters(acousticScenario, fs, audiofiles):
    if acousticScenario.fs != fs:
        raise Exception(f"Incorrect sample frequency in acoustic scenario. Expected {fs}, got {acousticScenario.fs}")
    if acousticScenario.RIRsAudio.shape[2] != len(audiofiles):
        raise Exception(f"Incorrect number of source audio files provided. Expected {acousticScenario.RIRsAudio.shape[2]}, got {len(audiofiles)}")
    return


def create_micsigs_week3(acousticScenario, nmics, speechfilenames, noisefilenames, duration=5):
    fs = acousticScenario.fs

    # 1) Loop over speech files
    speech_recs = []
    for filename in speechfilenames:
        data, samplerate = sf.read(f'sound_files/{filename}')
        data = data[0:samplerate*duration]
        resampled = ss.resample(data, fs*duration)
        speech_recs.append(resampled)

    # 2) Loop over noise files
    noise_recs = []
    for filename in noisefilenames:
        data, samplerate = sf.read(f'sound_files/{filename}')
        data = data[0:samplerate*duration]
        resampled = ss.resample(data, fs*duration)
        noise_recs.append(resampled)
        
    # 3) Retrieve computed RIRs
    mic_speech_recs = []
    mic_noise_recs = []
    mic_recs = None
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
                mic_rec = (np.array(speech_rec) + 0.01*np.array(noise_rec))

            else:
                mic_rec = speech_rec
            if mic_recs is None:
                mic_recs = np.empty((nmics, len(speech_recs), len(mic_rec)))
            mic_recs[j][i] = mic_rec
    # Sum the recorded signals for each source per mic
    mics_total = np.sum(mic_recs, axis=(1))
    speech_total = np.array(np.sum(mic_speech_recs,axis=(0)))
    noise_mic_total = np.array(np.sum(mic_noise_recs,axis=(0)))
    speech_recs = np.array(speech_recs)

    # PART 1 WEEK 3
    #_________________________________________________
    # 1) Compute the target audio source signal power in the first microphone.
    vadmic = abs(mics_total[0]) > np.std(mics_total[0]) * 1e-3
    mic0_power = np.var(mics_total[0, vadmic==1])

    vadspeech = abs(speech_total) > np.std(speech_total) * 1e-3
    Ps = np.var(speech_total[vadspeech==1])

    # 2) Generate white noise and scale its power to 10% of the signal power in the first microphone
    white_noise = np.random.normal(size=len(speech_total))
    noise_power = 0.1 * Ps
    scaled_noise = np.sqrt(noise_power / np.mean(white_noise**2)) * white_noise

    Pn = np.var(scaled_noise)

    # 3) If there is a noise source, add that noise to the white noise
    noise_total = []
    
    if len(noise_recs) > 0:
        noise_total = np.add(noise_mic_total, scaled_noise)

        # Compute the total noise power in that case
        Pn = np.var(noise_total)

    # 4) Compute the SNR of the first microphone
    SNR_first = 10*np.log10(Ps/Pn)

    print("SNR = {} [dB]".format(round(SNR_first,4)))
    # 5) Add the white microphone noise to each microphone signal
    for i in range(nmics):
        mics_total[i] = np.add(mics_total[i], scaled_noise)

    return mics_total, mic_recs, speech_recs, noise_recs



def create_micsigs_week4(acousticScenario, nmics, speechfilenames, noisefilenames, duration=10, startfrom=0):
    fs = acousticScenario.fs

    # 1) Loop over speech files
    speech_recs = []
    for filename in speechfilenames:
        data, samplerate = sf.read(f'sound_files/{filename}')
        data = data[startfrom:startfrom+samplerate*duration]
        print(startfrom+samplerate*duration)
        resampled = ss.resample(data, fs*duration)
        speech_recs.append(resampled)

    # 2) Loop over noise files
    noise_recs = []
    for filename in noisefilenames:
        data, samplerate = sf.read(f'sound_files/{filename}')
        data = data[startfrom:startfrom+samplerate*duration]
        resampled = ss.resample(data, fs*duration)
        noise_recs.append(resampled)
        
    # 3) Retrieve computed RIRs
    mic_speech_recs = []
    mic_noise_recs = []
    mic_recs = None
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
                mic_rec = (np.array(speech_rec) + 0.01*np.array(noise_rec))

            else:
                mic_rec = speech_rec
            if mic_recs is None:
                mic_recs = np.empty((nmics, len(speech_recs), len(mic_rec)))
            mic_recs[j][i] = mic_rec
    # Sum the recorded signals for each source per mic
    mics_total = np.sum(mic_recs, axis=(1))
    speech_total = np.array(np.sum(mic_speech_recs,axis=(0)))
    noise_mic_total = np.array(np.sum(mic_noise_recs,axis=(0)))
    speech_recs = np.array(speech_recs)

    # PART 1 WEEK 3
    #_________________________________________________
    # 1) Compute the target audio source signal power in the first microphone.
    vadmic = abs(mics_total[0]) > np.std(mics_total[0]) * 1e-3
    mic0_power = np.var(mics_total[0, vadmic==1])

    vadspeech = abs(speech_total) > np.std(speech_total) * 1e-3
    Ps = np.var(speech_total[vadspeech==1])

    # 2) Generate white noise and scale its power to 10% of the signal power in the first microphone
    white_noise = np.random.normal(size=len(speech_total))
    noise_power = 0.1 * Ps
    scaled_noise = np.sqrt(noise_power / np.mean(white_noise**2)) * white_noise

    Pn = np.var(scaled_noise)

    # 3) If there is a noise source, add that noise to the white noise
    noise_total = []
    
    if len(noise_recs) > 0:
        noise_total = np.add(noise_mic_total, scaled_noise)

        # Compute the total noise power in that case
        Pn = np.var(noise_total)

    # 4) Compute the SNR of the first microphone
    SNR_first = 10*np.log10(Ps/Pn)

    print("SNR = {} [dB]".format(round(SNR_first,4)))
    # 5) Add the white microphone noise to each microphone signal
    for i in range(nmics):
        mics_total[i] = np.add(mics_total[i], scaled_noise)

    return mics_total, mic_recs, speech_recs, noise_recs




def das_bf():
    return