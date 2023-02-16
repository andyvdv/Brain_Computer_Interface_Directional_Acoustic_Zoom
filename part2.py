import scipy.signal as ss
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from package.gui_utils import load_rirs, plot_rirs
from package.general import listen_to_array

def create_micsigs(acousticScenario, nmics=3, n_audio_sources=2, n_noise_source=1, duration=5):
    fs = acousticScenario.fs
    speechfilenames = ["speech1.wav", "speech2.wav"] #TODO check if file exists
    noisefilenames = ["whitenoise_signal_1.wav"]
    # Loop over speech files
    speech_recs = []
    for filename in speechfilenames:
        data, samplerate = sf.read(f'sound_files/{filename}')
        data = data[0:samplerate*duration]
        resampled = ss.resample(data, fs*duration)
        speech_recs.append(resampled)
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
            mic_rec = (list) (np.array(speech_rec) + np.array(noise_rec))
            mic_recs.append(mic_rec)
    # print(mic_recs)
    return mic_recs, speech_recs, noise_recs

acousticScenario = load_rirs(path="/Users/rudi/Dropbox/MSc EE/P&D ISSP/Brain_Computer_Interface_Directional_Acoustic_Zoom/rirs/rirs_20230216_16h10m08s.pkl.gz")

mic, _, _ = create_micsigs(acousticScenario=acousticScenario)

fig, axs = plt.subplots(2, 1)
axs[0].plot(mic[0])
axs[1].plot(mic)