import scipy.signal as ss
from package.gui_utils import load_rirs, plot_rirs

def create_micsigs(acousticScenario, nmics=3, n_audio_sources=2, n_noise_source=1):
    fs = acousticScenario.fs
    print(f"Sampling frequency is {fs}")
    speechfilenames = ["speech1.wav", "speech2.wav"]
    noisefilenames = ["whitenoise_signal1.wav"]
    rec_sig_duration = 10   # Seconds
    for filename in speechfilenames:
        with open(f"sound_files/{filename}", "r") as file:
            resampled = ss.resample(file, fs)

acousticScenario = load_rirs(path="/Users/rudi/Dropbox/MSc EE/P&D ISSP/Brain_Computer_Interface_Directional_Acoustic_Zoom/rirs/rirs_20230215_17h09m20s.pkl.gz")

create_micsigs(acousticScenario=acousticScenario)