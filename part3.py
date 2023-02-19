import scipy.signal as ss
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from package.gui_utils import load_rirs, plot_rirs
from package.general import listen_to_array
from helpers import TDOA_corr, create_micsigs

acousticScenario = load_rirs(path="/Users/rudi/Dropbox/MSc EE/P&D ISSP/Brain_Computer_Interface_Directional_Acoustic_Zoom/rirs/rirs_20230217_14h34m33s.pkl.gz")
corr = TDOA_corr(acousticScenario)

print(np.argmax(corr))