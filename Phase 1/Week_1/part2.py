import scipy.signal as ss
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from package.gui_utils import load_rirs, plot_rirs
from package.general import listen_to_array
from helpers import create_micsigs
import os

acousticScenario = load_rirs(path=f"{os.getcwd()}/rirs/rirs_20230225_13h52m06s.pkl.gz")
print(acousticScenario)
mic, speech, noise = create_micsigs(acousticScenario=acousticScenario, nmics=2, speechfilenames=["speech1.wav"])


plt.plot(acousticScenario.RIRsAudio[0:1000, 0, 0], label = 'Microphone 2')
plt.plot(acousticScenario.RIRsAudio[0:1000, 1, 0], label='Microphone 1')
plt.grid()
plt.title("RIRs of the two microphones with d = 1 cm, fs = 44.1 kHz")
plt.xlabel('Samples')
plt.ylabel('RIR')
plt.legend()
plt.show()

#listen_to_array(mic[0], 44100)