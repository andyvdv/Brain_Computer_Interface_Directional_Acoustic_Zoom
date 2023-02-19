import scipy.signal as ss
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from package.gui_utils import load_rirs, plot_rirs
from package.general import listen_to_array
from helpers import create_micsigs

acousticScenario = load_rirs(path="/Users/rudi/Dropbox/MSc EE/P&D ISSP/Brain_Computer_Interface_Directional_Acoustic_Zoom/rirs/rirs_20230217_13h48m45s.pkl.gz")

mic, speech, noise = create_micsigs(acousticScenario=acousticScenario, nmics=2, speechfilenames=["speech1.wav"])

# fig, axs = plt.subplots(3, 1)
fig = plt.figure()
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2,sharex=ax1)
# ax3 = fig.add_subplot(3, 1, 2,sharex=ax1)
# ax1.plot(mic[0])
# ax2.plot(mic[1])
print(acousticScenario.RIRsAudio.shape)
ax1.plot(acousticScenario.RIRsAudio[0:1000, 0, 0])
ax2.plot(acousticScenario.RIRsAudio[0:1000, 1, 0])
plt.show()
listen_to_array(mic[0], 44100)