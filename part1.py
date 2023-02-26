from package.gui_utils import load_rirs, plot_rirs
import os as os

# Load the RIR and plot the scenario

acousticScenario = load_rirs(path=f"{os.getcwd()}/rirs/rirs_20230223_18h04m07s.pkl.gz")
#acousticScenario.plot_asc()


#print(acousticScenario)
#print(acousticScenario.RIRsAudio.shape)

plot_rirs(acousticScenario.RIRsAudio, acousticScenario.RIRsNoise)