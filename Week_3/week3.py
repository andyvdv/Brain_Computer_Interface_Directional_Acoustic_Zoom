import os, sys
sys.path.append(os.getcwd())
from helpers import create_micsigs_week3, verify_parameters
from package.gui_utils import load_rirs
import scipy.signal as ss
import scipy.linalg as LA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def part1_no_noise_source():
    scenarioPath = "/rirs/Week_3/rirs_part1_no_noise.pkl.gz"
    acousticScenario = load_rirs(os.getcwd() + scenarioPath)
    nsources = acousticScenario.RIRsAudio.shape[2]
    nmics = acousticScenario.nMicsPerArray
    print("---- Without any noise source ----")
    mics_total, mic_recs, speech_recs, noise_recs = create_micsigs_week3(acousticScenario=acousticScenario, nmics=nmics, speechfilenames=["speech1.wav","speech2.wav"])
   
    return

def part1_noise_source():
    scenarioPath = "/rirs/Week_3/rirs_part1_noise.pkl.gz"
    acousticScenario = load_rirs(os.getcwd() + scenarioPath)
    nsources = acousticScenario.RIRsAudio.shape[2]
    nmics = acousticScenario.nMicsPerArray
    print("---- With babble noise source ----")
    mics_total, mic_recs, speech_recs, noise_recs = create_micsigs_week3(acousticScenario=acousticScenario, nmics=nmics, speechfilenames=["speech1.wav","speech2.wav"],noisefilenames=["Babble_noise1.wav"])

    return

def part1_closer():
    scenarioPath = "/rirs/Week_3/rirs_part1_closernoise.pkl.gz"
    acousticScenario = load_rirs(os.getcwd() + scenarioPath)
    nsources = acousticScenario.RIRsAudio.shape[2]
    nmics = acousticScenario.nMicsPerArray
    print("---- Babble noise source but closer ----")
    mics_total, mic_recs, speech_recs, noise_recs = create_micsigs_week3(acousticScenario=acousticScenario, nmics=nmics, speechfilenames=["speech1.wav","speech2.wav"],noisefilenames=["Babble_noise1.wav"])

    return



if __name__ == "__main__":
    part1_no_noise_source()
    part1_noise_source()
    part1_closer()
