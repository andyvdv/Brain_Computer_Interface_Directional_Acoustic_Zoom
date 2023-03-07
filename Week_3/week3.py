import os, sys
sys.path.append(os.getcwd())
from helpers import create_micsigs_week3, verify_parameters, create_micsigs
from package.gui_utils import load_rirs
import scipy.signal as ss
import scipy.linalg as LA
import numpy as np
from Week_2.music import *
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

def part2():
    scenarioPath = "/rirs/Week_3/rirs_part1_closernoise.pkl.gz"
    acousticScenario = load_rirs(os.getcwd() + scenarioPath)
    Q = acousticScenario.RIRsAudio.shape[2] + acousticScenario.RIRsNoise.shape[2] # Number of localized sources
    nmics = acousticScenario.numMics
    speechfilenames = ["speech1.wav"]
    d = acousticScenario.distBwMics
    wideband = True

    # Verify the sampling frequency is correct and the right amount of sound files are passed
    #verify_parameters(acousticScenario, 44100, speechfilenames)
    # Obtain the microphone signals
    micsigs, mic_recs, speech_recs, noise_recs = create_micsigs(acousticScenario, nmics, speechfilenames, duration=10)
    # Stack the STFTs of the microphone signals
    nfft = 1024
    noverlap = 512
    S, freqs_list = stack_stfts(micsigs, acousticScenario.fs, nfft, noverlap)
    # Define the angles to commpute the pseudospectrum for
    thetas = np.arange(0, 180, 0.5)
    # Compute the MUSIC pseudospectrum and DOAs
    spectrum, doas = music_wideband(S, nmics, Q, freqs_list, d, thetas) if wideband else music_narrowband(S, nmics, Q, freqs_list, d, find_max_bin(S), thetas)
    print(micsigs)
    print(doas)
    
    

if __name__ == "__main__":
    part1_no_noise_source()
    part1_noise_source()
    part1_closer()
    part2()