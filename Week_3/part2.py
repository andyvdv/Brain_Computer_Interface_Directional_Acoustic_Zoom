import os, sys
sys.path.append(os.getcwd())
from helpers import create_micsigs_week3, verify_parameters, create_micsigs
from package.gui_utils import load_rirs
from package.general import listen_to_array
import scipy.signal as ss
import scipy.linalg as LA
import numpy as np
from Week_2.music import *
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from week3_helpers import create_micsigs_wk3, calculate_snr
from das_beamformer import das_bf

def part_2():
    scenarioPath = "/rirs/Week_2/Part_2_1/45_135.pkl.gz"
    acousticScenario = load_rirs(os.getcwd() + scenarioPath)
    fs = 44100
    # Get AS parameters
    nsources = acousticScenario.RIRsAudio.shape[2]
    nmics = acousticScenario.nMicsPerArray
    d = acousticScenario.distBwMics
    # Define the speech and noise files
    speechfiles = ["speech1.wav", "speech2.wav"]
    noisefiles= []
    # Verify parameters
    verify_parameters(acousticScenario, fs, speechfiles)
    print("---- Without any noise source ----")
    mics_total, mic_recs, speech_recs, noise_recs = create_micsigs_wk3(acousticScenario=acousticScenario, 
                                                                        nmics=nmics, 
                                                                        speechfilenames=speechfiles,
                                                                        noisefilenames=noisefiles)
    # Stack the STFTs of the microphone signals
    nfft = 1024
    noverlap = 512
    S, freqs_list = stack_stfts(mics_total, acousticScenario.fs, nfft, noverlap)
    # Define the angles to commpute the pseudospectrum for
    thetas = np.arange(0, 180, 0.5)
    # Compute the MUSIC pseudospectrum and DOAs
    spectrum, doas = music_wideband(S, nmics, nsources, freqs_list, d, thetas)

    # plot_pseudspectrum(thetas, spectrum, "Figure")

    # Steer to source that is closest to 90°
    # Parameters
    num_mics = acousticScenario.nMicsPerArray   # Number of microphones
    d = acousticScenario.distBwMics   # Distance between adjacent microphones
    fs = acousticScenario.fs   # Sampling frequency
    c = 343   # Speed of sound
    speech_total = np.sum(speech_recs, axis=(2))
    noise_total = np.sum(noise_recs, axis=(2))
    # Find DOA nearest to 90°
    index = np.abs(doas - 90).argmin()
    DOA = doas[index]
    print(f"DOA steering towards: {DOA}")

    speechDAS = das_bf(speech_total, DOA, nmics, d, fs)
    noiseDAS = das_bf(noise_total, DOA, nmics, d, fs)
    
    DASout = speechDAS + noiseDAS
    
    SNRin = calculate_snr(speech_total, noise_total)
    SNRoutDAS = calculate_snr(speechDAS, noiseDAS)
    print(f"SNRin: {SNRin}\nSNRoutDAS: {SNRoutDAS} ")

    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(speech_total[:, 0] + noise_total[:, 0])
    axs[0].set_title("Mic 0 Recording")
    axs[1].plot(DASout)
    axs[1].set_title("DAS BF Output")
    
    # Compare mic 0 recording to the DAS output
    # listen_to_array(speech_total[:, 0] + noise_total[:, 0], fs)
    # listen_to_array(DASout, fs)

if __name__ == "__main__":
    part_2()
    plt.show()