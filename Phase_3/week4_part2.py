import os, sys
sys.path.append(os.getcwd())
from helpers import create_micsigs_week3, verify_parameters, create_micsigs, DOA_corr
from package.gui_utils import load_rirs
from package.general import listen_to_array
import scipy.signal as ss
import scipy.linalg as LA
import numpy as np
from Week_2.music import stack_stfts, music_wideband, plot_pseudspectrum
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from Week_3.week3_helpers import create_micsigs_wk3, das_bf, calculate_snr, create_micsigs_wk4
from Week_3.gsc import gsc_td, gsc_fd
from Week_2.music import music_wideband, compute_steering_vector
import scipy.signal as ss
import scipy
import math, json

def part2(reverb=False):
    rev = 0
    if reverb == True:
        rev=1
    scenarioPath = "/rirs/Week_4/part2/rirs_part2_1_{}rev.pkl.gz".format(rev)
    AS = load_rirs(os.getcwd() + scenarioPath)
    nmics = AS.nMicsPerArray
    d = AS.distBwMics
    Q = AS.RIRsAudio.shape[2]
    fs = AS.fs
    # Define the speech and noise files
    speechfiles = ["part1_track1_dry.wav","part1_track2_dry.wav"] # First for audio source, second for interfering source
    # Verify parameters
    verify_parameters(AS, fs, speechfiles)
    
    
    # Combine all the signals from all the scenario in the no reverberation case

    # DO IT ONCE FOR the first RIR
    mics_total_norev, mic_recs_norev, speech_recs_norev = create_micsigs_wk4(acousticScenario=AS, 
                                                                            nmics=nmics, 
                                                                            speechfilenames=speechfiles,
                                                                            noisefilenames=[],
                                                                            duration=10)

    print("Coordinates of the two currently used sources :", AS.audioCoords)
    nperseg = 1024
    noverlap = nperseg//2
    nfft=2048
    sqrt_hann = np.sqrt(ss.windows.hann(nperseg, "periodic"))

    mics_total_summed = np.sum(speech_recs_norev, axis=(2))
    S, freq_list = stack_stfts(mics_total_summed.T, fs=fs, nperseg=nperseg, nfft=nfft, noverlap=noverlap, window=sqrt_hann)
    thetas = np.arange(0, 180, 0.5)
    spectrum, doas = music_wideband(S, nmics=nmics, nsources=Q, freqs_list=freq_list, d=d, angles=thetas)
    DOA = doas[0]

    print("DOA = " + str(DOA) + "°")

    # NOW DO IT FOR THE REST OF THE RIRs
    for i in range(2,6):
        AS = load_rirs(os.getcwd() + "/rirs/Week_4/part2/rirs_part2_{0}_{1}rev.pkl.gz".format(i,rev))
        print("Coordinates of the two currently used sources :", AS.audioCoords)
        
        mics_total_summed = np.sum(speech_recs_norev, axis=(2))
        S, freq_list = stack_stfts(mics_total_summed.T, fs=fs, nperseg=nperseg, nfft=nfft, noverlap=noverlap, window=sqrt_hann)
        thetas = np.arange(0, 180, 0.5)
        spectrum, doas = music_wideband(S, nmics=nmics, nsources=Q, freqs_list=freq_list, d=d, angles=thetas)
        DOA = doas[0]

        print("DOA = " + str(DOA) + "°")

        mics_total, mic_recs, speech_recs= create_micsigs_wk4(acousticScenario=AS, 
                                                                            nmics=nmics, 
                                                                            speechfilenames=speechfiles,
                                                                            noisefilenames=[],
                                                                            duration=10,
                                                                            startfrom=fs*(i-1)*10)
        
        mics_total_norev = np.concatenate((mics_total_norev,mics_total),axis=1)
        mic_recs_norev = np.concatenate((mic_recs_norev,mic_recs),axis=1)
        speech_recs_norev = np.concatenate((speech_recs_norev,speech_recs),axis=1)


    mics_total_summed = np.sum(speech_recs_norev, axis=(2))
    S, freq_list = stack_stfts(mics_total_summed.T, fs=fs, nperseg=nperseg, nfft=nfft, noverlap=noverlap, window=sqrt_hann)
    thetas = np.arange(0, 180, 0.5)
    spectrum, doas = music_wideband(S, nmics=nmics, nsources=Q, freqs_list=freq_list, d=d, angles=thetas)
    DOA = doas[0]

    print("DOA = " + str(DOA) + "°")

    listen_to_array(mics_total_norev[1],fs)
    plt.plot(mics_total_norev[1])


    plt.show()

    
    return 



if __name__ == "__main__":
    part2(reverb=True)