import os, sys
sys.path.append(os.getcwd())
from helpers import create_micsigs_week3, verify_parameters, create_micsigs, DOA_corr, create_micsigs_week4
from package.gui_utils import load_rirs
from package.general import listen_to_array
import scipy.signal as ss
import scipy.linalg as LA
import numpy as np
from Week_2.music import stack_stfts, music_wideband, plot_pseudspectrum
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from Week_3.week3_helpers import create_micsigs_wk3, das_bf, calculate_snr
from Week_3.gsc import gsc_td, gsc_fd
from Week_2.music import music_wideband, compute_steering_vector
import scipy.signal as ss
import scipy
import math, json

def part2():
    scenarioPath = "/rirs/Week_4/part2/rirs_part2_1_0rev.pkl.gz"
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
    mics_total_norev, mic_recs_norev, speech_recs_norev, noise_recs_norev = create_micsigs_week4(acousticScenario=AS, 
                                                                            nmics=nmics, 
                                                                            speechfilenames=speechfiles,
                                                                            noisefilenames=[],
                                                                            duration=10)

    for i in range(2,6):
        AS = load_rirs(os.getcwd() + "/rirs/Week_4/part2/rirs_part2_{}_0rev.pkl.gz".format(i))
        print(AS.audioCoords)
        mics_total, mic_recs, speech_recs, noise_recs = create_micsigs_week4(acousticScenario=AS, 
                                                                            nmics=nmics, 
                                                                            speechfilenames=speechfiles,
                                                                            noisefilenames=[],
                                                                            duration=10,
                                                                            startfrom=fs*(i-1)*10)
        
        mics_total_norev = np.concatenate((mics_total_norev,mics_total),axis=1)
        mic_recs_norev = np.concatenate((mic_recs_norev,mic_recs),axis=1)
        speech_recs_norev = np.concatenate((speech_recs_norev,speech_recs),axis=1)



    listen_to_array(mics_total_norev[1],fs)
    plt.plot(mics_total_norev[1])
    
    plt.show()

    
    return 



if __name__ == "__main__":
    part2()