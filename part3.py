import scipy.signal as ss
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from package.gui_utils import load_rirs, plot_rirs
from package.general import listen_to_array
from helpers import create_micsigs
import os

def TDOA_corr_part3(acousticScenario, audiofile):
    # First do ground truth sample delay
    peak1 = np.argmax(acousticScenario.RIRsAudio[:, 0, 0])
    peak2 = np.argmax(acousticScenario.RIRsAudio[:, 1, 0])
    TDOA_gt = peak2 - peak1
    print(f"Ground truth TDOA: {TDOA_gt}")
    # Crosscorrelation
    mic, speech, noise = create_micsigs(acousticScenario=acousticScenario,
                                        nmics=2,
                                        speechfilenames=[audiofile], 
                                        noisefilenames=[])
    corr = ss.correlate(mic[0], mic[1])
    print(len(mic[0]))
    print(f"Crosscorrelation maximum value: {np.argmax(corr)}")

    # Plots
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0.5)
    ax1.plot(corr)
    ax1.grid(True)
    ax1.set_ylabel('Cross-correlation amplitude')
    ax1.set_title('Cross-correlation function')
    ax2.stem(len(mic[1]) - TDOA_gt - 1, 1)
    ax2.grid(True)
    ax2.set_xlabel('Sample index')
    ax2.set_ylabel('Peak amplitude')
    ax2.set_title('Expected maximum peak according to the ground truth')
    plt.show()
    
    # Print the estimation error (samples difference)
    difference = (len(mic[1]) - TDOA_gt) - np.argmax(corr) - 1
    print(f"DIFFERENCE: {difference} samples")
    return corr

def exercise3():
    # Test the file in the non-reverberant scenario
    audiofile = "whitenoise_signal_1.wav"
    acousticScenario = load_rirs(path=f"{os.getcwd()}/rirs/part3_135deg.pkl.gz")
    corr = TDOA_corr_part3(acousticScenario,audiofile)
    print(np.argmax(corr))

def exercise4():
    # Increase the reverberation time and check how much reverberation is allowed before the TDOA estimation starts to show errors
    audiofile = "whitenoise_signal_1.wav"
    acousticScenario = load_rirs(path=f"{os.getcwd()}/rirs/part3_135deg_0.75rev.pkl.gz")
    corr = TDOA_corr_part3(acousticScenario,audiofile)
    print(np.argmax(corr))

def exercise5_6():
    # Re-simulate the non-reverb scenario but with a speech signal
    audiofile = "part1_track1_dry.wav"
    acousticScenario = load_rirs(path=f"{os.getcwd()}/rirs/part3_135deg_0.21rev.pkl.gz") # Can increase the reverberation time here
    corr = TDOA_corr_part3(acousticScenario,audiofile)
    print(np.argmax(corr))



if __name__ == "__main__":
    #exercise3()
    #exercise4()
    exercise5_6()


