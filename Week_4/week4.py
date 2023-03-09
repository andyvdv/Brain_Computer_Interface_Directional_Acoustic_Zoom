import os, sys
sys.path.append(os.getcwd())
from helpers import create_micsigs_week3, verify_parameters, create_micsigs
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

def week_4():
    scenarioPath = "/rirs/Week_2/Part_2_1/45_135.pkl.gz"
    AS = load_rirs(os.getcwd() + scenarioPath)
    nmics = AS.nMicsPerArray
    d = AS.distBwMics
    Q = AS.RIRsAudio.shape[2]
    fs = AS.fs
    # Define the speech and noise files
    speechfiles = ["speech1.wav", "speech2.wav"]
    noisefiles= []
    # Verify parameters
    verify_parameters(AS, fs, speechfiles)
    print("---- Without any noise source ----")
    mics_total, mic_recs, speech_recs, noise_recs = create_micsigs_wk3(acousticScenario=AS, 
                                                                        nmics=nmics, 
                                                                        speechfilenames=speechfiles,
                                                                        noisefilenames=noisefiles)
    
    
    nfft = 1024
    noverlap = 512
    # Generate the square-root Hanning window
    sqrt_hann = np.sqrt(ss.windows.hann(nfft, "periodic"))
    mics_total = np.sum(speech_recs + noise_recs, axis=(2))
    S, freq_list = stack_stfts(mics_total.T, fs=fs, nfft=nfft, noverlap=noverlap, window=sqrt_hann)
    
    thetas = np.arange(0, 180, 0.5)
    spectrum, doas = music_wideband(S, nmics=nmics, nsources=Q, freqs_list=freq_list, d=d, angles=thetas)
    plot_pseudspectrum(thetas, spectrum, "MUSIC Spectrum")
    print(f"The estimated DOAs are: {doas}")
    # Find DOA nearest to 90°
    doa_index = np.abs(doas - 90).argmin()
    DOA = doas[doa_index]
    print(f"Steering towards source with DOA of {DOA}°")

    A = np.zeros((nmics, len(thetas), len(freq_list)), dtype="complex")
    for i, theta in enumerate(thetas):
        for j, freq in enumerate(freq_list):
            A[:, i, j] = np.reshape(compute_steering_vector(theta, nmics, d, freq), (5,))
    
    H = np.zeros_like(A)
    for i in range(A.shape[1]):
        for j in range(A.shape[2]):
            a = A[:, i, j]
            H[:, i, j] = a / A[0, i, j]
    print(H)
    # out = gsc_fd(S, H, DOA, doa_index, thetas, freq_list, sqrt_hann)
    out = []
    
    speech_total = np.sum(speech_recs, axis=(2))
    noise_total = np.sum(noise_recs, axis=(2))
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(speech_total[:, 0] + noise_total[:, 0])
    axs[0].set_title("Mic 0 Recording")
    axs[1].plot(out)
    axs[1].set_title("FD GSC Output")
    # a1 = compute_steering_vector(90, nmics, d, 100)
    # print(a1)s
    # listen_to_array(out, fs)
    
if __name__ == "__main__":
    week_4()
    plt.show()