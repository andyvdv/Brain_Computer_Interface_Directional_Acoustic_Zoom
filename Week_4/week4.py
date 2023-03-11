import os, sys
sys.path.append(os.getcwd())
from helpers import create_micsigs_week3, verify_parameters, create_micsigs, create_micsigs_week4
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
    print(H[:, 100, 100])
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
    

def week_4_2():
    AS_RIRS = load_rirs(os.getcwd() + "/rirs/Week_4/lots_of_rirs.pkl.gz")
    # AS.plot_asc()
    
    angles = np.zeros((AS_RIRS.nMicsPerArray, AS_RIRS.RIRsAudio.shape[2]))
    for i in range(AS_RIRS.RIRsAudio.shape[2]):
        source_pos = AS_RIRS.audioCoords[i]
        for j in range(AS_RIRS.nMicsPerArray):
            mic_pos = AS_RIRS.micsCoords[j]
            # Calculate the angle between the two points
            dx = mic_pos[0] - source_pos[0]
            dy = mic_pos[1] - source_pos[1]
            angle = math.atan2(dy, dx)
            # Convert the angle to degrees and shift the range to [0, 360)
            angle_degrees = math.degrees(angle)
            angle_degrees += 90
            # angle_degrees = (angle_degrees + 360) % 360
            angles[j, i] = angle_degrees
    AS = load_rirs(os.getcwd() + "/rirs/Week_4/single_source.pkl.gz")
    nmics = AS.nMicsPerArray
    d = AS.distBwMics
    fs = AS.fs 
    Q = AS.RIRsAudio.shape[2]
    speechfiles = ["speech1.wav"]
    noisefiles= []
    mics_total, mic_recs, speech_recs, noise_recs = create_micsigs_wk3(acousticScenario=AS, 
                                                                        nmics=nmics, 
                                                                        speechfilenames=speechfiles,
                                                                        noisefilenames=noisefiles,
                                                                        duration=5)
    nperseg = 1024
    noverlap = nperseg//2
    nfft=2048
    # Generate the square-root Hanning window
    sqrt_hann = np.sqrt(ss.windows.hann(nperseg, "periodic"))
    mics_total = np.sum(speech_recs + noise_recs, axis=(2))
    print(f"Mics total shape: {mics_total.shape}")
    S, freq_list = stack_stfts(mics_total.T, fs=fs, nperseg=nperseg, nfft=nfft, noverlap=noverlap, window=sqrt_hann)
    
    thetas = np.arange(0, 180, 0.5)
    spectrum, doas = music_wideband(S, nmics=nmics, nsources=Q, freqs_list=freq_list, d=d, angles=thetas)
    
    M, n_freqs, n_times = S.shape
    DOA = doas[0]
    print(f"Estimated DOAs: {doas}")
    # Find RIRs
    index = np.argmin(np.abs(angles - DOA), axis=1)
    print(f"RIR index: {index}")
    rirs = np.zeros((5, 22050))
    for i in range(rirs.shape[0]):
        rirs[i, :] = AS_RIRS.RIRsAudio[:, i, index[i]]
    print(f"AS RIRs shape: {AS_RIRS.RIRsAudio.shape}")
    print(f"STFT shape: {S.shape}")
    print(f"RIRS shape: {rirs.shape}")
    DFTS = np.fft.fft(rirs, axis=1)
    print(f"DFTs shape: {DFTS.shape}")
    print(f"DFTS: {DFTS}")
    mic_0_dft = DFTS[0, :].copy()
    for i in range(DFTS.shape[0]):
        DFTS[i, :] /= mic_0_dft
    H = DFTS
    print(f"H shape: {H.shape}")
    print(f"H: {H}")
    W = np.zeros_like(H, dtype="complex")
    for k in range(W.shape[1]):
        h = np.asmatrix(H[:, k]).T
        h_H = h.getH()
        W[:, k] = (h / (h_H @ h)).flatten() 
    print(f"W shape: {W.shape}")
    
    
    fas = np.zeros((S.shape[1], S.shape[2]), dtype="complex")
    # Loop over frequencies
    for i in range(n_freqs):
        # Loop over times
        for j in range(n_times):
            samples = S[:, i, j]
            freq = int(freq_list[i]) - 1
            w_fas = W[:, freq]
            out = samples * w_fas
            fas[i, j] = np.sum(out)
    print(f"FAS shape: {fas.shape}")
    recs_total = speech_recs + noise_recs
    recs_total = np.sum(recs_total, axis=(2))
    
    # Compute the blocking matrix for each frequency
    B = np.zeros((n_freqs, nmics-1, nmics), dtype=np.complex128)
    for i in range(n_freqs):
        h = np.asmatrix(H[:, i]).T
        b = np.asmatrix(scipy.linalg.null_space(h.getH())).getH()
        B[i, :, :] = b
        print("B matrix is not all zero") if not np.allclose(b @ h, 0) else None
    print(f"B shape: {B.shape}")
    mic_0_stft = S[0]
    print(f"Mic 0 STFT shape: {mic_0_stft.shape}")
    
    output = np.zeros_like(fas)
    # Pass STFTs through adaptive filter after multiplication with blocking matrix
    for i in range(n_freqs):
        f = ComplexNLMS(4, 0.1, 1e-6)
        for j in range(n_times):
            s = S[:, i, j]
            block_mat = B[i, :, :].T
            x = np.dot(s, block_mat)
            filter_out = f.update(x, fas[i, j])
            output[i, j] -= filter_out
            # y = np.dot(w, x)
            # error = fas[i, j] - y
            # w_prev = w
            # w = w + mu * np.conj(x) * error / np.sum(np.abs(x)**2)
            # output[i, j] = error
            
            # # Check for convergence of weights
            # delta_w = w - w_prev
            # if np.linalg.norm(delta_w) < eps:
            #     print(f"Converged at frequency {freq_list[i]} and time {j}")
            #     break
            
            
        # print(f"Weights: {w}")
    # window = np.sqrt(np.hanning(n_freqs))
    # freq_signal_windowed = output * window[:, np.newaxis]
    times, fas_reconstructed = ss.istft(fas, fs=fs, nfft=nfft, nperseg=nperseg, noverlap=noverlap, window=sqrt_hann)
    times, signal_reconstructed = ss.istft(output, fs=fs, nfft=nfft, nperseg=nperseg, noverlap=noverlap, window=sqrt_hann)
    times, mic_0_reconstructed = ss.istft(S[0, :, :], fs=fs, nfft=nfft, nperseg=nperseg, noverlap=noverlap, window=sqrt_hann)
    
    fig, axs = plt.subplots(4,1)
    axs[0].plot(speech_recs[:, 0])
    axs[0].set_title("First mic recording")
    axs[1].plot(times, mic_0_reconstructed)
    axs[1].set_title("First mic reconstructed")
    axs[2].plot(times, fas_reconstructed)
    axs[2].set_title("FAS reconstructed")
    axs[3].plot(times, signal_reconstructed)
    axs[3].set_title("Total reconstructed signal")
    
    # listen_to_array(mics_total[:, 0], fs)
    # listen_to_array(mic_0_reconstructed, fs)
    listen_to_array(fas_reconstructed, fs)
    listen_to_array(signal_reconstructed, fs)
    
class ComplexNLMS:
    def __init__(self, num_channels, step_size, reg_factor):
        self.num_channels = num_channels
        self.step_size = step_size
        self.reg_factor = reg_factor
        self.weights = np.zeros((num_channels,), dtype=np.complex128)
    
    def update(self, x, d):
        
        # compute error
        y = np.dot(self.weights, x)
        e = d - y
        # print(f"d: {d}")
        # print(f"e: {e}")
        # update weights
        norm_x = np.linalg.norm(x)
        norm_x_squared = norm_x * norm_x
        self.weights += self.step_size * np.conj(x) * e / (norm_x_squared + self.reg_factor)
        # print(f"weights: {self.weights}")
        return y
    
        
if __name__ == "__main__":
    # week_4()
    week_4_2()
    plt.show()