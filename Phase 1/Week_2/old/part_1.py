import os, sys
sys.path.append(os.getcwd())
from helpers import create_micsigs, verify_correct_fs
from package.gui_utils import load_rirs
import scipy.signal as ss
import scipy.linalg as LA
import numpy as np
import matplotlib.pyplot as plt

def music_algorithm(stft, num_sources):
    """
    Computes the MUSIC spectrum from an STFT array.
    
    Args:
        stft (ndarray): An STFT array of shape (M, N, T), where M is the number of microphones,
            N is the number of frequency bins, and T is the number of time steps.w
        num_sources (int): The number of sources to estimate.
    
    Returns:
        ndarray: A spectrum of shape (N,), containing the MUSIC power spectrum.
    """
    M, N, T = stft.shape

    # Compute the covariance matrix of the signal
    Rxx = np.zeros((M, M), dtype=np.complex128)
    for t in range(T):
        x = stft[:, :, t]
        Rxx += np.dot(x, x.conj().T)
    Rxx /= T
    # Compute the eigenvectors of the covariance matrix
    eigvals, eigvecs = np.linalg.eig(Rxx)

    # Sort the eigenvectors by their corresponding eigenvalues
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Compute the noise subspace matrix
    En = eigvecs[:, num_sources:]

    # Compute the MUSIC spectrum
    S = np.zeros((N,))
    for k in range(N):
        D = np.exp(-2j * np.pi * k * np.arange(M) / N)
        S[k] = 1 / np.dot(D.conj().T, np.dot(En, En.conj().T).dot(D))

    return S

import numpy as np

def doa_estimation(S, d, num_sources, f, theta_range=(0, 180)):
    """
    Estimates the DOA of the sources from the MUSIC spectrum and the distance between the microphones.
    
    Args:
        S (ndarray): The MUSIC spectrum.
        d (float): The distance between the microphones in meters.
        num_sources (int): The number of sources to estimate.
        theta_range (tuple): A tuple (theta_min, theta_max) containing the range of possible
            DOA angles in degrees (default: (0, 180)).
    
    Returns:
        ndarray: An array of shape (num_sources,), containing the estimated DOA angles in degrees.
    """
    c = 343.0  # speed of sound in m/s
    theta = np.linspace(theta_range[0], theta_range[1], num=S.shape[0])
    sin_theta = np.sin(np.deg2rad(theta))

    # Compute the peaks in the spectrum
    peak_idx = np.argsort(S)[-num_sources:]
    peak_theta = theta[peak_idx]

    # Refine the peak estimates using parabolic interpolation
    refined_peaks = []
    for p in peak_idx:
        if p == 0 or p == len(S) - 1:
            refined_peaks.append(theta[p])
            continue
        # Fit a parabola to the spectrum around the peak
        A = np.vstack((sin_theta[p-1:p+2]**2, sin_theta[p-1:p+2], np.ones(3))).T
        b = S[p-1:p+2]
        try:
            x = np.linalg.solve(A, b)
            refined_peak = -0.5 * x[1] / x[0]
            # Make sure the refined peak is within the range of possible DOAs
            if refined_peak >= theta_range[0] and refined_peak <= theta_range[1]:
                refined_peaks.append(refined_peak)
            else:
                refined_peaks.append(theta[p])
        except np.linalg.LinAlgError:
            refined_peaks.append(theta[p])

    # Convert DOA angles to delays and compute corresponding phase shifts
    delays = d * np.sin(np.deg2rad(refined_peaks)) / c
    phase_shifts = np.exp(-2j * np.pi * delays * f)

    # Compute the noise subspace and the MUSIC spectrum
    U = svd_vectors[:, num_sources:]
    S_noise = np.abs(np.sum(U @ phase_shifts[:, np.newaxis] * X, axis=0)) ** 2
    S_music = 1 / S_noise

    # Compute the estimated DOA angles from the refined peaks and the MUSIC spectrum
    peak_idx = np.argsort(S_music)[-num_sources:]
    estimated_doas = theta[peak_idx]

    return estimated_doas

# Load the scenario
nmics = 5
# acousticScenario = load_rirs(os.getcwd() + "/rirs/rirs_20230225_19h22m23s.pkl.gz")
# acousticScenario = load_rirs(os.getcwd() + "/rirs/rirs_20230225_19h22m40s.pkl.gz") 
acousticScenario = load_rirs(os.getcwd() + "/rirs/rirs_20230226_09h06m07s.pkl.gz")
verify_correct_fs(acousticScenario, 44100)
# Define the speech file
speechfilenames = ["speech1.wav"]
micsigs, speech, noise = create_micsigs(acousticScenario, nmics, speechfilenames, n_audio_sources=1, duration=10)
micsigs = np.array(micsigs)
print(micsigs.shape)
# Define sampling freq and FFT specs
fs = 44100
nfft = 1024
noverlap = 512
stack = None
# compute the frequency resolution of the STFT
df = fs / nfft  # Hz
# compute the frequency index corresponding to each STFT bin
freq_idx = np.arange(nfft // 2 + 1)
# convert the frequency index to frequency values in Hertz
freqs_list = freq_idx * df
# add a DC offset to obtain the center frequency of each bin
freq_bounds = freqs_list + df / 2
for i, micsig in enumerate(micsigs):
    freqs, times, zxx = ss.stft(micsig, fs=fs, nperseg=nfft, noverlap=noverlap)
    if stack is None:
        stack = np.empty((nmics, len(freqs), len(times)), dtype="complex")
    stack[i] = zxx


spectrum = music_algorithm(stack, 1)

# Estimate the DOA of the sources
theta_range = (0, 180)  # Range of possible DOA angles in degrees
doa = doa_estimation(spectrum, 0.05, 4, freqs_list)

# Print the estimated DOA angles
print('Estimated DOA angles:', np.rad2deg(doa))

# Plot the MUSIC spectrum
import matplotlib.pyplot as plt
freq_bins = np.arange(stack.shape[1])
plt.plot(freq_bins, 10 * np.log10(np.abs(spectrum)))
plt.xlabel('Frequency bin')
plt.ylabel('Power (dB)')
plt.show()

