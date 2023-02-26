import os, sys
sys.path.append(os.getcwd())
from helpers import create_micsigs, verify_correct_fs
from package.gui_utils import load_rirs
import scipy.signal as ss
import scipy.linalg as LA
import numpy as np
import matplotlib.pyplot as plt

def music(S, M, N, d):
    """
    MUSIC algorithm for direction-of-arrival (DOA) estimation.

    Args:
        S: Complex-valued steering matrix of shape (M, N, L), where M is the number of sensors, N is
           the number of frequency bins, and L is the number of snapshots.
        M: Number of sources.
        N: Number of frequency bins.
        d: Spacing between sensors.

    Returns:
        doa: Estimated DOA in radians, with shape (M,).
    """
    print(S.shape)
    # Compute the sample covariance matrix
    R = np.mean(np.conj(S.transpose(1, 0, 2)) @ S, axis=-1)
    print(R.shape)
    # Compute the eigendecomposition of the covariance matrix
    eigvals, eigvecs = np.linalg.eig(R)

    # Sort the eigenvalues in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Compute the noise subspace
    En = eigvecs[:, M:]
    # Compute the spatial spectrum
    theta = np.linspace(-np.pi/2, np.pi/2, 180)
    P = np.zeros_like(theta)
    for i, th in enumerate(theta):
        a = np.exp(-1j * 2 * np.pi * d * np.sin(th) * np.arange(M))
        P[i] = 1 / np.linalg.norm(En.conj().transpose() @ a)**2

    # Estimate the DOA using the peaks of the spatial spectrum
    doa = theta[np.argsort(P)[-M:]]
    return doa


# Load the scenario
nmics = 5
acousticScenario = load_rirs(os.getcwd() + "/rirs/rirs_20230225_18h48m24s.pkl.gz")
verify_correct_fs(acousticScenario, 44100)
# Define the speech file
speechfilenames = ["speech1.wav"]
# speechfilenames = ["whitenoise_signal_1.wav"]
micsigs, speech, noise = create_micsigs(acousticScenario, nmics, speechfilenames, n_audio_sources=1, duration=10)

stack = None
freqs = []
times = []
fig, axs = plt.subplots(nmics, 1, sharex="col")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
x = []
fs = 44100
nfft = 1024
noverlap = 512
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
    # plt.pcolormesh(t, f, np.abs(zxx))]
    # ax = axs[i]
    # ax.set_ylim([0, fs/2])
    # ax.pcolormesh(times, freqs, np.abs(zxx))
    # plt.show()

# doa = music(stack, 5, 513, 0.1)
M = 5
d = 0.05
power = np.square(np.absolute(stack))
mean = power.mean(axis=(0,2))
x = power.mean(axis=(2))
max_bin = np.argmax(mean)
print(f"Max frequency bin is bin {max_bin}, which corresponds to {round(freq_bounds[max_bin], 2)} Hz")
Y = x[:, max_bin]
print(Y)

Y = np.asmatrix(Y)
R = (Y.transpose() * Y.transpose().getH()) 
# Compute the eigendecomposition of the covariance matrix
eigvals, eigvecs = np.linalg.eig(R)

# Sort the eigenvalues in descending order
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# Compute the noise subspace
En = eigvecs[:, 1:]
print(En)
# Compute the spatial spectrum
theta = np.linspace(-np.pi/2, np.pi/2, 180)
P = np.zeros_like(theta)
for i, th in enumerate(theta):
    a = np.exp(-1j * 2 * np.pi * d * np.sin(th) * np.arange(M))
    P[i] = 1 / np.linalg.norm(En.conj().transpose() @ a)**2

# Estimate the DOA using the peaks of the spatial spectrum
doa = theta[np.argsort(P)[-M:]]
print(np.rad2deg(doa))

# power = np.square(np.absolute(stack))
# mean = power.mean(axis=(0,2))
# x = power.mean(axis=(2))
# max_bin = np.argmax(mean)
# print(f"Max frequency bin is bin {max_bin}, which corresponds to {round(freq_bounds[max_bin], 2)} Hz")
# Y = x[:, max_bin]
# print(Y)

# Y = np.asmatrix(Y)
# Ryy = (Y.transpose() * Y.transpose().getH()) 
# print(Ryy)
# D, E = LA.eig(Ryy)
# idx = D.argsort()[::-1]
# lmbd = D[idx]# Vector of sorted eigenvalues
# E = E[:, idx]# Sort eigenvectors accordingly
# En = E[:, nmics:len(E)]# Noise eigenvectors (ASSUMPTION: M IS KNOWN)w
# print(E.shape)
# E = np.asmatrix(E)
# p_w = 1 / (E * E.getH())

# print(p_w)
# # print(np.sort(stack.mean(axis=(0,2))))