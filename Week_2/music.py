import scipy.signal as ss
import scipy.linalg as LA
import numpy as np

def music_narrowband(S, nmics, nsources, freqs_list, d):
    """
    Evaluate the pseudospectrum at the frequency bin with the highest average power
    """
    max_bin = find_max_bin(S)
    S_mb = S[:, max_bin, :]
    Ryy = (np.asmatrix(S_mb) @ np.asmatrix(S_mb).getH()) / S_mb.shape[1]
    # Compute the eigenvectors of the covariance matrix
    eigvals, eigvecs = np.linalg.eig(Ryy)
    M, N, T = S.shape
    # Sort the eigenvectors by their corresponding eigenvalues
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    U_N = eigvecs[:, nsources:]
    max_freq = freqs_list[max_bin]
    # Calculate the pseudospectrum over all thetas
    thetas = np.arange(-90, 90, 0.5)
    P = np.empty(len(thetas))
    for i, theta in enumerate(thetas):
        a = np.asmatrix(steering_vector(theta, M, d, max_freq))
        squared_norm = a.getH() @ np.asmatrix(U_N) @ np.asmatrix(U_N).getH() @ a
        P[i] = np.abs((1 / squared_norm.item()))
    thetas += 90
    return P, thetas

def steering_vector(theta, M, d, f):
    # Speed of sound in m/s
    c = 343
    # Convert angle from degrees to radians
    theta = np.deg2rad(theta)
    # Compute the wave number
    k = 2 * np.pi * f / c
    # Compute the distances from the source to each microphone
    mics = np.arange(M) * d
    distances = mics * np.sin(theta)
    # Compute the phase shifts for each microphone
    phases = np.exp(-1j * k * distances)
    # Construct the steering vector
    A = phases.reshape(-1, 1)
    return A

def stack_stfts(micsigs, fs, nfft, noverlap):
    stack = None
    for i, micsig in enumerate(micsigs):
        freqs, times, zxx = ss.stft(micsig, fs=fs, nperseg=nfft, noverlap=noverlap)
        if stack is None:
            stack = np.empty((len(micsigs), len(freqs), len(times)), dtype="complex")
        stack[i] = zxx

    df = fs / nfft  # Hz
    # compute the frequency index corresponding to each STFT bin
    freq_idx = np.arange(nfft // 2 + 1)
    # convert the frequency index to frequency values in Hertz
    freqs_list = freq_idx * df
    # add a DC offset to obtain the center frequency of each bin
    freq_bounds = freqs_list + df / 2
    return stack, freqs_list

def find_max_bin(S):
    P = np.square(np.absolute(S))
    P_avg = P.mean(axis=(0,2))
    return np.argmax(P_avg)
