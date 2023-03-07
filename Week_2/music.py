import scipy.signal as ss
import scipy.linalg as LA
import numpy as np
import matplotlib.pyplot as plt

def music_wideband(S, nmics, nsources, freqs_list, d, angles):
    """
    Evaluate the MUSIC pseudospectrum as the geometric average of pseudospectrums over all frequency bins
    """
    M, N, T = S.shape
    P = None
    pwr = 1 / ((N // 2) - 1)
    for k in range(0, N//2 - 1):
        Pk, _ = music_narrowband(S, nmics, nsources, freqs_list, d, k+2, angles)
        Pk = np.power(Pk, pwr)    
        if P is None:
            P = np.empty((N//2 - 1, len(Pk))) 
        P[k] = Pk
    P = np.prod(P, axis=(0))
    peaks = ss.find_peaks(P)[0]
    print(peaks)
    n_largest_peaks = P[peaks]
    doas = angles[peaks]
    return P, doas

def music_narrowband(S, nmics, nsources, freqs_list, d, bin_index, angles):
    """
    Evaluate the MUSIC pseudospectrum for a specific frequency bin
    """
    M, N, T = S.shape
    thetas = angles.copy()
    thetas -= 90 # Convert thetas to new coordinate system, measured from mic array normal
    Sk = S[:, bin_index, :]
    Ryy = (np.asmatrix(Sk) @ np.asmatrix(Sk).getH()) / Sk.shape[1]
    # Compute the eigenvectors of the covariance matrix
    eigvals, eigvecs = np.linalg.eig(Ryy)
    # Sort the eigenvectors by their corresponding eigenvalues
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Find the noise subspace
    U_N = eigvecs[:, nsources:]
    max_freq = freqs_list[bin_index]

    # Calculate the pseudospectrum over all thetas
    P = np.empty(len(thetas), dtype=np.float64)
    for i, theta in enumerate(thetas):
        a = np.asmatrix(compute_steering_vector(theta, M, d, max_freq))
        squared_norm = a.getH() @ np.asmatrix(U_N) @ np.asmatrix(U_N).getH() @ a
        P[i] = np.abs((1 / squared_norm.item()))
    thetas += 90
    doas = thetas[ss.find_peaks(P)[0]]
    return P, doas

def compute_steering_vector(theta, M, d, f):
    """
    Compute the steering vector, a, for a mic array of M mics a distance d apart
    at frequency f and angle theta
    """
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

def plot_pseudspectrum(a, s, figure_title, window_title="Figure", normalise=True, stems=[]):
    # Normalise the spectrum to a range of [0,100] if normalise is True
    s = (s / s[np.argmax(s)]) * 100 if normalise else s
    # Define the figure
    fig = plt.figure(num=window_title)
    fig.suptitle(figure_title, fontweight='bold')
    ax = fig.add_subplot(1,1,1)
    ax.plot(a, s, 'b', label=f"P(θ)")
    for stem in stems:
        ax.stem(stem, max(s), markerfmt='', linefmt='r--', label=f"{stem}°")
    
    ax.set_ylabel('Power (Normalised)' if normalise else 'Power', fontweight='bold', fontsize='12')
    ax.set_xlabel("Angle (θ)", fontweight='bold', fontsize='12')
    ax.legend()

def plot_multiple(X, Y, title, normalise=True, stems=None, extra=None, cols=2, rows=None):
    rows = len(Y) // 2 if rows == None else rows
    fig, axs = plt.subplots(rows, cols, sharex=True, sharey=normalise)
    fig.subplots_adjust(top=0.9)
    for i in range(len(Y)):
        a = X[i]
        s = (Y[i] / Y[i][np.argmax(Y[i])]) * 100 if normalise else Y[i]
        if cols == 1:
            axis = axs[i]
        elif cols == 2:
            axis = axs[i // 2][i % 2]
        axis.plot(a, s, 'b', label=f"P(θ)")
        if stems is not None:
            stem = stems[i]
            for st in stem:
                axis.stem(st, max(s), markerfmt='', linefmt='r--', label=f"{st}°")
        axis.legend(title=extra[i] if extra is not None else None)
    fig.suptitle(title, fontweight='bold')
    fig.text(0.02, 0.5, 'Power (Normalised)' if normalise else 'Power', va='center', rotation='vertical', fontweight='bold', fontsize='12')
    fig.text(0.5, 0.04, 'Angle (θ)', ha='center', va='center', fontweight='bold', fontsize='12')
