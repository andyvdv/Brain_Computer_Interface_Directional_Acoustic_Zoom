from Week_3.das_beamformer import das_bf
import numpy as np
import scipy
import scipy.signal as ss
from Week_3.FIR import FIR_filter

def gsc_td(speech, noise, DOA, num_mics, d, fs, L=1024, mu=0.1):
    """
    Generalized Sidelobe Canceller (GSC) with DAS beamforming and Griffiths-Jim blocking matrix
    
    Parameters:
    x (ndarray): Input signal from the sensor array (shape: (N_sensors, N_samples) or (N_samples, N_sensors))
    DOA (float): Direction of arrival (in degrees)
    num_mics (int): Number of microphones in the array
    d (float): Distance between microphones (in meters)
    fs (float): Sampling frequency (in Hz)
    L (int): Filter length for adaptive filter matrix
    mu (float): Step size for NLMS algorithm
    
    Returns:
    y (ndarray): Output signal after GSC processing (shape: (N_samples,))
    """
    x = speech + noise
    # Ensure input signal has shape (N_samples, N_sensors)
    if x.shape[0] < x.shape[1]:
        x = x.T
    print(x.shape)
    # Set filter spacing
    delta = L // 2
    
    # Perform DAS beamforming
    y = das_bf(x, DOA, num_mics, d, fs)
    
    # Compute reference signal
    ref = np.zeros_like(y)
    for n in range(L // 2, len(x)):
        ref[n] = np.sum(y[n-delta:n+delta+1]) / (2*delta+1)
    
    # Compute error signal
    e = x[:, 0] - ref
    
    # Create the Griffiths-Jim blocking matrix
    B = np.zeros((num_mics-1, num_mics))
    for i in range(num_mics-1):
        B[i, 0] = 1
        B[i, i+1] = -1
    # Compute the input to the LMS filter
    X = np.dot(x, B.T)
    
    # Implement filter
    L = 1024
    mu = 0.1
    f = FIR_filter(4, L, mu)
    gsc_output = np.zeros_like(y)
    
    # Update the filter weights
    f.update(X.T, ref)
    gsc_output = y - np.sum(f.filter(X.T), axis=(0))

    return gsc_output

def gsc_fd(S, H, DOA, doa_index, thetas, freq_list, window):
    print(S.shape)
    w_fas = np.zeros_like(H)
    B = []
    for j, freq in enumerate(freq_list):
        h = np.asmatrix(H[:, doa_index, j]).T
        h_H = h.getH()
        w_fas[:, doa_index, j] = (h / (h_H @ h)).flatten()
        # Compute the null space of h
        b = scipy.linalg.null_space(h.T)
        B.append(b)
        if not np.allclose(b.T.dot(h), 0):
            raise Exception("B matrix is not correct")
            
    # GSC for each frequency bin
    GSCout_fd = np.zeros((S.shape[1], S.shape[2]), dtype="complex128")
    for k, freq in enumerate(freq_list):
        w = w_fas[:, doa_index, k]
        freq_dom_signal = S[:, k, :]
        y = freq_dom_signal.T @ w
        blocking_mat = B[k]
        filter_input = freq_dom_signal.T @ blocking_mat
        
        # Implement filters
        L = 128
        mu = 0.1
        f = FIR_filter(4, L, mu)
        
        # Update the filter weights
        # f.update(np.abs(filter_input.T), np.abs(y))
        # out = f.filter(np.abs(filter_input))
        Y, W = complex_multichannel_nlms(filter_input, y, 1, 0.1, 1e-10)
        GSCout_fd[k, :] = y - Y.flatten()
    
    # print(GSCout_fd.shape)
    # Define the overlap ratio (50% in this case)
    overlap_ratio = 0.5

    # Get the number of samples in each window
    window = np.sqrt(ss.windows.hann(475, "periodic"))
    window_length = len(window)

    # Calculate the hop size (number of samples to shift the window)
    hop_size = int(window_length * (1 - overlap_ratio))

    stft_signals = GSCout_fd
    # Get the number of windows
    num_windows = stft_signals.shape[1]

    # Calculate the length of the output signal
    signal_length = (num_windows - 1) * hop_size + window_length

    # Create an empty array for the output signal
    signal_reconstructed = np.zeros(signal_length)

    # Compute the ISTFT for each windowed segment of the frequency domain signal
    for i in range(num_windows):
        # Compute the inverse Fourier transform for the current windowed segment
        signal_segment = np.real(np.fft.ifft(stft_signals[i]))
        
        # Add the current segment to the output signal using the overlap-add technique
        start_idx = i * hop_size
        end_idx = start_idx + window_length
        signal_reconstructed[start_idx:end_idx] += signal_segment * window

    # Normalize the output signal
    signal_reconstructed /= np.max(signal_reconstructed)
    
    return signal_reconstructed

def complex_multichannel_nlms(x, d, M, mu, eps):
    """
    Apply the complex-valued multi-channel normalized least-mean-squares (NLMS)
    algorithm to a set of complex-valued input signals.

    Parameters
    ----------
    x : array_like, shape (num_samples, num_channels)
        An array containing the complex-valued input signals, where `num_samples`
        is the number of samples and `num_channels` is the number of channels.
    d : array_like, shape (num_samples,)
        An array containing the complex-valued desired signal.
    M : int
        The length of the filter.
    mu : float
        The step size parameter for the NLMS algorithm.
    eps : float
        A small positive constant to avoid division by zero.

    Returns
    -------
    y : array_like, shape (num_samples,)
        The output signal of the filter.
    w : array_like, shape (M, num_channels)
        The final weight vector of the filter.

    """
    num_samples, num_channels = x.shape

    # Initialize the weight vector
    w = np.zeros((M, num_channels), dtype=np.complex128)

    # Initialize the output signal
    y = np.zeros(num_samples, dtype=np.complex128)

    # Apply the NLMS algorithm to each sample
    for n in range(M - 1, num_samples):
        # Extract the current input vector
        x_n = x[n - M + 1:n + 1, :]

        # Compute the filter output
        R = np.dot(x_n.conj().T, np.atleast_2d(w[:, -1]).T)
        y[n] = np.sum(R)

        # Compute the error signal
        e_n = d[n] - y[n]

        # Compute the step size factor
        alpha_n = mu / (eps + np.dot(x_n.conj().T, x_n))

        # Update the weight vector
        w_new = w[:, -1].reshape(-1, 1) + alpha_n * e_n * x_n.conj().T
        w = np.hstack((w, w_new.T))

    return y, w

def compute_fas_bf(H, doa_index, freq):
    # Access pre-filled look-up table to obtain FAS BF vector for given DOA and frequency
    # Vector of multipliers is obtained from measured or generated steering vector for DOA and frequency
    # Blocking matrix is computed such that it is orthogonal to the steering vector
    # Return FAS BF vector
    h = np.asmatrix(H[:, doa_index, j]).T
    h_H = h.getH()
    w_fas = (h / (h_H @ h)).flatten()