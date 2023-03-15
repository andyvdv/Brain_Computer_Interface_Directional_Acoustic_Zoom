from Week_3.das_beamformer import das_bf
import numpy as np
import scipy
import scipy.signal as ss
from Week_3.FIR import FIR_filter

def gsc_td(speech, noise, DOA, num_mics, d, fs, L=1024, mu=0.1, vad=None):
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
    f.update(X.T, y, vad)
    gsc_output = y - np.sum(f.filter(X.T), axis=(0))

    return gsc_output
