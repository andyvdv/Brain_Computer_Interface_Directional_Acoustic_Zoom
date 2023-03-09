import numpy as np

def das_bf(data, DOA, num_mics, d, fs):
    """
    Implements Delay-and-Sum beamforming for a microphone array.
    
    Parameters:
    data (ndarray): Input signal from the microphone array (shape: (N_samples, N_sensors))
    DOA (float): Direction of arrival of the desired signal (in degrees)
    num_mics (int): Number of microphones in the array
    d (float): Distance between microphones (in meters)
    fs (float): Sampling frequency (in Hz)
    
    Returns:
    output (ndarray): Output signal after DAS beamforming (shape: (N_samples,))
    """
    c = 343  # Speed of sound in air
    # Generate microphone positions
    mic_pos = np.zeros((num_mics, 3))
    mic_pos[:, 0] = np.arange(num_mics) * d

    # Generate steering vector for DOA
    theta = np.radians(DOA)
    steer_vec = np.exp(-1j * 2 * np.pi * fs * mic_pos[:, 0] * np.sin(theta) / c)

    # Initialize delay-and-sum beamformer weights
    weights = np.ones((num_mics,)) / num_mics
    
    # DAS-BF
    # Apply beamformer weights
    bf_data = np.multiply(data, weights)
    # Apply delay to each microphone
    delays = np.zeros((num_mics,))
    for i in range(num_mics):
        delays[i] = -1 * mic_pos[i, 0] * np.sin(theta) / c
    # Create delay matrix
    delay_mat = np.zeros((num_mics, data.shape[0]))
    for i in range(num_mics):
        delay_mat[i, :] = np.roll(bf_data[:, i], int(delays[i] * fs))
    # Sum delayed signals
    output = np.sum(delay_mat, axis=0)
    
    return output