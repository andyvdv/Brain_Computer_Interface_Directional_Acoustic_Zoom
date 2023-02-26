import scipy.signal as ss
import scipy.linalg as LA
import numpy as np

def music_spectrum(stfts, p, fs):
    """
    Calculates the MUSIC spectrum for a given STFT array and number of sources.

    Parameters:
        stfts (ndarray): STFT array of shape (num_mics, freq_bins, time_frames).
        num_sources (int): Number of sources to localize.

    Returns:
        ndarray: MUSIC spectrum of shape (freq_bins,)
    """
    # Compute covariance matrix from STFT array
     # Compute covariance matrix from STFT array
    X = np.transpose(stfts, (2, 0, 1))  # Shape: (time_frames, num_mics, freq_bins)
    Rxx = np.matmul(X, np.transpose(np.conj(X), (0, 2, 1)))  # Shape: (num_mics, freq_bins, freq_bins)
    Rxx = Rxx.mean(axis=0)
    print(LA.ishermitian(Rxx))
    # Perform eigenvalue decomposition of the covariance matrix
     # Perform eigenvalue decomposition of the covariance matrix
    eigenValues, eigenVectors = LA.eig(Rxx)
    print(eigenVectors.shape)
    # Sort eigenvectors by eigenvalue in descending order
    sort_indices = np.argsort(eigenValues)[::-1]
    eigenValues = eigenValues[sort_indices]
    sRaeigenVectors = eigenVectors[:, sort_indices]

    # # Construct noise subspace matrix
    # En = eigvecs[:, num_sources:]
    # En_h = np.asmatrix(En).getH()
    # print(En_h)
    # # Compute MUSIC spectrum
    # freq_bins = stfts.shape[1]
    # music_spectrum = np.zeros((freq_bins,))
    # for k in range(freq_bins):
    #     identity_matrix = np.eye(stfts.shape[0], dtype=np.complex128)
    #     identity_matrix = np.expand_dims(identity_matrix, axis=0)
    #     identity_matrix = np.repeat(identity_matrix, stfts.shape[1], axis=0)
    #     Vk = np.fft.fft(identity_matrix * stfts[:, k:k+1, :], axis=1)
    #     S = np.sum(np.abs(np.matmul(np.transpose(np.conj(noise_subspace)), Vk))**2, axis=0)
    #     music_spectrum[k] = 1 / np.sum(S)

    # return music_spectrum

    signal_eigen = eigenVectors[0:p]
    noise_eigen = eigenVectors[p:len(eigenVectors)]     # noise subspace
    print("Signal\n", signal_eigen)
    print("Noise\n", noise_eigen)
    N = stfts.shape[1]
    spectrum = []
    num_slice = fs * N

    for f_int in range(0, fs):
        sum1 = 0
        frequencyVector = np.zeros(len(noise_eigen[0]), dtype=np.complex_)
        f = float(f_int) / num_slice

        for i in range(0,len(noise_eigen[0])):
            # create a frequency vector with e to the 2pi *k *f and taking the conjugate of the each component
            frequencyVector[i] = np.conjugate(complex(np.cos(2 * np.pi * i * f), np.sin(2 * np.pi * i * f)))
            # print f, i, np.pi, np.cos(2 * np.pi * i * f)

        # print frequencyVector

        for u in range(0,len(noise_eigen)):
            # sum the squared dot product of each noise eigenvector and frequency vector.
            sum1 += (abs(np.dot(np.asarray(frequencyVector).transpose(), np.asarray( noise_eigen[u]) )))**2
        spectrum.append(1/sum1)
        print(f_int)
    return spectrum