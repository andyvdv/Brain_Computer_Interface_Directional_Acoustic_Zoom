import numpy as np

class FIR_filter:
    """
    A class for implementing a Finite Impulse Response (FIR) filter.

    Attributes:
    - num_channels (int): The number of channels in the input signal.
    - num_taps (int): The number of filter taps.
    - step_size (float): The step size of the filter.
    - leaky_factor (float): The factor by which the weights are multiplied at each update.

    Methods:
    - filter(input_signal): Filters an input signal with the FIR filter.
    - update(X, d): Updates the weights of the FIR filter using the filtered signal and a desired output signal.
    """

    def __init__(self, num_channels, num_taps, step_size, leaky_factor=1.0):
        """
        Initializes the FIR filter.

        Parameters:
        - num_channels (int): The number of channels in the input signal.
        - num_taps (int): The number of filter taps.
        - step_size (float): The step size of the filter.
        - leaky_factor (float, optional): The factor by which the weights are multiplied at each update. Default is 1.0.
        """
        self.num_taps = num_taps
        self.step_size = step_size
        self.leaky_factor = leaky_factor
        self.weights = np.zeros((num_taps,))
        self.buffer = np.zeros((num_channels, num_taps))
        
    def filter(self, input_signal):
        """
        Filters an input signal with the FIR filter.

        Parameters:
        - input_signal (ndarray): The input signal to be filtered.

        Returns:
        - output_signal (ndarray): The filtered output signal.
        """
        num_channels, signal_length = input_signal.shape
        output_signal = np.zeros((num_channels, signal_length - self.num_taps + 1))
        for i in range(signal_length - self.num_taps + 1):
            self.buffer[:, 1:] = self.buffer[:, :-1]
            self.buffer[:, 0] = input_signal[:, i]
            filtered = np.dot(self.buffer, self.weights)
            output_signal[:, i] = filtered

        # Pad filtered output with zeros
        if output_signal.shape[1] < input_signal.shape[1]:
            pad_length = input_signal.shape[1] - output_signal.shape[1]
            output_signal = np.concatenate((output_signal, np.zeros((input_signal.shape[0], pad_length))), axis=1)

        return output_signal
    
    def update(self, X, d):
        """
        Updates the weights of the FIR filter using the filtered signal and a desired output signal.

        Parameters:
        - X (ndarray): The input signal to the filter.
        - d (ndarray): The desired output signal.

        Returns:
        None
        """
        filtered_signal = self.filter(X)
        filtered_signal_1d = np.sum(filtered_signal, axis=0)
        delta = d.shape[0] - filtered_signal_1d.shape[0]
        if delta > 0:
            filtered_signal_1d = np.concatenate((filtered_signal_1d, np.zeros(delta)))
        error_signal = d - filtered_signal_1d
        for n in range(X.shape[1] - self.num_taps + 1):
            x = X[:, n:n + self.num_taps]
            error = error_signal[n]
            y = np.sum(x * self.weights, axis=1)
            self.weights += self.step_size * x.T @ (error / np.sum(x ** 2, axis=1))
            self.weights *= self.leaky_factor