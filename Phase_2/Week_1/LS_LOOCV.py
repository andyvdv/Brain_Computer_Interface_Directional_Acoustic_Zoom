from brian2 import Hz, kHz
from brian2hears import Sound, erbspace, Gammatone, Filterbank
from scipy.signal import butter, filtfilt, sosfiltfilt, lfilter
import librosa as lb
import os, sys
sys.path.append(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from scipy.optimize import minimize
from scipy.stats import pearsonr
from scipy.optimize import OptimizeResult
from timeit import default_timer as timer
from sklearn.preprocessing import MinMaxScaler


class EnvelopeFromGammatoneFilterbank(Filterbank):
    """Converts the output of a GammatoneFilterbank to an envelope."""

    def __init__(self, source):
        """Initialize the envelope transformation.

        Parameters
        ----------
        source : Gammatone
            Gammatone filterbank output to convert to envelope
        """
        super().__init__(source)

        self.nchannels = 1

    def buffer_apply(self, input_):

        # 6. take absolute value of the input_
        compressed_subbands = np.abs(input_)**0.6

        combined_envelope = np.sum(compressed_subbands, axis=1)

        return  combined_envelope.reshape(combined_envelope.shape[0], 1)
    

def load_eeg_data(filename):
    data = np.load(filename)
    return data

def bandpass_filter_eeg(eeg_data, fs, low_freq, high_freq):
    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(1, [low, high], btype='band')
    filtered_eeg = lfilter(b, a, eeg_data, axis=0)
    return filtered_eeg

def downsample_eeg(filtered_eeg, fs, target_fs):
    downsample_ratio = fs // target_fs
    downsampled_eeg = lb.resample(filtered_eeg.T, orig_sr=fs, target_sr=target_fs).T
    return downsampled_eeg


def preprocessing_audio(filepath,downsampled_eeg_array):

    # ----- Audio data -----

    # Step 1 : Read audio filename.wav to the variable A.
    audio_signal, sr = lb.load(filepath, sr=44100)

    #print(f"Audio signal length: {len(audio_signal)}")

    # Convert the numpy array to a Sound object
    audio_sound = Sound(audio_signal, samplerate=sr*Hz)

    # Step 2 : Decompose the audio in 28 frequency subbands
    num_filters = 28
    center_freqs = np.linspace(50*Hz, 5*kHz, num_filters)
    gammatone_filterbank = Gammatone(audio_sound, center_freqs)

    # Step 3 & 4 : Perform a power-law compression on each subband signals and linearly
    # combine the bandpass filtered envelopes to a single envelope (See EnvelopeFromGammatoneFilterbank() class)
    envelope_calculation = EnvelopeFromGammatoneFilterbank(gammatone_filterbank)
    combined_envelope = envelope_calculation.process()

    # Step 5 : Filter the audio using a BPF (butterworth)

    # (5.1) Choose the frequency range based on the method
    low_freq = 1
    high_freq = 9

    # (5.2) Design a Butterworth bandpass filter
    order = 1
    nyquist = 0.5 * sr
    low = low_freq / nyquist
    high = high_freq / nyquist
    sos = butter(order, [low, high], btype='band', output='sos')

    normalized_envelope = combined_envelope / np.max(combined_envelope)
    filtered_envelope = sosfiltfilt(sos, normalized_envelope[:, 0])

    # Step 6 : Downsample the resulting signals
    # Choose the downsampling rate based on the method
    downsampling_rate = 20

    resampled_envelope = lb.resample(filtered_envelope, orig_sr=sr, target_sr=downsampling_rate)

    return resampled_envelope[:downsampled_eeg_array.shape[0]]


def preprocessing_eeg(eeg_path):
    
    # ----- EEG data -----

    # Step 1 : Load the EEG data
    
    eeg_data = load_eeg_data(eeg_path)
    eeg = eeg_data['eeg']
    fs = eeg_data['fs']

    #print("Speech attended: ", eeg_data['stimulus_attended'])
    #print("Speech unattended: ",eeg_data['stimulus_unattended'])

    # Step 2 : Filter the EEG signals using a bandpass filter
    low_freq, high_freq = 1, 9  # For linear regression
    filtered_eeg = bandpass_filter_eeg(eeg, fs, low_freq, high_freq)

    # Step 3 : Downsample the EEG signals
    target_fs = 20  # For linear regression
    downsampled_eeg_array = downsample_eeg(filtered_eeg, fs, target_fs)

    return downsampled_eeg_array, eeg_data['stimulus_attended'], eeg_data['stimulus_unattended']


def get_speech_resampled(sa_path,eeg_array):
    """
    The function first constructs the file path for the processed SA/SU and checks if the file already exists.
    If the file exists, it loads the SA/SU from the .npy file and returns it. 
    If the file does not exist, the function calls the preprocessing function to process the resampled SA and the corresponding EEG data. 
    The processed SA/SU is then saved as a .npy file in the audioprocessed folder and returned.

    Parameters
    ----------
    resampled_path : str
        The path to the resampled attended/unattended speech signal file.

    Returns
    -------
    s : ndarray
        The resampled attended/unattended speech signal.
    """

    last_path = sa_path.split('/')[-1].split('.')[0] + '.npy'
    
    folder_audioprocessed =  os.getcwd() + "/Phase_2/data/audioprocessed" # audio
    npy_path = os.path.join(folder_audioprocessed, last_path)

    if os.path.exists(npy_path) == False:
        
        s =  preprocessing_audio(sa_path, eeg_array)
        np.save(npy_path, s)
        return s
    
    
    return np.load(npy_path)



def get_sa_su_eeg(eeg_name):
    folder_eeg = os.getcwd() + "/Phase_2/data/eeg" # audio
    eeg_path = os.path.join(folder_eeg,eeg_name)
    
    eeg, sa_name, su_name = preprocessing_eeg(eeg_path) # get the processed eeg and the attended/unattended speech name of that EEG

    # ---- GET ATTENDED SPEECH ARRAY
    folder_audio = os.getcwd() + "/Phase_2/data/stimuli" # audio
    resampled_sa_path = os.path.join(folder_audio,str(sa_name))
    sa = get_speech_resampled(resampled_sa_path,eeg)

    # ---- GET UNATTENDED SPEECH ARRAY
    resampled_su_path = os.path.join(folder_audio,str(su_name))
    su = get_speech_resampled(resampled_su_path,eeg)

    return sa, su, eeg

def construct_m(sa, eeg):
    Nl = int(250 * pow(10, -3) * 20)
    t = 0
    m = np.array(eeg[t:Nl]).ravel('F')

    m_t = np.zeros((len(sa) - Nl + 1, m.shape[0]))

    for i in range(len(sa) - Nl + 1):
        if eeg[i:i + Nl].shape[0] != Nl:
            m_t = m_t[:-1]
            break
        m_t[i] = np.array(eeg[i:i + Nl]).ravel('F')

    m_t = np.array(m_t).T

    # Apply MinMaxScaler to the EEG data
    scaler = MinMaxScaler()
    m_t = scaler.fit_transform(m_t)

    sa = sa[:m_t.shape[1]]  # cut the 5 last samples (because of time lags)

    return m_t, sa

 

def least_squares(all_files, k, K, min_length=float('inf')):
    """
    Perform least squares optimization for a specific EEG trial.

    Parameters
    ----------
    all_files : list
        List of all EEG trial file names.
    k : int
        Index of the EEG trial to exclude from the optimization.
    K : int
        Total number of EEG trials.
    min_length : float, optional
        Minimum length of the speech signals, default is infinity.

    Returns
    -------
    d_k : ndarray
        The optimized parameter vector for the given EEG trial.
    """
    M = []
    SA = []

    # Prepare M & SA arrays from all instances but k
    for i in range(K):
        if i != k:

            sa_i, su_i, eeg_i = get_sa_su_eeg(all_files[i])
            m_i, sa_i = construct_m(sa_i, eeg_i)

           
            if len(sa_i) < min_length:
                min_length = len(sa_i)
            
            M.append(m_i)
            SA.append(sa_i)

    new_SA = []; new_M = []
    for i in range(len(M)):
        new_SA.append(SA[i][:min_length])

        new_M_i = []
        for j in range(len(M[i])):
            new_M_i.append(M[i][j][:min_length])
            
        
        new_M.append(new_M_i)

    
    new_SA = np.array(new_SA); new_M = np.array(new_M)

    M_stacked = np.concatenate(new_M, axis=1)
    SA_stacked = np.concatenate(new_SA)

    #print("M_stacked shape:", M_stacked.shape)
    #print("SA_stacked shape:", SA_stacked.shape)
    #print("M_stacked shape:", M_stacked.shape)
    #print("SA_stacked shape:", SA_stacked.shape)

    # Perform least squares using np.linalg.lstsq
    d_k, _, _, _ = np.linalg.lstsq(M_stacked.T, SA_stacked.T, rcond=None)

    return d_k
    

def get_sa_su(eeg_name):
    folder_eeg = os.getcwd() + "/Phase_2/data/eeg" # audio
    eeg_path = os.path.join(folder_eeg,eeg_name)
    
    eeg, sa_name, su_name = preprocessing_eeg(eeg_path) # get the processed eeg and the attended speech name of that EEG

    # ---- GET ATTENDED SPEECH ARRAY
    folder_audio = os.getcwd() + "/Phase_2/data/stimuli" # audio
    resampled_sa_path = os.path.join(folder_audio,str(sa_name))
    sa = get_speech_resampled(resampled_sa_path,eeg_path)

    # ---- GET UNATTENDED SPEECH ARRAY
    resampled_su_path = os.path.join(folder_audio,str(su_name))
    su = get_speech_resampled(resampled_su_path,eeg_path)

    return sa, su, eeg

def build(eeg_name, window=float('inf')):
    """
    Build an EEG signal processing pipeline for a specific EEG trial.

    Parameters
    ----------
    eeg_name : str
        Name of the EEG trial file.
    window : float, optional
        Length of the window for processing the EEG signal, default is infinity.

    Returns
    -------
    processed_eeg : ndarray
        The processed EEG data for the given trial.
    """
    sa, su, eeg = get_sa_su(eeg_name)

    m_t, sa = construct_m(sa,eeg)

    if window != float('inf'):
        m_t = m_t[:, :window]
        return m_t, sa[:window], su[:window]
    
    return m_t, sa, su


# ---------- TEST ACCURACY ----------

def test_accuracy(sa,su,d_k,m):
    """
    Test the accuracy of a machine learning model on a given dataset.

    Parameters
    ----------
    sa : ndarray
        The source array containing input data.
    su : ndarray
        The target array containing the true labels.
    d_k : int
        The dimension of the keys in the model.
    m : Model
        The machine learning model to be tested.

    Returns
    -------
    accuracy : float
        The accuracy of the model on the given dataset, expressed as a percentage.
    """

    sa = sa.ravel()
    su = su.ravel()

    sa_est = d_k.T @ m
    sa_est = sa_est.ravel()

    min_length = min(len(sa_est), len(sa), len(su))
    sa_est = sa_est[:min_length]
    sa = sa[:min_length]
    su = su[:min_length]

    corr_sa, p_value = pearsonr(sa_est, sa)
    corr_su, p_value = pearsonr(sa_est, su)

    print("Pearson correlation coefficient with sa:", corr_sa)
    print("Pearson correlation coefficient with su:", corr_su)

    if corr_sa > corr_su:
        print('The right speech has been selected :-)')
        return 1

    else:
        print('The wrong speech has been selected :-(')
        return 0


# ---------- MAIN ----------


if __name__ == "__main__":

    folder_eeg = os.getcwd() + "/Phase_2/data/eeg" # EEG
    all_files = os.listdir(folder_eeg); all_files.sort(); all_files = all_files[1:]

    # ----- TO CHOOSE -----
    K = 100 # Number of EEG trials to use (Choose 50, 100, 200, 300 or 484)
    max_test = 20 # Number of EEG trials that we actually one to test and look for their average accuracy (Put K if you want to use all)
    w_length = 300 # Window length: Number of samples to use (Choose 1200, 600, 300, 100) = (60s, 30s, 15s, 5s)
    folder_dk = os.getcwd() + "/Phase_2/data/d_k" # d_k file to use (Ex: replace d_k with d_k_K100_30s for precomputed d_k with K=100, w_length = 30s * 20Hz = 600)
    # ----- END -----

    d_k_list = []
    

    for k in range(max_test):
        print(f"Processing k = {k}")
        d_k_path = os.path.join(folder_dk,"d_k_{}.npy".format(k))
        if os.path.exists(d_k_path) == False:
            start = timer()
            d_k = least_squares(all_files, k, K, w_length)
            end = timer()
            print("Elapsed time for d_{0} : {1} seconds".format(k, round(end-start,4)))
            np.save(d_k_path,d_k)
            d_k_list.append(d_k)
        else:
            d_k = np.load(d_k_path)
            d_k_list.append(d_k)


    D = np.array(d_k_list)

    #________________________________
    # >>>>>> Training accuracy <<<<<<
    sum_correct = 0
    for index_train in range(max_test):
        print("\n")
        print(" -> TEST FOR k =",index_train)
        print("\n")
        m_train, sa_train, su_train = build(all_files[index_train],window=w_length)
        sum_correct += test_accuracy(sa_train,su_train,D[index_train],m_train)
    
    print("Test accuracy with LOOCV = {} %".format(100 * sum_correct/max_test))