from keras.utils import Sequence
import numpy as np
import os, typing
import random
import pathlib
import matplotlib.pyplot as plt

class DataGenerator(Sequence):
    """Generate data for the Match/Mismatch task."""

    def __init__(
        self,
        files: typing.Sequence[typing.Union[str, pathlib.Path]],
        base_path: str = os.getcwd() + "/data/train_cnn",
        eeg_base_path: str = os.getcwd() + "/data/eeg",
        time_window: int = 10,
        batch_size: int = 32,
        stage = 'train',
        subjects = None,
        channels = 64
    ):
        """Initialize the DataGenerator."""

        self.files = files
        self.base_path = base_path
        self.eeg_base_path = eeg_base_path
        self.time_window = time_window * 64
        self.batch_size = batch_size
        self.stage = stage
        self.subjects = subjects
        self.channels = channels
        
        if self.subjects is not None:
            self.files = [f for f in self.files if f.split("_")[0] in self.subjects]
            print(f'{self.stage} files: {self.files}')
            
        random.seed(42)
        random.shuffle(self.files)

    def __len__(self):
        return int(np.ceil(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        # Generate a batch of data
        eeg_batch = None
        env1_batch = None
        env2_batch = None
        labels_batch = None

        for i in range(self.batch_size):
            recording_index = (index * self.batch_size) + i
            if recording_index >= len(self.files):
                break

            eeg, env1, env2, labels = self._load_data(recording_index)
            if eeg_batch is None:
                eeg_batch = eeg
            if env1_batch is None:
                env1_batch = env1
            if env2_batch is None:
                env2_batch = env2
            if labels_batch is None:
                labels_batch = labels

            if eeg is not None and labels is not None:
                eeg_batch = np.append(eeg_batch, np.asarray(eeg), axis=0)
                env1_batch = np.append(env1_batch, np.asarray(env1), axis=0)
                env2_batch = np.append(env2_batch, np.asarray(env2), axis=0)
                labels_batch = np.append(labels_batch, np.asarray(labels), axis=0)
        eeg_batch = eeg_batch.reshape(-1, self.time_window, self.channels)
        env1_batch = env1_batch.reshape(-1, self.time_window, 1)
        env2_batch = env2_batch.reshape(-1, self.time_window, 1)
        
        return (eeg_batch, env1_batch, env2_batch), labels_batch

    def _load_data(self, index):
        eeg_filename = self.files[index]
        eeg_data = np.load(os.path.join(self.eeg_base_path, eeg_filename))

        attended_filename = str(eeg_data["stimulus_attended"]).replace(".wav", ".npy")
        unattended_filename = str(eeg_data["stimulus_unattended"]).replace(".wav", ".npy")
        
        eeg_filepath = os.path.join(self.base_path, eeg_filename.replace(".npz", ".npy"))
        attended_filepath = os.path.join(self.base_path, attended_filename)
        unattended_filepath = os.path.join(self.base_path, unattended_filename)

        if not os.path.exists(eeg_filepath) or not os.path.exists(attended_filepath) or not os.path.exists(unattended_filepath):
            return None, None, None

        eeg_preprocessed = np.load(eeg_filepath)
        attended_preprocessed = np.load(attended_filepath)
        unattended_preprocessed = np.load(unattended_filepath)
        
        # Take only self.channels channels from eeg_preprocessed.shape[1]
        eeg_preprocessed = eeg_preprocessed[:, :self.channels]
        
        min_data_len = min(eeg_preprocessed.shape[0], attended_preprocessed.shape[0], unattended_preprocessed.shape[0])
        new_data_length = (min_data_len // self.time_window) * self.time_window
        
        eeg_preprocessed = eeg_preprocessed[:new_data_length]
        attended_preprocessed = attended_preprocessed[:new_data_length]
        unattended_preprocessed = unattended_preprocessed[:new_data_length]
        
        if self.stage == 'train':
            eeg_preprocessed = eeg_preprocessed[:int(eeg_preprocessed.shape[0] * 0.7)]
            attended_preprocessed = attended_preprocessed[:int(attended_preprocessed.shape[0] * 0.7)]
            unattended_preprocessed = unattended_preprocessed[:int(unattended_preprocessed.shape[0] * 0.7)]
        elif self.stage == 'val':
            eeg_preprocessed = eeg_preprocessed[int(eeg_preprocessed.shape[0] * 0.7):int(eeg_preprocessed.shape[0] * 0.9)]
            attended_preprocessed = attended_preprocessed[int(attended_preprocessed.shape[0] * 0.7):int(attended_preprocessed.shape[0] * 0.9)]
            unattended_preprocessed = unattended_preprocessed[int(unattended_preprocessed.shape[0] * 0.7):int(unattended_preprocessed.shape[0] * 0.9)]
        elif self.stage == 'test':
            eeg_preprocessed = eeg_preprocessed[int(eeg_preprocessed.shape[0] * 0.9):]
            attended_preprocessed = attended_preprocessed[int(attended_preprocessed.shape[0] * 0.9):]
            unattended_preprocessed = unattended_preprocessed[int(unattended_preprocessed.shape[0] * 0.9):]
        
        # # Shuffle data with seed 42
        # np.random.seed(42)
        # np.random.shuffle(eeg_preprocessed)
        # np.random.seed(42)
        # np.random.shuffle(attended_preprocessed)
        # np.random.seed(42)
        # np.random.shuffle(unattended_preprocessed)
        
        min_data_len = min(eeg_preprocessed.shape[0], attended_preprocessed.shape[0], unattended_preprocessed.shape[0])
        new_data_length = (min_data_len // self.time_window) * self.time_window
        
        eeg_preprocessed = eeg_preprocessed[:new_data_length]
        attended_preprocessed = attended_preprocessed[:new_data_length]
        unattended_preprocessed = unattended_preprocessed[:new_data_length]
    
        labels = np.ones(eeg_preprocessed.reshape(-1, self.time_window, self.channels).shape[0])
        eeg, env1, env2, labels = self.batch_equalizer(eeg_preprocessed, attended_preprocessed, unattended_preprocessed, labels)
        return eeg, env1, env2, labels

    def on_epoch_end(self):
        pass
        # random.shuffle(self.files)
        
    def __iter__(self):
        for idx in range(self.__len__()):
            yield self.__getitem__(idx)

            if idx == self.__len__() - 1:
                self.on_epoch_end()
        
    def batch_equalizer(self, eeg, env_1, env_2, labels):
        # present each of the eeg segments twice, where the envelopes, and thus the labels 
        # are swapped around. EEG presented in small segments [bs, window_length, 64]
        return np.concatenate([eeg, eeg], axis=0), np.concatenate([env_1, env_2], axis=0), np.concatenate([ env_2, env_1], axis=0), np.concatenate([labels, (labels+1)%2], axis=0)