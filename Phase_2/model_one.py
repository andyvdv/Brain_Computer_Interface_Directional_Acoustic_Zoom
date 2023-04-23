# Setup
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Activation, Lambda, GlobalAveragePooling1D
from keras import regularizers
import numpy as np
import tensorflow as tf
import os
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Activation, Dot, Flatten, Reshape, MaxPooling1D, Dropout
from keras.optimizers import Adam
from keras.backend import l2_normalize
from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping
from typing import Sequence, Union
import pathlib
import mne
import random

from IPython.display import clear_output
import matplotlib.pyplot as plt

class DataGenerator(Sequence):
    """Generate data for the Match/Mismatch task."""

    def __init__(
        self,
        files: Sequence[Union[str, pathlib.Path]],
        base_path: str = os.getcwd() + "/data/train_cnn",
        eeg_base_path: str = os.getcwd() + "/data/eeg",
        time_window: int = 60 * 10,
        batch_size: int = 32,
    ):
        """Initialize the DataGenerator."""

        self.files = files
        self.base_path = base_path
        self.eeg_base_path = eeg_base_path
        self.time_window = time_window
        self.batch_size = batch_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # Generate a batch of data
        inputs_batch = []
        labels_batch = []

        for i in range(self.batch_size):
            recording_index = (index * self.batch_size) + i
            if recording_index >= len(self.files):
                break

            inputs, labels = self._load_data(recording_index)

            if inputs is not None and labels is not None:
                inputs_batch.append(inputs)
                labels_batch.append(labels)

        # Convert lists to arrays and return
        inputs_batch = [np.array(x) for x in zip(*inputs_batch)]
        labels_batch = np.array(labels_batch)

        return inputs_batch, labels_batch

    def _load_data(self, recording_index):
        eeg_filename = self.files[recording_index]
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
    
        min_data_len = min(eeg_preprocessed.shape[0], attended_preprocessed.shape[0], unattended_preprocessed.shape[0])
        new_data_length = (min_data_len // self.time_window) * self.time_window
        
        eeg_preprocessed = eeg_preprocessed[:new_data_length]
        attended_preprocessed = attended_preprocessed[:new_data_length]
        unattended_preprocessed = unattended_preprocessed[:new_data_length]

        # Reshape the attended and unattended envelopes to have an extra dimension
        attended_preprocessed = attended_preprocessed.reshape(-1, self.time_window, 1)
        unattended_preprocessed = unattended_preprocessed.reshape(-1, self.time_window, 1)

        # Reshape the EEG data
        eeg_preprocessed = eeg_preprocessed.reshape(-1, self.time_window, 64)
        
        labels = np.ones((eeg_preprocessed.shape[0], 1))
        
        inputs, labels = self.batch_equalizer(eeg_preprocessed, attended_preprocessed, unattended_preprocessed, labels)
        
        return (inputs, labels)

    def on_epoch_end(self):
        random.shuffle(self.files)
        
    def batch_equalizer(self, eeg, env_1, env_2, labels):
        # present each of the eeg segments twice, where the envelopes, and thus the labels 
        # are swapped around. EEG presented in small segments [bs, window_length, 64]
        return [np.concatenate([eeg, eeg], axis=0), np.concatenate([env_1, env_2], axis=0), np.concatenate([ env_2, env_1], axis=0)], np.concatenate([labels, (labels+1)%2], axis=0)
    
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len (physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

data_folder = os.getcwd() + '/data/eeg'
all_files = [f for f in os.listdir(data_folder) if f.endswith('.npz')]

data_generator = DataGenerator(all_files, batch_size=32)

# BUILDING THE MODEL
time_window = data_generator.time_window
# early_stopping = EarlyStopping(monitor='val_loss', patience=5)

eeg = Input(shape=(time_window, 64))
env1 = Input(shape=(time_window, 1))
env2 = Input(shape=(time_window, 1))

conv1 = Conv1D(filters=16, kernel_size=16, activation='relu', padding='causal')(eeg)

cos_sim1 = tf.keras.layers.Dot(axes=(1, 1), normalize=True)([conv1, env1])
cos_sim2 = tf.keras.layers.Dot(axes=(1, 1), normalize=True)([conv1, env2])

# Classification
out1 = Dense(1, activation="sigmoid")(tf.keras.layers.Flatten()(tf.keras.layers.Concatenate()([cos_sim1, cos_sim2])))

# 1 output per batch
out = tf.keras.layers.Reshape([1], name="output_name")(out1)
model = tf.keras.Model(inputs=[eeg, env1, env2], outputs=[out])

# Compile the model
# Use legacy adam optimizer
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(data_generator, epochs=1, steps_per_epoch=100, verbose=2)

training_loss = history.history['loss']
validation_loss = history.history['val_loss']
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

# Print the training and validation loss and accuracy
print("Training Loss:", training_loss)
print("Validation Loss:", validation_loss)
print("Training Accuracy:", training_accuracy)
print("Validation Accuracy:", validation_accuracy)

# Plot the training and validation loss
plt.figure()
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the training and validation accuracy
plt.figure()
plt.plot(training_accuracy, label='Training Accuracy')
plt.plot(validation_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
