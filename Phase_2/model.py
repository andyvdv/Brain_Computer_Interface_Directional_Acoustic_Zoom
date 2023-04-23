from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Activation, Dot, Flatten, Reshape, MaxPooling1D, Dropout, BatchNormalization, LeakyReLU, Concatenate
from keras.optimizers import Adam
from keras.backend import l2_normalize
from sklearn.model_selection import train_test_split
from keras.utils import Sequence
from keras.callbacks import EarlyStopping
from keras import layers
import typing

import random
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from datagenerator import DataGenerator
from kerastuner.tuners import RandomSearch

class CNN():
    
    def __init__(self, 
                 time_window, 
                 files, 
                 subjects=None, 
                 batch_size=1, 
                 model='simple', 
                 patience=5, 
                 learning_rate=0.00445, 
                 channels=64, 
                 dropout=False) -> None:
        self.time_window = time_window
        self.channels=channels
        self.files = files
        random.shuffle(self.files)
        self.setup_generators(batch_size, subjects, channels=channels)
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.batch_size = batch_size
        self.subjects = subjects
        
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        print("Num GPUs Available: ", len (physical_devices))
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        
        if model == 'simple':
            self.model = self.build_simple_model()
        elif model == 'complex':
            self.model = self.build_complex_model()
        
        
    def setup_generators(self, batch_size, subjects=None, channels=64):
        self.train_generator = DataGenerator(self.files, 
                                             time_window=self.time_window, 
                                             batch_size=batch_size, 
                                             stage='train', 
                                             subjects=subjects,
                                             channels=channels)
        self.val_generator = DataGenerator(self.files, 
                                           time_window=self.time_window, 
                                           batch_size=batch_size, 
                                           stage='val', 
                                           subjects=subjects,
                                           channels=channels)
        self.test_generator = DataGenerator(self.files, 
                                            time_window=self.time_window, 
                                            batch_size=batch_size, 
                                            stage='test', 
                                            subjects=subjects,
                                            channels=channels)
        
    def build_simple_model(self):
        eeg = Input(shape=(self.time_window, self.channels))
        env1 = Input(shape=(self.time_window, 1))
        env2 = Input(shape=(self.time_window, 1))

        conv1 = Conv1D(filters=16, kernel_size=16, activation='relu', padding='same')(eeg)

        cos_sim1 = tf.keras.layers.Dot(axes=(1, 1), normalize=True)([conv1, env1])
        cos_sim2 = tf.keras.layers.Dot(axes=(1, 1), normalize=True)([conv1, env2])

        # Classification
        out1 = Dense(1, activation="sigmoid")(tf.keras.layers.Flatten()(tf.keras.layers.Concatenate()([cos_sim1, cos_sim2])))

        # 1 output per batch
        out = tf.keras.layers.Reshape([1], name="output_name")(out1)
        model = tf.keras.Model(inputs=[eeg, env1, env2], outputs=[out])

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
        
    def build_complex_model(self):
        # BUILDING THE MODEL
        time_window = self.train_generator.time_window
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)

        eeg = Input(shape=(time_window, self.channels))
        env1 = Input(shape=(time_window, 1))
        env2 = Input(shape=(time_window, 1))

        conv1 = Conv1D(filters=64, kernel_size=16, padding='same')(eeg)
        conv1 = BatchNormalization()(conv1)
        conv1 = LeakyReLU()(conv1)
        conv1 = Dropout(0.1)(conv1)

        conv2 = Conv1D(filters=64, kernel_size=8, padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU()(conv2)
        conv2 = Dropout(0.2)(conv2)

        conv3 = Conv1D(filters=32, kernel_size=4, padding='same')(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU()(conv3)
        conv3 = Dropout(0.3)(conv3)
            
        # Convolutional layers for env1 and env2
        env1_conv = Conv1D(filters=32, kernel_size=16, padding='same')(env1)
        env1_conv = LeakyReLU()(env1_conv)
        env2_conv = Conv1D(filters=32, kernel_size=16, padding='same')(env2)
        env2_conv = LeakyReLU()(env2_conv)

        cos_sim1 = tf.keras.layers.Dot(axes=(1, 1), normalize=True)([conv3, env1_conv])
        cos_sim2 = tf.keras.layers.Dot(axes=(1, 1), normalize=True)([conv3, env2_conv])

        # Classification
        merged = tf.keras.layers.Flatten()(tf.keras.layers.Concatenate()([cos_sim1, cos_sim2]))
        fc1 = Dense(32, activation="relu")(merged)
        fc1 = Dropout(0.4)(fc1)
        fc2 = Dense(32, activation="relu")(fc1)
        out1 = Dense(1, activation="sigmoid")(fc2)

        # 1 output per batch
        out = tf.keras.layers.Reshape([1], name="output_name")(out1)
        model = tf.keras.Model(inputs=[eeg, env1, env2], outputs=[out])

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    
    
        # # BUILDING THE MODEL
        # time_window = self.train_generator.time_window
        # early_stopping = EarlyStopping(monitor='val_loss', patience=5)

        # from keras.layers import LeakyReLU

        # eeg = Input(shape=(time_window, 64))
        # env1 = Input(shape=(time_window, 1))
        # env2 = Input(shape=(time_window, 1))

        # conv1 = Conv1D(filters=16, 
        #                kernel_size=16, 
        #                activation=hp.Choice('activation_1', ['relu', 'tanh']),
        #                padding='same')(eeg)
        # conv1 = BatchNormalization()(conv1)
        # conv1 = LeakyReLU()(conv1)
        # # conv1 = Dropout(0.3)(conv1)

        # conv2 = Conv1D(filters=64, 
        #                kernel_size=8, 
        #                activation=hp.Choice('activation_2', ['relu', 'tanh']),
        #                padding='same')(conv1)
        # conv2 = BatchNormalization()(conv2)
        # conv2 = LeakyReLU()(conv2)
        # # conv2 = Dropout(0.3)(conv2)

        # conv3 = Conv1D(filters=128, 
        #                kernel_size=4, 
        #                activation=hp.Choice('activation_3', ['relu', 'tanh']),
        #                padding='same')(conv2)
        # conv3 = BatchNormalization()(conv3)
        # conv3 = LeakyReLU()(conv3)
        # # conv3 = Dropout(0.3)(conv3)

        # cos_sim1 = tf.keras.layers.Dot(axes=(1, 1), normalize=True)([conv3, env1])
        # cos_sim2 = tf.keras.layers.Dot(axes=(1, 1), normalize=True)([conv3, env2])

        # # Classification
        # merged = tf.keras.layers.Flatten()(tf.keras.layers.Concatenate()([cos_sim1, cos_sim2]))
        # fc1 = Dense(64, activation="relu")(merged)
        # fc1 = Dropout(0.5)(fc1)
        # fc2 = Dense(32, activation="relu")(fc1)
        # out1 = Dense(1, activation="sigmoid")(fc2)

        # # 1 output per batch
        # out = tf.keras.layers.Reshape([1], name="output_name")(out1)
        # model = tf.keras.Model(inputs=[eeg, env1, env2], outputs=[out])

        # # Compile the model
        # model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), 
        #               loss='binary_crossentropy',
        #               metrics=['accuracy'])
        
        # return model
    
    def build_complex_model_for_tuning(self, hp):
        time_window = 128  # replace this with your actual time window value
        channels = 8       # replace this with your actual channel count value

        eeg = Input(shape=(time_window, channels))
        env1 = Input(shape=(time_window, 1))
        env2 = Input(shape=(time_window, 1))

        conv1 = Conv1D(filters=hp.Int('conv1_filters', 32, 64, step=32), kernel_size=16, padding='same')(eeg)
        conv1 = BatchNormalization()(conv1)
        conv1 = LeakyReLU()(conv1)
        conv1 = Dropout(hp.Float('conv1_dropout', 0.1, 0.5, step=0.1))(conv1)

        conv2 = Conv1D(filters=hp.Int('conv2_filters', 32, 64, step=32), kernel_size=8, padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU()(conv2)
        conv2 = Dropout(hp.Float('conv2_dropout', 0.1, 0.5, step=0.1))(conv2)

        conv3 = Conv1D(filters=hp.Int('conv3_filters', 32, 64, step=32), kernel_size=4, padding='same')(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU()(conv3)
        conv3 = Dropout(hp.Float('conv3_dropout', 0.1, 0.5, step=0.1))(conv3)

        env1_conv = Conv1D(filters=32, kernel_size=16, padding='same')(env1)
        env1_conv = LeakyReLU()(env1_conv)
        env2_conv = Conv1D(filters=32, kernel_size=16, padding='same')(env2)
        env2_conv = LeakyReLU()(env2_conv)

        cos_sim1 = layers.Dot(axes=(1, 1), normalize=True)([conv3, env1_conv])
        cos_sim2 = layers.Dot(axes=(1, 1), normalize=True)([conv3, env2_conv])

        merged = layers.Flatten()(layers.Concatenate()([cos_sim1, cos_sim2]))
        fc1 = Dense(hp.Int('dense1_units', 32, 96, step=32), activation="relu")(merged)
        fc1 = Dropout(hp.Float('dense1_dropout', 0.1, 0.5, step=0.1))(fc1)
        fc2 = Dense(hp.Int('dense2_units', 16, 64, step=16), activation="relu")(fc1)
        out1 = Dense(1, activation="sigmoid")(fc2)

        out = layers.Reshape([1], name="output_name")(out1)
        model = Model(inputs=[eeg, env1, env2], outputs=[out])

        model.compile(
            optimizer=Adam(
                learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
            ),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self):
        self.history = self.model.fit(self.train_generator, validation_data=self.val_generator, epochs=100, callbacks=[self.early_stopping], verbose=2)
    
    def test_model(self):
        score = self.model.evaluate(self.test_generator, verbose=2)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
        
    def tune_complex_model(self):
        tuner = RandomSearch(
            self.build_complex_model_for_tuning,
            objective='val_accuracy',
            max_trials=15,  # Set this to the desired number of trials
            executions_per_trial=3,  # Set this to the desired number of executions per trial
            directory='tuning_results',
            project_name='cnn_model'
        )

        # Replace these with your actual train and validation data generators
        train_generator = self.train_generator
        val_generator = self.val_generator

        tuner.search(
            train_generator,
            epochs=20,
            validation_data=val_generator,
            callbacks=[EarlyStopping(monitor='val_loss', patience=5)]
        )

        # Get the best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        return best_hps

    
    def plot_history(self):
        training_loss = self.history.history['loss']
        training_accuracy = self.history.history['accuracy']
        val_loss = self.history.history['val_loss']
        val_accuracty = self.history.history['val_accuracy']

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        title = f'CNN model history'
        fig.suptitle(title)
        # Make x-axis label
        axs[0].set_xlabel('Epoch')
        axs[1].set_xlabel('Epoch')
        # Make y-axis labels
        axs[0].set_ylabel('Loss')
        axs[1].set_ylabel('Accuracy')
        
        axs[0].plot(training_loss, label='Training Loss')
        axs[0].plot(val_loss, label='Validation Loss')
        axs[0].legend()

        axs[1].plot(training_accuracy, label='Training Accuracy')
        axs[1].plot(val_accuracty, label='Validation Accuracy')
        axs[1].legend()

        plt.show()