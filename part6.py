from helpers import *
from package.gui_utils import load_rirs
import os, itertools, math
import numpy as np
import matplotlib.pyplot as plt

speechfiles = ["part1_track1_dry.wav"]
acousticScenario = load_rirs(path= f"{os.getcwd()}/rirs/part5_45deg135deg0.00t60.pkl.gz")
nmics = 4
c = 340 # m/s

rirs = ["HMIR_L1.wav","HMIR_L2.wav","HMIR_R1.wav","HMIR_R2.wav"]

def RIRs_part6(rirs, duration):
    RIRs = [] # Will store all the RIR for each angles s.t [[yL1-30, yL2-30, ...], [yL1-60, ...]]
    folders = [x[0] for x in os.walk(f'sound_files/head_mounted_rirs')][1:] # list of all the subdirectories ('sound_files/head_mounted_rirs/s60', 'sound_files/head_mounted_rirs/s-60', ...)

    for i in range(len(folders)):
        newanglefolder = []
        for j in range(len(rirs)):
            data, samplerate = sf.read(f'{folders[i]}/{rirs[j]}')
            data = data[0:samplerate*duration]
            resampled = ss.resample(data, fs*duration)
            newanglefolder.append(resampled)
        RIRs.append(newanglefolder)

        # listen_to_array(resampled, fs)
    print(np.shape(RIRs))   #  (6, 4, 220500)

    return RIRs



### Test ###

fs = 44100
nmics = 4





print(create_micsigs_part6(fs, nmics, speechfilenames, n_audio_sources=2, n_noise_source=1, duration=5))