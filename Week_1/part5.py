from helpers import DOA_corr, TDOA_corr
from package.gui_utils import load_rirs
import os, itertools, math
import numpy as np
import matplotlib.pyplot as plt

whitenoises = ["whitenoise_signal_1.wav", "whitenoise_signal_2.wav"]
speechfiles = ["part1_track1_dry.wav", "part1_track2_dry.wav"]
acousticScenario = load_rirs(path= f"{os.getcwd()}/rirs/part5_45deg135deg0.00t60.pkl.gz")
nmics = 5
c = 340 # m/s
DOA = np.array([])
# For each unique combination of mics
for j,k in itertools.combinations(range(0, nmics), 2):
    diffs = TDOA_corr(acousticScenario, nmics, j, k, whitenoises)
    d = 0.1 * (k - j)  # Distance between microphones j and k
    doas = []
    skip = False
    for diff in diffs:
        x = diff * 1/acousticScenario.fs * c    # (samples * seconds/sample * m/second)
        _doa = np.arccos(x/d) * 180 / math.pi
        doas.append(_doa)
        if np.isnan(_doa):
            skip = True
    if not skip:
        DOA = np.append(DOA, doas)

DOA = np.sort(DOA)
source_1_doas, source_2_doas = np.hsplit(DOA, 2)
source_1_doa_est = sum(source_1_doas) / len(source_1_doas)
source_2_doa_est = sum(source_2_doas) / len(source_2_doas)

# Display results
print(f"Source 1 has an approximate DOA of {round(source_1_doa_est, 2)} degrees")
print(f"Source 2 has an approximate DOA of {round(source_2_doa_est, 2)} degrees")