import os, sys
sys.path.append(os.getcwd())
from helpers import create_micsigs, verify_correct_fs
from package.gui_utils import load_rirs
from music import *
import scipy.signal as ss
import scipy.linalg as LA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def narrowband_exercise(scenarioPath, speechfilenames):
    # Load the scenario
    nmics = 5
    nsources = len(speechfilenames)
    acousticScenario = load_rirs(os.getcwd() + scenarioPath)
    verify_correct_fs(acousticScenario, 44100)
    # Obtain the microphone signals
    micsigs, _, _, _ = create_micsigs(acousticScenario, nmics, speechfilenames, duration=10)
    # Stack the STFTs of the microphone signals
    S, freqs_list = stack_stfts(micsigs, acousticScenario.fs, 1024, 512)
    # Find the pseudospectrum for the frequency bin with the maximum power
    spectrum, angles = music_narrowband(S, nmics, nsources, freqs_list, 0.05)
    doa = angles[np.argmax(spectrum)]
    return spectrum, angles, doa

def part_1_1(normalise=True):
    speechfilenames = ["speech1.wav"]
    source_doas = [0, 45, 90, 135, 158, 180]
    fig, axs = plt.subplots((len(source_doas) // 2), 2, sharex=True, sharey=normalise)
    for i, deg in enumerate(source_doas):
        s, a, doa = narrowband_exercise(f"/rirs/Week_2/Part_1_1/{deg}deg.pkl.gz", speechfilenames)
        if normalise:
            # Normalise the spectrum to a range of [0,100]
            s = (s / s[np.argmax(s)]) * 100
        axis = axs[i // 2][i % 2]
        axis.plot(a, s, 'b', label=f"P(θ)")
        axis.stem(deg, max(s), markerfmt='', linefmt='r--', label=f"{deg}°")
        axis.legend()
    fig.suptitle('Narrowband MUSIC Pseudospectrums', fontweight='bold')
    fig.text(0.02, 0.5, 'Power (Normalised)' if normalise else 'Power', va='center', rotation='vertical', fontweight='bold', fontsize='14')
    fig.text(0.5, 0.04, 'Angle (θ)', ha='center', va='center', fontweight='bold', fontsize='14')
    plt.show()

def part_1_5(normalise=True):
    speechfilenames = ["speech1.wav"]
    source_doas = [0, 45, 90, 135, 180]
    fig, axs = plt.subplots((len(source_doas) // 2)+1, 2, sharex=True, sharey=normalise)
    for i, deg in enumerate(source_doas):
        s, a, doa = narrowband_exercise(f"/rirs/Week_2/Part_1_5/{deg}deg.pkl.gz", speechfilenames)
        if normalise:
            # Normalise the spectrum to a range of [0,100]
            s = (s / s[np.argmax(s)]) * 100
        axis = axs[i // 2][i % 2]
        axis.plot(a, s, 'b', label=f"P(θ)")
        axis.stem(deg, max(s), markerfmt='', linefmt='r--', label=f"{deg}°")
        axis.legend()
    fig.suptitle('Narrowband MUSIC Pseudospectrums', fontweight='bold')
    fig.text(0.02, 0.5, 'Power (Normalised)' if normalise else 'Power', va='center', rotation='vertical', fontweight='bold', fontsize='14')
    fig.text(0.5, 0.04, 'Angle (θ)', ha='center', va='center', fontweight='bold', fontsize='14')
    plt.show()

def part_2_1(normalise=True):
    speechfilenames = ["speech1.wav", "speech2.wav"]
    s, a, doa = narrowband_exercise("/rirs/Week_2/Part_2_1/45_135.pkl.gz", speechfilenames)
    # Normalise the spectrum to a range of [0,100] if normalise is True
    s = (s / s[np.argmax(s)]) * 100 if normalise else s
    plt.plot(a, s, 'b', label=f"P(θ)")
    # plt.stem(deg, max(s), markerfmt='', linefmt='r--', label=f"{deg}°")
    plt.suptitle("Narrowband MUSIC Pseudospectrum, Multiple Sources", fontweight='bold')
    plt.ylabel('Power (Normalised)' if normalise else 'Power', fontweight='bold')
    plt.xlabel("Angle (θ)", fontweight='bold')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # part_1_1()
    # part_1_5(normalise=True)
    part_2_1(normalise=True)