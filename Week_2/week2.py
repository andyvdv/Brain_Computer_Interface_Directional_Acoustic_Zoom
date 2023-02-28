import os, sys
sys.path.append(os.getcwd())
from helpers import create_micsigs, verify_parameters
from package.gui_utils import load_rirs
from music import *
import scipy.signal as ss
import scipy.linalg as LA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def wideband_exercise(scenarioPath, speechfilenames):
     # Load the scenario
    acousticScenario = load_rirs(os.getcwd() + scenarioPath)
    nsources = acousticScenario.RIRsAudio.shape[2]
    nmics = acousticScenario.nMicsPerArray
    d = acousticScenario.distBwMics
    verify_parameters(acousticScenario, 44100, speechfilenames)
    # Obtain the microphone signals
    micsigs, _, _, _ = create_micsigs(acousticScenario, nmics, speechfilenames, duration=10)
    # Stack the STFTs of the microphone signals
    S, freqs_list = stack_stfts(micsigs, acousticScenario.fs, 1024, 512)
    # Find the pseudospectrum for the frequency bin with the maximum power
    thetas = np.arange(0, 180, 0.5)
    spectrum, doas = music_wideband(S, nmics, nsources, freqs_list, d, thetas)
    return spectrum, thetas, doas

def narrowband_exercise(scenarioPath, speechfilenames):
    # Load the scenario
    acousticScenario = load_rirs(os.getcwd() + scenarioPath)
    nsources = acousticScenario.RIRsAudio.shape[2]
    nmics = acousticScenario.nMicsPerArray
    d = acousticScenario.distBwMics
    verify_parameters(acousticScenario, 44100, speechfilenames)
    # Obtain the microphone signals
    micsigs, _, _, _ = create_micsigs(acousticScenario, nmics, speechfilenames, duration=10)
    # Stack the STFTs of the microphone signals
    S, freqs_list = stack_stfts(micsigs, acousticScenario.fs, 1024, 512)
    # Find the pseudospectrum for the frequency bin with the maximum power
    max_bin_index = find_max_bin(S)
    thetas = np.arange(0, 180, 0.5)
    spectrum, doas = music_narrowband(S, nmics, nsources, freqs_list, d, max_bin_index, thetas)
    return spectrum, thetas, doas

def part_1_1(normalise=True):
    speechfilenames = ["speech1.wav"]
    source_doas = [0, 45, 90, 135, 158, 180]
    fig, axs = plt.subplots((len(source_doas) // 2), 2, sharex=True, sharey=normalise)
    for i, deg in enumerate(source_doas):
        s, a, doas = narrowband_exercise(f"/rirs/Week_2/Part_1_1/{deg}deg.pkl.gz", speechfilenames)
        if normalise:
            # Normalise the spectrum to a range of [0,100]
            s = (s / s[np.argmax(s)]) * 100
        axis = axs[i // 2][i % 2]
        axis.plot(a, s, 'b', label=f"P(θ)")
        axis.stem(deg, max(s), markerfmt='', linefmt='r--', label=f"{deg}°")
        axis.legend()
    fig.suptitle('Narrowband MUSIC Pseudospectrums', fontweight='bold')
    fig.text(0.02, 0.5, 'Power (Normalised)' if normalise else 'Power', va='center', rotation='vertical', fontweight='bold', fontsize='12')
    fig.text(0.5, 0.04, 'Angle (θ)', ha='center', va='center', fontweight='bold', fontsize='12')

def part_1_5(normalise=True):
    speechfilenames = ["speech1.wav"]
    source_doas = [0, 45, 90, 135, 180]
    fig, axs = plt.subplots((len(source_doas) // 2)+1, 2, sharex=True, sharey=normalise)
    for i, deg in enumerate(source_doas):
        s, a, doas = narrowband_exercise(f"/rirs/Week_2/Part_1_5/{deg}deg.pkl.gz", speechfilenames)
        if normalise:
            # Normalise the spectrum to a range of [0,100]
            s = (s / s[np.argmax(s)]) * 100
        axis = axs[i // 2][i % 2]
        axis.plot(a, s, 'b', label=f"P(θ)")
        axis.stem(deg, max(s), markerfmt='', linefmt='r--', label=f"{deg}°")
        axis.legend()
    fig.suptitle('Narrowband MUSIC Pseudospectrums', fontweight='bold')
    fig.text(0.02, 0.5, 'Power (Normalised)' if normalise else 'Power', va='center', rotation='vertical', fontweight='bold', fontsize='12')
    fig.text(0.5, 0.04, 'Angle (θ)', ha='center', va='center', fontweight='bold', fontsize='12')

def part_2_1(normalise=True):
    """
    MUSIC Narrowband with 2 sources, one at 45deg and the other at 135deg
    """
    speechfilenames = ["speech1.wav", "speech2.wav"]
    s, a, doas= narrowband_exercise("/rirs/Week_2/Part_2_1/45_135.pkl.gz", speechfilenames)
    plot_pseudspectrum(a, s, "Narrowband MUSIC, 2 sources with DOA diff 90°", 
                        window_title="Part 2", stems=[45, 135])

def part_3_1():
    """
    Demonstrate the use of the narrowband MUSIC implementation on a scenario involving 2 sources
    that are near each other.
    Here DOA1 = 128.66 and DOA2 = 135

    OBSERVATIONS:
        When there are 2 sources with DOAs that are not far off, the narrowband implementation shows no
        distinction between the 2 sources and estimates a single DOA. Compare this to part 2 where, if 
        the sources have DOAs that are significantly different from each other (> 45 deg) the narrowband
        implementation easily seperates them and is able to determine their individual DOAs.
    """
    speechfilenames = ["speech1.wav", "speech2.wav"]
    s, a, doas= narrowband_exercise("/rirs/Week_2/Part_3/part3_1.pkl.gz", speechfilenames)
    plot_pseudspectrum(a, s, "Narrowband MUSIC, 2 Sources with DOA diff < 10°", 
                        window_title="Part 3-1", stems=[128.66, 135])

def part_3_2():
    """
    Demonstrate the use of wideband MUSIC and how it improves on the narrowband implementation
    """
    speechfilenames = ["speech1.wav"]
    s, a, doas = wideband_exercise(f"/rirs/Week_2/Part_1_1/45deg.pkl.gz", speechfilenames)
    plot_pseudspectrum(a, s, "Wideband MUSIC, 1 source", window_title="Part 3-2", normalise=True, stems=[45])

def part_3_3():
    """
    Demonstrate the use of wideband MUSIC and how it improves on the narrowband implementation where the difference in DOAs
    of the sources are <10°.

    OBSERVATIONS:
        The wideband implementation definitely improves on the narrowband one
    """
    speechfilenames = ["speech1.wav", "speech2.wav"]
    s, a, doas = wideband_exercise(f"/rirs/Week_2/Part_3/part3_1.pkl.gz", speechfilenames)
    plot_pseudspectrum(a, s, "Wideband MUSIC, 2 Sources with DOA diff < 10°", window_title="Part 3-3", normalise=True, stems=[128.66, 135])

def part_3_4():
    """
    Demonstrate the use of wideband MUSIC and how it improves on the narrowband implementation for 2 sources with 90° diff in DOA

    OBSERVATIONS:
        In the wideband pseudospectrum the peaks are clearly narrower than in the narrowband spectrum.
    """
    speechfilenames = ["speech1.wav", "speech2.wav"]
    s, a, doas = wideband_exercise(f"/rirs/Week_2/Part_2_1/45_135.pkl.gz", speechfilenames)
    plot_pseudspectrum(a, s, "Wideband MUSIC, 2 Sources with DOA diff = 90°", window_title="Part 3-4", normalise=True, stems=[45, 135])

def part_3_5():
    """
    Demonstrate the use of wideband MUSIC and how it improves on the narrowband implementation for 3 sources

    OBSERVATIONS:
        In the wideband pseudospectrum the peaks are clearly narrower than in the narrowband spectrum.
    """
    speechfilenames = ["speech1.wav", "speech2.wav", "speech1.wav"]
    s, a, doas = wideband_exercise(f"/rirs/Week_2/Part_2_1/45_90_135.pkl.gz", speechfilenames)
    plot_pseudspectrum(a, s, "Wideband MUSIC, 3 Sources with DOA diffs = 45°", window_title="Part 3-5", normalise=True, stems=[45, 90, 135])

def part_4_1():
    pass

if __name__ == "__main__":
    part_1_1()
    part_1_5()
    part_2_1()
    part_3_1()
    part_3_2()
    part_3_3()
    part_3_4()
    part_3_5()
    plt.show()