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

def music_exercise(scenarioPath, speechfilenames, wideband=True):
    # Load the scenario and retrieve the parameters
    acousticScenario = load_rirs(os.getcwd() + scenarioPath)
    nsources = acousticScenario.RIRsAudio.shape[2]
    nmics = acousticScenario.nMicsPerArray
    d = acousticScenario.distBwMics
    # Verify the sampling frequency is correct and the right amount of sound files are passed
    verify_parameters(acousticScenario, 44100, speechfilenames)
    # Obtain the microphone signals
    micsigs, _, _, _ = create_micsigs(acousticScenario, nmics, speechfilenames, duration=10)
    # Stack the STFTs of the microphone signals
    nfft = 1024
    noverlap = 512
    print(micsigs.shape)
    S, freqs_list = stack_stfts(micsigs, acousticScenario.fs, nfft, noverlap)
    # Define the angles to commpute the pseudospectrum for
    thetas = np.arange(0, 180, 0.5)
    # Compute the MUSIC pseudospectrum and DOAs
    spectrum, doas = music_wideband(S, nmics, nsources, freqs_list, d, thetas) if wideband else music_narrowband(S, nmics, nsources, freqs_list, d, find_max_bin(S), thetas)
    return spectrum, thetas, doas

def part_1_1(normalise=True):
    speechfilenames = ["speech1.wav"]
    source_doas = [0, 45, 90, 135, 158, 180]
    fig, axs = plt.subplots((len(source_doas) // 2), 2, sharex=True, sharey=normalise)
    for i, deg in enumerate(source_doas):
        s, a, doas = music_exercise(f"/rirs/Week_2/Part_1_1/{deg}deg.pkl.gz", speechfilenames, wideband=False)
        # Normalise the spectrum to a range of [0,100]
        s = (s / s[np.argmax(s)]) * 100 if normalise else s
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
        s, a, doas = music_exercise(f"/rirs/Week_2/Part_1_5/{deg}deg.pkl.gz", speechfilenames, wideband=False)
        # Normalise the spectrum to a range of [0,100]
        s = (s / s[np.argmax(s)]) * 100 if normalise else s
        axis = axs[i // 2][i % 2]
        axis.plot(a, s, 'b', label=f"P(θ)")
        axis.stem(deg, max(s), markerfmt='', linefmt='r--', label=f"{deg}°")
        axis.legend()
    fig.suptitle('Narrowband MUSIC Pseudospectrums', fontweight='bold')
    fig.text(0.02, 0.5, 'Power (Normalised)' if normalise else 'Power', va='center', rotation='vertical', fontweight='bold', fontsize='12')
    fig.text(0.5, 0.04, 'Angle (θ)', ha='center', va='center', fontweight='bold', fontsize='12')

def part_2_1(normalise=True):
    """
    Narrowband MUSIC with 2 sources, one at 45deg and the other at 135deg
    """
    speechfilenames = ["speech1.wav", "speech2.wav"]
    S = []
    A = []
    s, a, doas= music_exercise("/rirs/Week_2/Part_2_1/45_135.pkl.gz", speechfilenames, wideband=False)
    S.append(s)
    A.append(a)
    s, a, doas= music_exercise("/rirs/Week_2/Part_3/part3_1.pkl.gz", speechfilenames, wideband=False)
    S.append(s)
    A.append(a)
    plot_multiple(A, S, "Narrowband MUSIC, 2 sources", cols=1, rows=2, stems=[[45, 135], [128.66, 135]], extra=["Big DOA diff", "Small DOA diff"])
    # plot_pseudspectrum(a, s, "Narrowband MUSIC, 2 sources with DOA diff 90°", 
    #                     window_title="Part 2", stems=[45, 135])

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
    
    plot_pseudspectrum(a, s, "Narrowband MUSIC, 2 Sources with DOA diff < 10°", 
                        window_title="Part 3-1", stems=[128.66, 135])

def part_3_2():
    """
    Demonstrate the use of wideband MUSIC and how it improves on the narrowband implementation
    """
    speechfilenames = ["speech1.wav"]
    s, a, doas = music_exercise(f"/rirs/Week_2/Part_1_1/45deg.pkl.gz", speechfilenames)
    plot_pseudspectrum(a, s, "Wideband MUSIC, 1 source", window_title="Part 3-2", normalise=True, stems=[45])

def part_3_3():
    """
    Demonstrate the use of wideband MUSIC and how it improves on the narrowband implementation where the difference in DOAs
    of the sources are <10°.

    OBSERVATIONS:
        The wideband implementation definitely improves on the narrowband one
    """
    S = []
    A = []
    speechfilenames = ["speech1.wav", "speech2.wav"]
    
    s, a, doas = music_exercise(f"/rirs/Week_2/Part_2_1/45_135.pkl.gz", speechfilenames)
    S.append(s)
    A.append(a)
    s, a, doas = music_exercise(f"/rirs/Week_2/Part_3/part3_1.pkl.gz", speechfilenames)
    S.append(s)
    A.append(a)
    plot_multiple(A, S, "Wideband MUSIC, 2 sources", cols=1, rows=2, stems=[[45, 135], [128.66, 135]], extra=["Big DOA diff", "Small DOA diff"])

    # plot_pseudspectrum(a, s, "Wideband MUSIC, 2 Sources with DOA diff < 10°", window_title="Part 3-3", normalise=True, stems=[128.66, 135])

def part_3_4():
    """
    Demonstrate the use of wideband MUSIC and how it improves on the narrowband implementation for 2 sources with 90° diff in DOA

    OBSERVATIONS:
        In the wideband pseudospectrum the peaks are clearly narrower than in the narrowband spectrum.
    """
    speechfilenames = ["speech1.wav", "speech2.wav"]
    s, a, doas = music_exercise(f"/rirs/Week_2/Part_2_1/45_135.pkl.gz", speechfilenames)
    plot_pseudspectrum(a, s, "Wideband MUSIC, 2 Sources with DOA diff = 90°", window_title="Part 3-4", normalise=True, stems=[45, 135])

def part_3_5():
    """
    Demonstrate the use of wideband MUSIC and how it improves on the narrowband implementation for 3 sources

    OBSERVATIONS:
        In the wideband pseudospectrum the peaks are clearly narrower than in the narrowband spectrum.
    """
    speechfilenames = ["speech1.wav", "speech2.wav", "speech1.wav"]
    s, a, doas = music_exercise(f"/rirs/Week_2/Part_2_1/45_90_135.pkl.gz", speechfilenames)
    plot_pseudspectrum(a, s, "Wideband MUSIC, 3 Sources with DOA diffs = 45°", window_title="Part 3-5", normalise=True, stems=[45, 90, 135])

def part_4():
    """
    Demonstrate the effect of reverberation on the MUSIC algorithm for different source distances from the
    microphone array. Mic array's bottom is at (4,4). Sources start at (1,1) then (2,2) and finally (3,3).
    Room dimension is (5,5)

    OBSERVATIONS:
        For case 1 and 2 the MUSIC algorithm is unable to identify the DOA of the source from a peak in the 
        pseudospectrum. The distance between the source and the array is large compared to the dimensions of
        the room and thus the reverb has a major influence on the signal received by the mics.
        For case 3 where the source is at (3,3) the pseudospectrum starts showing a peak at the correct angle.
    """
    speechfilenames = ["speech1.wav"]
    S = []
    A = []
    s, a, doas = music_exercise(f"/rirs/Week_2/Part_4_1/135deg0.5rev.pkl.gz", speechfilenames)
    # plot_pseudspectrum(a, s, "Wideband MUSIC, 1 Sources with T60 = 0.5", window_title="Part 4-1", normalise=True, stems=[135])
    S.append(s)
    A.append(a)
    s, a, doas = music_exercise(f"/rirs/Week_2/Part_4_2/135deg0.5rev22pos.pkl.gz", speechfilenames)
    # plot_pseudspectrum(a, s, "Wideband MUSIC, 1 Sources with T60 = 0.5", window_title="Part 4-2", normalise=True, stems=[135])
    S.append(s)
    A.append(a)
    s, a, doas = music_exercise(f"/rirs/Week_2/Part_4_2/135deg0.5rev33pos.pkl.gz", speechfilenames)
    S.append(s)
    A.append(a)
    s, a, doas = music_exercise(f"/rirs/Week_2/Part_4_2/135deg0.5revveryclose.pkl.gz", speechfilenames)
    # plot_pseudspectrum(a, s, "Wideband MUSIC, 1 Sources with T60 = 0.5", window_title="Part 4-3", normalise=True, stems=[135])
    S.append(s)
    A.append(a)
    plot_multiple(A, S, 
                    "Wideband MUSIC, 1 source with T60 = 0.5, DOA = 135°", 
                    stems=[[135],[135],[135],[135]],
                    extra=["Dist = 4.24m", "Dist = 2.83m", "Dist = 1.41m", "Dist = 0.707m"])



if __name__ == "__main__":
    # part_1_1()
    # part_1_5()
    # part_2_1()
    # part_3_1()
    # part_3_2()
    part_3_3()
    # part_3_4()
    # part_3_5()
    # part_4()
    plt.show()