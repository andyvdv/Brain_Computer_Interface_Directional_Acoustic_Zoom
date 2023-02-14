import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from package.gui_utils import RIRg_GUI
import simpleaudio as sa


def check_plot_tdoas(doaEstTarget, doaEstAll, asc: RIRg_GUI):
    """
    Generates a plot of the acoustic scenario to check the validity of TDOA
    estimation obtained, e.g., via the frequency-domain MUSIC algorithm.

    Paramters
    ---------
    doaEstTarget : list of floats
        Estimated DOA for target talker(s).
    doaEstAll : list of floats
        All estimated DOAs (target talker(s) and noise source(s)).
    asc : RIRg_GUI class instance
        Acoustic scenario parameters.
    """

    def _get_line(doaRadiants, startCoords, roomDim):
        """
        Get 2D line (x and y coordinates) from DOA.
        """
        x0, y0 = startCoords[0], startCoords[1]
        x1 = x0 + roomDim * np.cos(doaRadiants)
        y1 = y0 + roomDim * np.sin(doaRadiants)
        return [x0, x1], [y0, y1]

    # Check inputs
    if isinstance(doaEstAll, float):
        doaEstAll = [doaEstAll]  # convert to list
    # Plot room
    fig = asc.plot_asc()
    # Plot DOA estimates
    for ii in range(len(doaEstAll)):
        x, y = _get_line(
            doaRadiants=doaEstAll[ii] + np.pi / 2,  # align to space orientation
            startCoords=np.mean(asc.micsCoords, axis=0),
            roomDim=asc.roomDim
        )
        if doaEstAll[ii] in doaEstTarget:
            rayColor = 'g'  # indicate selected (target) DOAs with green
        else:
            rayColor = '0.7'
        plt.plot(x, y, '--', color=rayColor)
        plt.text(
            np.mean(x),
            np.mean(y),
            f'$\\hat{{\\theta}}_{ii+1}$={np.round(doaEstAll[ii] * 180 / np.pi, 2)}$^\\circ$',
            c=rayColor
        )
    return fig


def select_latest_rir(path):
    """
    Returns the full path to the latest RIR that was generated (via the
    RIR-generating GUI) in the folder `path`.
    """

    p = Path(path).glob('**/*')
    files = [x for x in p if x.is_file()]
    if str(files[-1])[-2:] == 'gz':
        rirFile = files[-1]
    else:
        rirFile = files[-2]
    
    return rirFile


def oracle_choice_doa(DOAestimates, asc: RIRg_GUI):
    """
    Select the DOA(s) corresponding to the target speaker(s) among all
    estimate DOAs containted in `DOAestimates`, based the information contained
    in the RIR-GUI output object `asc`.
    """

    coordinatesTalkers = asc.audioCoords
    coordinatesNoises = asc.noiseCoords
    coordinatesMicArray = np.mean(np.array(asc.micsCoords), axis=0)

    oracleDOAtalkers = np.zeros(len(coordinatesTalkers))
    for ii in range(len(coordinatesTalkers)):
        oracleDOAtalkers[ii] = np.arcsin(
            (coordinatesMicArray[0] - coordinatesTalkers[ii][0]) /\
            np.linalg.norm(coordinatesMicArray - coordinatesTalkers[ii])
        )

    oracleDOAnoises = np.zeros(len(coordinatesNoises))
    for ii in range(len(coordinatesNoises)):
        oracleDOAnoises[ii] = np.arcsin(
            (coordinatesMicArray[0] - coordinatesNoises[ii][0]) /\
            np.linalg.norm(coordinatesMicArray - coordinatesNoises[ii])
        )

    chosenDOAs = np.zeros_like(oracleDOAtalkers)
    for ii in range(len(oracleDOAtalkers)):
        chosenDOAs[ii] = DOAestimates[
            (np.abs(DOAestimates - oracleDOAtalkers[ii])).argmin()
        ]

    return chosenDOAs, oracleDOAtalkers


def listen_to_array(audioArray: np.ndarray, fs):
    """Plays back a sounds from an array."""
    # Ensure correct input type and orientation
    if isinstance(audioArray, list):
        audioArray = np.array(audioArray)
    if len(audioArray.squeeze().shape) > 2:
        raise ValueError(f'The provided audio array has too many dimensions (shape: {audioArray.shape}).')
    elif len(audioArray.shape) == 2:  # stereo
        if audioArray.shape[0] < audioArray.shape[1]:
            audioArray = audioArray.T  # transpose if needed
        nChannels = 2
    else:  # mono
        nChannels = 1

    # Prep for playing
    audioArray *= 32767 / np.amax(abs(audioArray), axis=0)
    audioArray = audioArray.astype(np.int16)

    # Play until done
    playObj = sa.play_buffer(
        audio_data=audioArray,
        num_channels=nChannels,
        bytes_per_sample=2,
        sample_rate=fs
    )
    playObj.wait_done()