from helpers import DOA_corr, TDOA_corr
from package.gui_utils import load_rirs
import os, itertools

speechfiles = ["speech1.wav", "whitenoise_signal_2.wav"]
acousticScenario = load_rirs(path= f"{os.getcwd()}/rirs/part5-1.pkl.gz")
nmics = 5
# crosscorr = TDOA_corr(acousticScenario, nmics, 0, 1, speechfiles)

for j,k in itertools.combinations(range(0,nmics), 2):
    print(f"Computing crosscorrelation for mics {j} and {k}")
    crosscorr, sample_diff = TDOA_corr(acousticScenario, nmics, j, k, speechfiles)
    print(f"Sample diff for mics {j} and {k} is {sample_diff}")
