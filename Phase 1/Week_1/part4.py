from helpers import DOA_corr
from package.gui_utils import load_rirs
import os

degs = ["90", "135"]
t60s = ["0.00", "0.25", "0.50", "0.75", "1.00", "1.50", "2.00"]
for deg in degs:
    for t60 in t60s:
        print(f"DOA Estimation for source at {deg} degrees with T60 = {t60}")
        acousticScenario = load_rirs(path= f"{os.getcwd()}/rirs/2mic10cm{deg}deg{t60}reverb441.pkl.gz")
        DOA_corr(acousticScenario)
