import os, sys
sys.path.append(os.getcwd())
from helpers import create_micsigs_week3, verify_parameters, create_micsigs
from package.gui_utils import load_rirs
from package.general import listen_to_array
import scipy.signal as ss
import scipy.linalg as LA
import numpy as np
from Week_2.music import *
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from week3_helpers import create_micsigs_wk3, das_bf, calculate_snr, gsc_td