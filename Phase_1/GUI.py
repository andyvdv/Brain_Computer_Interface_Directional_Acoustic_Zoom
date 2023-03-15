from package.gui_utils import RIRg_GUI
from package.gui_utils import load_rirs, plot_rirs
import os as os
from matplotlib import pyplot as plt
#RIRg_GUI(exportFolder=f"{os.getcwd()}/rirs") # opens the GUI

acousticScenario = load_rirs(path=os.getcwd() + "/rirs/Week_4/part2/rirs_part2_1_0rev.pkl.gz")
acousticScenario.plot_asc()
plt.show()