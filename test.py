#from package.gui_utils import RIRg_GUI
#RIRg_GUI()

from package.gui_utils import load_rirs

acousticScenario = load_rirs(path="/Users/andyvandervaeren/Documents/Brain_Computer_Interface_Directional_Accounstric_Zoom/rirs/20230214_10h16m40s.pkl.gz")

roomDim = acousticScenario.roomDim

print(acousticScenario)