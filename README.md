# P&D ISSP 2023 base scripts
This repository contains base scripts for the P&D ISSP KU Leuven course, 2023 edition.

Set things up:

1. Make sure Git is installed on your machine.
2. Clone the repository with `git clone https://github.com/p-didier/pandd2023-base` in the folder of your choice.
3. Make sure Python 3.9 is installed on your machine.
4. In Visual Studio Code, select the appropriate Python environment.
5. Open a command line prompt (in Visual Studio Code or by typing `cmd` in the Windows search bar).  
6. Set the current directory to the folder where you cloned the repository (with the command `cd`).
7. Run the following in the prompt to install necessary Python packages: `pip install -r requirements.txt`.
8. All set! :-)

Repository structure:

* `./package/gui_utils.py`: scripts for the GUI. Not to be modified.
* `./package/general.py`: general functions, not GUI-related. You can add functions to that file to your convenience. Feel free to create other `.py` files as well.
* `./rirs/`: folder where your RIRs and acoustic scenario information will be stored (as Pickle archives: `name.pkl.gz`).
* `./sound_files/`: folder containing provided sound files to conduct your tests.
* `./notebook_skeleton.ipynb`: skeleton notebook to start your work from.
* `./requirements.txt`: package requirements file.
