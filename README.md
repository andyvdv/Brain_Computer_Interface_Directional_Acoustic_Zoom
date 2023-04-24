# Brain_Computer_Interface_Directional_Acoustic_Zoom

## Linear regression model

The python script of the LR model can be found in Phase_2/Week_1/LS_LOOCV.py and the main arguments that can be modified are in the main. For instance:

• K: The number of EEG trials to use. Choose from 50, 100, 200, 300, or 484.\

• max_test: The number of EEG trials that you want to test and look for their average accuracy. Set this to K if you want to use all available trials.\

• w_length: The window length, which is the number of samples to use. Choose from 1200, 600, 300, or 100, which correspond to 60s, 30s, 15s, and 5s, respectively.\

• folder_dk: The path to the d_k file to use. Replace "d_k" with the appropriate filename, such as "d_k_K100_30s" for a precomputed d_k file with K=100 and w_length=30s * 20Hz = 600.
