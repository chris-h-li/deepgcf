# DeepGCF for Three Species

The code contained in this repository amends code files from https://github.com/liangend/DeepGCF so that DeepGCF can be applied to learn the functional conservation between pig, human, and a third species based on the epigenome profiles from these three species.

## Overview of Code Amendments
Only three code files from the original DeepGCF are amended: 
- shared_deepgcf_three_species.py amends https://github.com/liangend/DeepGCF/blob/main/src/shared_deepgcf.py (this code defines the neural network architecture).
- train_deepgcf_three_species.py amends https://github.com/liangend/DeepGCF/blob/main/src/train_deepgcf.py
- predict_deepgcf_three_species.py amends https://github.com/liangend/DeepGCF/blob/main/src/predict_deepgcf.py

These three files are amended so that a Pseudo-Siamese Network with three sub-networks, one for each species, can be trained and used for prediction. These three sub-networks then all feed into the final network which calculates the DeepGCF score.

## Running DeepGCF for Three Species
To run the DeepGCF model from start to finish with three species, first follow the first three tutorials here: https://github.com/liangend/DeepGCF/tree/main/tutorial. In addition to doing this pre-processing for human and pig data, also conduct similar steps for data from the third species. This will involve writing new code to create non-orthologous regions across the three species.

Then, follow tutorials 4 and 5 in the original repository (https://github.com/liangend/DeepGCF/blob/main/tutorial/), except first replace shared_deepgcf.py, train_deepgcf.py, and predict_deepgcf.py in the src folder with the amdended versions of these code files contained in this repository.

### Training (Tutorial 4)
When running train_deepgcf_three_species.py for **hyperparameter search** for training, there are some additional parameters that must be specified in addition to those specified in the tutorial 4 hyperparameter search example:
- G: path to third species' orthologous training data file
- H: path to third species' shuffled/non-orthologous training data file
- I: path to third species' orthologous validation data file
- J: path to third species' shuffled/non-orthologous validation data file
- spf: number of features for species 3 in input vector
  
When running train_deepgcf_three_species.py to fit a model using **pre-specified hyperparameters**, then the following parameters must be provided in addition to the those contained in the tutorial 4 example:
- G: path to third species' orthologous training data file
- H: path to third species' shuffled/non-orthologous training data file
- I: path to third species' orthologous validation data file
- J: path to third species' shuffled/non-orthologous validation data file
- spf: number of features for species 3 in input vector
- nnsp1: number of neurons in first hidden layer of third species' sub-network
- nnsp2: number of neurons in second hidden layer of third species' sub-network

### Prediction (Tutorial 5)
When running predict_deepgcf_three_species.py, there are some additional parameters that must be specified in addition to those specified in tutorial 5:
- SP: path to third species' feature data file
- spf: number of features for species 3 in input vector

## A couple final notes:
- More details on the additional parameters required when training and predicting using three species DeepGCF can be found in the code files themselves.
- The code files contain comments which indicate where I made changes to the original code to adapt it to be compatible witha third species. 
- The code was edited to explicitly call on GPU computing, but if this is not necessary, then the comments in the code explain what lines can be removed.

