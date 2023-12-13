# DeepGCF for Three Species

The code contained in this repository amends code files from https://github.com/liangend/DeepGCF so that DeepGCF can be applied to learn the functional conservation between pig, human, and a third species based on the epigenome profiles from these three species.

Only three code files from the original DeepGCF are amended: 
- shared_deepgcf_three_species.py amends https://github.com/liangend/DeepGCF/blob/main/src/shared_deepgcf.py (this code defines the neural network architecture).
- train_deepgcf_three_species.py amends https://github.com/liangend/DeepGCF/blob/main/src/train_deepgcf.py
- predict_deepgcf_three_species.py amends https://github.com/liangend/DeepGCF/blob/main/src/predict_deepgcf.py

These three files are amended so that a Pseudo-Siamese Network with three sub-networks, one for each species, can be trained and used for prediction. These three sub-networks then all feed into the final network which calculates the DeepGCF score.

The code files contain comments which indicate where I made changes to the original code to adapt it to be compatible witha third species. 

The code was edited to explicitly call on GPU computing, but if this is not necessary, then the comments in the code explain what lines can be removed.

