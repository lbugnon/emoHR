# emoHR: Dimensional Affect Recognition from HRV

This is a distribution of the source code used in: 

L.A. Bugnon, R.A. Calvo and D.H. Milone, "Dimensional Affect Recognition from HRV: an Approach Based on Supervised SOM and ELM", (to appear in) IEEE Transactions on Affective Computing , 2017. 

This code can be used, modified or distributed for academic purposes under GNU GPL. 

Supervised Self-Organizing Maps (sSOM) and two types of Extreme Learning Machines (kELM and nELM) are provided. For a very quick test, you can try these methods in a web-demo application: http://sinc.unl.edu.ar/web-demo/dimensional-affect-recognition/. The only requirement is a web browser. 

To use this code, please run the ‘main.m’ with Matlab software (R2013 or higher). In this implementation, methods are trained and tested using a reduced feature set extracted from RECOLA dataset. Only Heart Rate Variability (HRV) is used as input to estimate dimensional affect targets: arousal and valence. The outputs of the script are: the correlation concordance coefficient (ccc), the classifiers outputs and the graphical interpretation by sSOM.
 
Classifier hyperparameters are in 'config/'. This parameters can be easily modified in the 'parameters' structure. Other features can be used by placing them in the 'features/' folder, just replicate the data structure used in methods.  

The HRV features can be extracted directly from the HR signal provided in the RECOLA dataset. If you want to do so, please download the dataset from https://diuf.unifr.ch/diva/recola/ and set the flag ‘forceFeatureExtraction=true’ in ‘main.m’. From the AVEC_2016, copy the folders 'recordings_physio' and 'ratings_gold_standard' into the 'recola' folder.

sinc(i) - www.sinc.unl.edu.ar
Leandro Bugnon, Rafael Calvo, Diego Milone
_lbugnon@sinc.unl.edu.ar_
