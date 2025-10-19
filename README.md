This repository includes two models:
1. SHS-GAN
2. 3D CNN hyperspectral segmentor

For training the GAN, download 2 data sets:
   1. RGB data set from:
   2. HS data set (ICVL) from:
Next, train the GAN model in SHS_GAN.py, model parameters will be saved.
For creating the synthesis HS dataset, run the trained model in SYNTHESIS_HS_DATASET.py

For training the 3D CNN segmentor without pretraining, download the Pain Tree hyperspectral dataset and run the model in 3DCNN.py
For training the 3D CNN segmentor with pretaining on the synthesis dataset and finetuning on the segmentation task, run the model in pretraing.py
