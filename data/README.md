# PETA DATASET
## About this dataset
- PETA dataset consists of 10 contexts, each context includes bounding box of a person with attribute annotations. All base images is saved in folder /data/PETA_dataset/{context}/archive/
- Processed dataset is created by the PETA dataset in a way that divides the image into two parts: lower body and upper body. All images is saved in folder /data/PETA_dataset/{context}/processed/

##  How to use this dataset

- Access link [PETA_dataset](https://drive.google.com/file/d/10f5vkS9vfiY9BxE88InU3j_usV1B6-Rb/view?usp=sharing) and extract to folder /data/
- The image annotations (for both training and testing samples) within each context are stored in /data/PETA_dataset/{context}/Label_process.json 
- When you try to use them, you must change directory of image in all annotation_paths first.
- The best way to split dataset into 3 parts: train, val, test is using TownCentre context for validating, VIPeR for testing and others for training.
