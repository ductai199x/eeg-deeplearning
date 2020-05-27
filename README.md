# 1. Deep learning with EEG

1. [Deep learning with EEG](#deep-learning-with-eeg)
2. [Introduction](#introduction)
3. [Dataset](#dataset)
4. [Data Extraction](#data-extraction)
5. [Neural Network](#neural-network)

# 2. Introduction

This repository is dedicated to the attempt to classify human intentions using
EEG and Deep Learning as a part of developing a Brain-Computer Interface (BCI).

# 3. Papers

1. Upper limb movements can be decoded from the time-domain of low-frequency EEG by Ofner et. al (August 2017)
2. Cascade and Parallel Convolutional Recurrent Neural Networks on EEG-Based Intention Recognition for Brain Computer Interface by Zhang et. al (2018)

# 4. Dataset

The dataset used come from the paper: _Upper limb movements can be decoded from
the time-domain of low-frequency EEG_ by Ofner et. al (August 2017), and it's
available at: http://bnci-horizon-2020.eu/database/data-sets (dataset #001-2017)

It's description is also available at: http://bnci-horizon-2020.eu/database/data-sets/001-2017/dataset_description.pdf

# 5. Data Extraction

The data extraction code is located in the `data_extraction/src` folder. You can run the code by:

1. Create a virtual environment using `virtualenv` under `data_extraction/`: `virtualenv data_extraction/ --python=python3`
2. Change directory to the `data_extraction` folder: `cd data_extraction`
3. Activate the virtualenv: `source bin/activate` on Linux
4. Install necessary dependencies: `pip install -r requirements.txt`
5. Change directory to the `src` folder: `cd src`
6. Edit the path to your dataset folder (`database_dir = "/home/sweet/1-workdir/eeg001-2017/"`). Run the data_extract script: `python data_extract.py`. (lots of data processing, so expect your computer working at 100%)
7. Run the create_database script: `python create_databases.py`

The final step above will create 2 files: `prelim_ME_db.pickle` and `prelim_MI_db.pickle`. Each dataset will contain of a python map with 7 keys representing 7 classes (6 movement classes + 1 rest class). The value at each key is a python list, where each element in the list is a trial of that class. Each trial is a M x N data mesh, where the data has been mapped from 1D to 2D using the locations of the electrodes on the head.

# 6. Neural Network

The neural network constructed in this project comes from the paper: _Cascade and Parallel Convolutional Recurrent Neural Networks on EEG-Based Intention Recognition for Brain Computer Interface_ by Zhang et. al (2018).
The neural network code is located in the `neural_network/src` folder. You can run the code by:

1. Create a virtual environment using `virtualenv` under `neural_network/`: `virtualenv neural_network/ --python=python3`
2. Change directory to the `neural_network` folder: `cd neural_network`
3. Activate the virtualenv: `source bin/activate` on Linux
4. Install necessary dependencies: `pip install -r requirements.txt`
... to be completed