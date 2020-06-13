# Deep learning with EEG

- [Deep learning with EEG](#deep-learning-with-eeg)
- [Introduction](#introduction)
- [Authors](#authors)
- [Papers](#papers)
- [Dataset](#dataset)
- [Data Extraction and Preliminary Data Processing](#data-extraction-and-preliminary-data-processing)
- [Post-preliminary Data Processing](#post-preliminary-data-processing)
- [Neural Network](#neural-network)
- [Saliency Mapping](#saliency-mapping)
- [Results](#results)
- [Footnotes](#footnotes)

# Introduction

This repository is dedicated to the attempt to classify human intentions using
EEG and Deep Learning as a part of developing a Brain-Computer Interface (BCI).

# Authors

Tai Nguyen, Liangyu Tao, Cherelle Conors

# Papers

1. Upper limb movements can be decoded from the time-domain of low-frequency EEG
   by Ofner et. al (August 2017)
2. Cascade and Parallel Convolutional Recurrent Neural Networks on EEG-Based
   Intention Recognition for Brain Computer Interface by Zhang et. al (2018)

# Dataset

The dataset used come from the paper: _Upper limb movements can be decoded from
the time-domain of low-frequency EEG_ by Ofner et. al (August 2017), and it's
available at: http://bnci-horizon-2020.eu/database/data-sets (dataset #001-2017)

It's description is also available at:
http://bnci-horizon-2020.eu/database/data-sets/001-2017/dataset_description.pdf

# Data Extraction and Preliminary Data Processing

The data extraction code is located in the `data_extraction/src` folder. You can
run the code by:

1. Create a virtual environment using `virtualenv` under `data_extraction/`:
   `virtualenv data_extraction/ --python=python3`
2. Change directory to the `data_extraction` folder: `cd data_extraction`
3. Activate the virtualenv: `source bin/activate` on Linux
4. Install necessary dependencies: `pip install -r requirements.txt`
5. Change directory to the `src` folder: `cd src`
6. Edit the path to your dataset folder
   (`database_dir = "/home/sweet/1-workdir/eeg001-2017/"`) to your own path. Run
   the data_extract script: `python data_extract.py`. (lots of data processing,
   so expect your computer working at 100%)
7. Run the create_database script: `python create_databases.py`

The final step above will create a few files:

1. `prelim_ME_db_128.pickle`: <sup>[1](#128hz)</sup> _The database of EEG signals from 64
   channels_
2. `noneeg_ME_db_128.pickle`: _The database of nonEEG signals (movement
   sensors)_
3. `reject_ME_db_128.pickle`: _The database of contain trial rejection
   information_

The current state of the code will only deal with the Motor Execution (ME)
dataset. For the Motor Imagination (MI) dataset, the MI's databases can be
created in very similar manner.

`prelim_ME_db_128.pickle` will contain of a python map with 7 keys representing
7 classes (6 movement classes + 1 rest class). The value at each key is a python
list, where each element in the list is a trial of that class. Each trial is a
timesteps x 64 matrix (64 is the number of available EEG channels).

# Post-preliminary Data Processing

The post-preliminary data processing is also located in the
`data_extraction/src` folder. You can run the code by:

1. Activate the virtualenv used in the previous section
2. Run the post_prelim_processing script: `python post_prelim_processing.py`

Running this script will produce a single file called
`mesh_ME_db_128.pickle`<sup>[2](#link)</sup>, a python map with 7 keys representing 7 classes
(6 movement classes + 1 rest class). The value at each key is a python list,
where each element in the list is a trial of that class. Each trial is a
timesteps x 9 x 9 matrix.

The post-preliminary data processing steps include, but not limited to:

- 1st-order baseline subtraction
- NaNs/Infs trial rejection
- Reject trials due to joint probability
- Reject trials due to Kurtosis
- Movement onset detection and alignment
- Converting 1D data to 2D mesh base on the location of the actual electrodes.

# Neural Network

The neural network constructed in this project comes from the paper: _Cascade
and Parallel Convolutional Recurrent Neural Networks on EEG-Based Intention
Recognition for Brain Computer Interface_ by Zhang et. al (2018). The neural
network code is located in the `neural_network/src` folder. You can run the code
by:

1. Create a virtual environment using `virtualenv` under `neural_network/`:
   `virtualenv neural_network/ --python=python3`
2. Change directory to the `neural_network` folder: `cd neural_network`
3. Activate the virtualenv: `source bin/activate` on Linux
4. Install necessary dependencies: `pip install -r requirements.txt`
5. Open Jupyter Notebook: `jupyter notebook`
6. Open the Jupyter Notebook named `neural_network.ipynb`
7. Change the `db_dir` variable to indicate the location of the
   `mesh_ME_db_128.pickle` file generated in the above section.
8. Adjust these variables: `IS_RUNNING_PAIRWISE`, `IS_SAVE_TRAINING_HISTORY`,
   and `GEN_PICKLE_INPUT_TARGET` to your own settings.
9. Run the notebook.

The notebook will run the code which does all 7 class classfication and report
the accuracies after 50 epochs (the number of epochs can be adjusted in the
variable `n_epochs`).

Finally, the confusion matrix available in our report are generated using the
`confusion_matrix.ipynb` notebook.

# Saliency Mapping

To be written...

# Results

Our results are shown in the paper attached in the file `EEG_BCI_CNN_LSTM.pdf`.
The paper includes all of our methods, data processing pipelines and final
results.

# Footnotes

<a name="128hz">1</a>: 128 means 128Hz. The original dataset is 512Hz. We downsampled to 128Hz.

<a name="link">2</a>: This database file is also available at https://drexel0-my.sharepoint.com/:u:/g/personal/tdn47_drexel_edu/EdobbPf6Qm5Cpcr-36cQz_EByAIyC44n25WX0-WuiujCog?e=1ao3bu Please cite our work if you decided to use this database.