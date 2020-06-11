#!/usr/bin/env python3

import numpy as np
import scipy as sp
from scipy import signal
import time
import os
import sys
import pickle
import copy
import math

from data_preprocess import data_1D_to_2D
from channel_maps import channel_label_map, channel_loc_map
from MvmtOnsetDetect import *

db_dir = os.getcwd()
ME_db_fname = "prelim_ME_db_128.pickle"
ME_Kin_db_fname = "noneeg_ME_db_128.pickle"
rej_ME_db_fname = "reject_ME_db_128.pickle"
fs = 128
ME_db = {}
ME_kin_db = {}
rej_ME_db = {}

t1 = time.time()
with open(db_dir + "/" + rej_ME_db_fname, "rb") as f:
    rej_ME_db = pickle.load(f)
with open(db_dir + "/" + ME_db_fname, "rb") as f:
    ME_db = pickle.load(f)
with open(db_dir + "/" + ME_Kin_db_fname, "rb") as f:
    ME_kin_db = pickle.load(f)
print("Loaded ME database in %f s" % (time.time()-t1))

# baseline subtraction and infs/NaNs rejection
t1 = time.time()
ME_db_norm = copy.deepcopy(ME_db)
for i in range(1,8):
    for j in range(0,900):
        try:
            signal.detrend(ME_db_norm[i][j], axis=0, overwrite_data=True)
        except ValueError as e:    # add trials with infs/NaNs to rejected db
            rej_ME_db[i][j] = 1
            
print("Baseline subtraction and infs/NaNs rejection finished in %f s" % (time.time()-t1))

# map event type to event label
# class 1: 0x600 = 1536 (elbow flexion)
# class 2: 0x601 = 1537 (elbow extension)
# class 3: 0x602 = 1538 (supination)
# class 4: 0x603 = 1539 (pronation)
# class 5: 0x604 = 1540 (hand close)
# class 6: 0x605 = 1541 (hand open)
# class 7: 0x606 = 1542 (rest)


onsetAll = np.zeros((8,900))
chElbow = np.array([87, 88, 89]) - 65    # adjust for offset as indexed in ME_kin_db
chForeArm = np.array([94]) - 65
chHand = np.arange(65, 80) - 65
plot = False
t1 = time.time()

detectOnset(ME_kin_db, onsetAll, 1, chElbow, baselineEnd=16, threshV=1, threshdV=0.01, filt=17, plot=plot)
detectOnset(ME_kin_db, onsetAll, 2, chElbow, baselineEnd=16, threshV=1, threshdV=0.01, filt=17, plot=plot)
detectOnset(ME_kin_db, onsetAll, 3, chForeArm, baselineEnd=16, threshV=1, threshdV=0.01, filt=17, plot=plot)
detectOnset(ME_kin_db, onsetAll, 4, chForeArm, baselineEnd=16, threshV=1, threshdV=0.01, filt=17, plot=plot)
detectOnsetPCA(ME_kin_db, onsetAll, 5, chHand, baselineEnd=16, threshV=1, threshdV=0.01, filt=17, plot=plot)
detectOnsetPCA(ME_kin_db, onsetAll, 6, chHand, baselineEnd=16, threshV=1, threshdV=0.01, filt=17, plot=plot)

onsetAll[7,:] = np.mean(onsetAll[1:7,:])
onsetAll = onsetAll.astype(int)

print("Found movement onset in %f s" % (time.time()-t1))


t1 = time.time()
ME_db_aligned = alignTrials(ME_db_norm, onsetAll, fs)
print("Created ME_db_aligned in %f s" % (time.time()-t1))


t1 = time.time()

num_good_trials = np.zeros(8, dtype=int) # list storing the number of good trials per class after trial rejection
ME_db_aligned_no_art = {}
for clas in range(1,8):
    ME_db_aligned_no_art[clas] = None

for clas in range(1,8):
    reject_mask = np.array(rej_ME_db[clas])
    ME_db_aligned_no_art[clas] = np.delete(ME_db_aligned[clas], np.nonzero(reject_mask == 1), axis=0)
    num_good_trials[clas] = ME_db_aligned_no_art[clas].shape[0]
        
print("Removing artifacts %f s" % (time.time()-t1))

min_num_good_trials = np.min(num_good_trials[1:])
for clas in range(1,8):
    ME_db_aligned_no_art[clas] = ME_db_aligned_no_art[clas][0:min_num_good_trials,:,:]

print(ME_db_aligned_no_art[1].shape)

CLM = channel_loc_map()
# populate the mesh with the electrodes
mesh = [ [ "" for y in range(0,9) ] for x in range(0,9) ] 
for chan in range(0,np.shape(CLM)[0]):
    mesh[CLM[chan][0]][CLM[chan][1]] = channel_label_map[chan+1]

# print the 2D mesh of channels
for x in range(0,9): 
    print(mesh[x])

t1 = time.time()
ME_db_final_2D_mesh = data_1D_to_2D(ME_db_aligned_no_art, 9, 9, CLM)
print("Converting 1D to 2D mesh takes %f s" % (time.time()-t1))

t1 = time.time()
with open("mesh_ME_db_128.pickle", "wb") as f:
    i_str = pickle.dumps(ME_db_final_2D_mesh)
    f_size = sys.getsizeof(i_str)/1048576
    f.write(i_str)
print("Finished writing %.2f MB of data to mesh_ME_db_128.pickle in %f s" % (f_size, time.time()-t1))