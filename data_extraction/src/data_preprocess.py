import numpy as np
from scipy import signal
import biosig
import time
import pickle
import bz2
import sys

def data_segmentation(seqs_v_class_map, initTime=0, finTime=700, width=100, nOverlap=0):
    
    bins = []
    for low in range(initTime, finTime, width-nOverlap):
        bins.append((low, low+width))

    # get the number of classes, trials and dimension of the data
    nClass = len(seqs_v_class_map)
    nTrials = np.shape(seqs_v_class_map[1])[0]
    nDim = np.size(np.shape(seqs_v_class_map[1]))
    
    # if the channels is in 1D vector
    if nDim == 3:
        # get the number of channels
        nChannels = np.shape(seqs_v_class_map[1])[2]    

        seqs_v_class_map_binned = {}
        # create a new dictionary with same classes and trials as the original one. But bin the times into bins
        [seqs_v_class_map_binned.setdefault(x, np.zeros((nTrials, np.size(bins,0), width, nChannels))) \
         for x in range(1,nClass+1)]# the dictionary is 1 based, not 0 based

        # seed the new dictionary
        for clas in range(1,nClass+1):
            for trial in range(0,nTrials):
                for b in range(0,np.size(bins,0)):
                    seqs_v_class_map_binned[clas][trial][b] = seqs_v_class_map[clas][trial][bins[b][0]:bins[b][1]]
    
    # if the channels is in 2D mesh                
    else:
        # get the mesh of channels
        nX = np.shape(seqs_v_class_map[1])[2]
        nY = np.shape(seqs_v_class_map[1])[3]

        seqs_v_class_map_binned = {}
        # create a new dictionary with same classes and trials as the original one. But bin the times into bins
        [seqs_v_class_map_binned.setdefault(x, np.zeros((nTrials, np.size(bins,0), width, nX, nY))) \
         for x in range(1,nClass+1)]# the dictionary is 1 based, not 0 based

        # seed the new dictionary
        for clas in range(1,nClass+1):
            for trial in range(0,nTrials):
                for b in range(0,np.size(bins,0)):
                    seqs_v_class_map_binned[clas][trial][b] = seqs_v_class_map[clas][trial][bins[b][0]:bins[b][1]]

    return seqs_v_class_map_binned

def data_1D_to_2D(seqs_v_class_map, nX = 10, nY = 11, maps = [(0,0)]):

    # get the number of classes, trials and dimension of the data
    nClass = len(seqs_v_class_map)
    nTrials = np.shape(seqs_v_class_map[1])[0]
    nTime = np.shape(seqs_v_class_map[1])[1]
    nChannel = np.shape(seqs_v_class_map[1])[2]

    seqs_v_class_map_mesh = {}
    # create a new dictionary with same classes and trials as the original one. But bin the times into bins
    [seqs_v_class_map_mesh.setdefault(x, np.zeros((nTrials, nTime, nX, nY))) \
    for x in range(1,nClass+1)]# the dictionary is 1 based, not 0 based

    # seed the new dictionary
    for clas in range(1,nClass+1):
        for trial in range(0,nTrials):
            for time in range(0,nTime):
                data = seqs_v_class_map[clas][trial][time]

                data_2D = np.zeros([nX, nY])
                
                # convert from 1D channels to the 2D mesh
                for chan in range(0,nChannel):
                    data_2D[maps[chan][0]][maps[chan][1]] = data[chan]
                
                # populate the trial with the 2D mesh
                seqs_v_class_map_mesh[clas][trial][time] = data_2D
    
    return seqs_v_class_map_mesh
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    