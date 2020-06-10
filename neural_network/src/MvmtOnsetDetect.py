import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from scipy import signal
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
import math
from channel_maps import channel_label_map, channel_loc_map

def alignTrials(ME_db_norm, onsetAll, fs=128):
    nPtInit = np.shape(ME_db_norm[1])[1]
    nClass = len(ME_db_norm)
    nTrials = np.shape(ME_db_norm[1])[0]
    nTime = fs*2+1
    nChan = np.shape(ME_db_norm[1])[2]
    display([nPtInit, nClass, nTrials, nTime, nChan])

    ME_kin_db_New = {}
    for key in range(1,8):
        ME_kin_db_New[key] = np.zeros((nTrials,fs*2+1,nChan))

    for clas in range(1,nClass+1):
        for trial in range(0,nTrials):
            tmpDat = ME_db_norm[clas][trial]
            tmpRepEnd = np.ones(np.shape(tmpDat))*tmpDat[nPtInit-1,:]
            tmpRepStart = np.ones(np.shape(tmpDat))*tmpDat[0,:]
            padDat = np.concatenate((tmpRepStart, tmpDat, tmpRepEnd))

            tSpan = range(int(onsetAll[clas,trial]-fs+nPtInit),int(onsetAll[clas,trial]+fs+1+nPtInit))

            ME_kin_db_New[clas][trial] = padDat[tSpan,:]
    return ME_kin_db_New

def detectOnset(seqs_v_class_map, onsetAll, classes=1, channels=[86], baselineEnd=32, threshV=1, threshdV=0.05, filt=17, plot=False):
    nTrials = np.shape(seqs_v_class_map[classes])[0]
    nChan = np.size(channels)
    
    if plot:
        plt.figure(figsize = (15,5))
    
    for i in range(0,nTrials):
        for c in range(0,nChan):
            seqs_v_class_map[classes][i][:,channels[c]] = seqs_v_class_map[classes][i][:,channels[c]] -np.mean(seqs_v_class_map[classes][i][0:baselineEnd,channels[c]])
        
        tmp = abs(np.sum(seqs_v_class_map[classes][i][:,channels],axis=1))
        tmp = savgol_filter(tmp, filt, 3)
        midInc = next(x for x, val in enumerate(tmp) if val > np.max(tmp)/4)
        tmp2 = np.gradient(tmp)
        try:
            onsetUpper = midInc-next(x for x, val in enumerate(tmp[midInc:None:-1]) if val < threshV)
            onset = onsetUpper-next(x for x, dval in enumerate(tmp2[onsetUpper:None:-1]) if dval < threshdV)
        except:
            onset=0
        
        onsetAll[classes][i] = onset
        
        if plot:
            plt.plot(tmp, label=str(i))
            plt.plot([onset, onset],[0, 10], label=str(i), color=plt.gca().lines[-1].get_color())
            plt.legend()
    return onsetAll

def detectOnsetPCA(seqs_v_class_map, onsetAll, classes=5, channels=[66], baselineEnd=32, threshV=1, threshdV=0.05, filt=17, plot=False):
    nTrials = np.shape(seqs_v_class_map[classes])[0]
    nChan = np.size(channels)
    
    if plot:
        plt.figure(figsize = (15,5))
    
    for i in range(0,nTrials):
        X = seqs_v_class_map[classes][i][:,channels]
        np.shape(X)
        pca = PCA(n_components=np.shape(X)[1])
        PC = pca.fit_transform(X)
        #display(np.mean(PC[0:baselineEnd,0]))
        
        tmp = abs(PC[:,0]-np.mean(PC[0:baselineEnd,0]))
        
        tmp = savgol_filter(tmp, filt, 3)
        midInc = next(x for x, val in enumerate(tmp) if val > np.max(tmp)/4)
        tmp2 = np.gradient(tmp)
        try:
            onsetUpper = midInc-next(x for x, val in enumerate(tmp[midInc:None:-1]) if val < threshV)
            onset = onsetUpper-next(x for x, dval in enumerate(tmp2[onsetUpper:None:-1]) if dval < threshdV)
        except:
            onset=0
        
        onsetAll[classes][i] = onset
        
        if plot:
            plt.plot(tmp, label=str(i))
            plt.plot([onset, onset],[0, 1], label=str(i))
            plt.legend()
    return onsetAll

def plotMRCP(ME_db_norm, ch=[14, 27, 30, 31]):
    b, a = signal.butter(3, 0.4, 'lp')
    nChan = len(ch)
    nPt = np.shape(ME_db_norm[1])[1]
    nClas = len(ME_db_norm)
    nTrial = np.shape(ME_db_norm[1])[0]
    
    mean = np.zeros((nChan,nPt,nClas))
    std = np.zeros((nChan,nPt,nClas))
    for c in range(0, nChan):
        for i in range(1,nClas+1):
            for j in range(0,nTrial):
                mean[c,:,i-1] = signal.lfilter(b, a, ME_db_norm[i][j][:,ch[c]])+mean[c,:,i-1]

    for c in range(0, nChan):
        for i in range(1,nClas+1):
            for j in range(0,nTrial):
                std[c,:,i-1] = np.square(signal.lfilter(b, a, ME_db_norm[i][j][:,ch[c]])-mean[c,:,i-1])+std[c,:,i-1]

    fig, (axs) = plt.subplots(2, 2, figsize = (8,8))
    for c in range(0, nChan):
        for i in range(0,nClas):
            axs[math.floor(c/2), c%2].plot(mean[c,:,i]/nTrial,label=i)
        #axs[math.floor(c/2), c%2].set_ylim([-0.25,0.25])
        axs[math.floor(c/2), c%2].set_xlim([0,nPt])
        axs[math.floor(c/2), c%2].legend()
        axs[math.floor(c/2), c%2].set_title(channel_label_map[ch[c]+1])

    fig, (axs) = plt.subplots(2, 2, figsize = (8,8))
    for c in range(0, nChan):
        for i in range(0,nClas):
            axs[math.floor(c/2), c%2].plot(np.sqrt(std[c,:,i]/nTrial)/np.sqrt(nTrial),label=channel_label_map[c+1])
        #axs[math.floor(c/2), c%2].set_ylim([-0.25,0.25])
        axs[math.floor(c/2), c%2].set_xlim([0,nPt])
        axs[math.floor(c/2), c%2].legend()