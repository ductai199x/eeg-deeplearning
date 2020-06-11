import copy

import numpy as np
from scipy.stats import kurtosis


def rejectChannels(seqs_v_class_map, rejChan, copyLib=False):
    # copy the library or write over the old library
    if copyLib:
        seqs_v_class_map_no_artifact = copy.deepcopy(seqs_v_class_map)
    else:
        seqs_v_class_map_no_artifact = seqs_v_class_map

    # reject the previously marked epoch channels by setting the channel to 0
    for r in range(0, np.shape(rejChan)[0]):
        currClas = rejChan[r, 0]
        currTrial = rejChan[r, 1]
        currChan = rejChan[r, 2]
        # seqs_v_class_map_no_artifact[currClas][currTrial][:,currChan] = 0

    return seqs_v_class_map_no_artifact


def remove_np_arr_from_list(array, arrays):
    """
    Remove the `array` from the `list` of `arrays`
    Operates inplace on the `list` of `arrays` given

    :param array: `np.ndarray`
    :param arrays: `list:np.ndarray`
    :return: None
    """

    assert isinstance(arrays, list), f'Expected a list, got {type(arrays)} instead'
    assert isinstance(array, np.ndarray), f'Expected a numpy.ndarray, got {type(array)} instead'
    for a in arrays:
        assert isinstance(a, np.ndarray), f'Expected a numpy.ndarray instances in arrays, found {type(a)} instead'

    # Numpy ndarrays are not hashable by default, so we create
    # our own hashing algorithm. The following will do the job ...
    def _hash(a):
        return hash(a.tobytes())

    try:
        # We create a list of hashes and search for the index
        # of the hash of the array we want to remove.
        index = [_hash(a) for a in arrays].index(_hash(array))
    except ValueError as e:
        # It might be, that the array is not in the list at all.
        print(f'Array not in list. Leaving input unchanged.')
    else:
        # Only in the case of no exception we pop the array
        # with the same index/position from the original
        # arrays list
        arrays.pop(index)


def rejectTrials(seqs_v_class_map, rejTrial, copyLib=False):
    # copy the library or write over the old library
    seqs_v_class_map_no_artifact = None
    if copyLib:
        seqs_v_class_map_no_artifact = copy.deepcopy(seqs_v_class_map)
    else:
        seqs_v_class_map_no_artifact = seqs_v_class_map

    # reject the previously marked epoch channels by setting the channel to 0

    list_reject_trials = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}

    for r in range(0, np.shape(rejTrial)[0]):
        currClas = rejTrial[r, 0]
        currTrial = rejTrial[r, 1]

        seqs_v_class_map_no_artifact[currClas][currTrial][:, :] = 0

    return seqs_v_class_map_no_artifact


def markArtifactJointProb(seqs_v_class_map, nBins=20, threshold=5):
    # get number of each dimension
    nClass = len(seqs_v_class_map)
    nTrials = np.shape(seqs_v_class_map[1])[0]
    nChannel = np.shape(seqs_v_class_map[1])[2]
    nPoints = np.shape(seqs_v_class_map[1])[1]

    rejChan = np.zeros((1, 3), dtype=int)

    # marking channels to reject based on all eeg channels in an epoch (population)
    for clas in range(1, nClass + 1):
        for trial in range(0, nTrials):
            # use the timexchannel signal and calculate rejections
            signal = seqs_v_class_map[clas][trial]
            jp_z_score = calcJointProb(signal, nBins, threshold)
            rej = [x for x in range(0, nChannel) if abs(jp_z_score[x]) > threshold]

            # store any rejected channel class, trial, and channel information
            for r in range(0, np.size(rej)):
                rejChan = np.append(rejChan, [(clas, trial, rej[r])], axis=0)

    # marking channels to reject based on individual channel across all epochs 
    # (single)
    for chan in range(0, nChannel):
        signal = np.zeros((nClass * nTrials, nPoints))
        key = np.zeros((nClass * nTrials, 3), dtype=int)
        k = 0

        # get the timex(classxtrials) signal matrix describing channel in all epochs
        for clas in range(1, nClass + 1):
            for trial in range(0, nTrials):
                signal[k, :] = seqs_v_class_map[clas][trial][:, chan]
                key[k, :] = (clas, trial, chan)
                k = k + 1

        # use the timex(classxtrials) signal to calculate probability of trials and find rejections
        jp_z_score = calcJointProb(np.transpose(signal), nBins, threshold)
        rej = [x for x in range(0, nClass * nTrials) if abs(jp_z_score[x]) > threshold]

        # store any rejected channel class, trial, and channel information
        for r in range(0, np.size(rej)):
            currClas = key[rej[r], 0]
            currTrial = key[rej[r], 1]
            rejChan = np.append(rejChan, [(currClas, currTrial, chan)], axis=0)

    rejChan = np.delete(rejChan, (0), axis=0)
    return rejChan


def markArtifactKurtosis(seqs_v_class_map, threshold=5):
    # get the number of eeg channels and data points
    nClass = len(seqs_v_class_map)
    nTrials = np.shape(seqs_v_class_map[1])[0]
    nChannel = np.shape(seqs_v_class_map[1])[2]
    nPoints = np.shape(seqs_v_class_map[1])[1]

    rejChan = np.zeros((1, 3), dtype=int)

    for clas in range(1, nClass + 1):
        for trial in range(0, nTrials):
            # use the timexchannel signal and calculate kurtosis
            signal = seqs_v_class_map[clas][trial]
            kurt = kurtosis(signal, axis=0)

            # find rejections
            rej = [x for x in range(0, nChannel) if abs(kurt[x]) > threshold]

            # store any rejected channel class, trial, and channel information
            for r in range(0, np.size(rej)):
                rejChan = np.append(rejChan, [(clas, trial, rej[r])], axis=0)

    rejChan = np.delete(rejChan, (0), axis=0)
    return rejChan


def markArtifactSigVal(seqs_v_class_map, threshold=200):
    # get the number of eeg channels and data points
    nClass = len(seqs_v_class_map)
    nTrials = np.shape(seqs_v_class_map[1])[0]
    nChannel = np.shape(seqs_v_class_map[1])[2]
    nPoints = np.shape(seqs_v_class_map[1])[1]

    rejChan = np.zeros((1, 3), dtype=int)
    for clas in range(1, nClass + 1):
        for trial in range(0, nTrials):
            # use the timexchannel signal and calculate rejections
            signal = seqs_v_class_map[clas][trial]
            # calculate absolute difference from the mean for each channel
            absDev = np.absolute(signal - np.mean(signal, axis=0))

            # find rejections
            rej = [x for x in range(0, nChannel) if any(y > threshold for y in absDev[:, 1])]

            # store any rejected channel class, trial, and channel information
            for r in range(0, np.size(rej)):
                rejChan = np.append(rejChan, [(clas, trial, rej[r])], axis=0)

    rejChan = np.delete(rejChan, (0), axis=0)


def calcJointProb(signal, nBins=20, threshold=4):
    # get the number of eeg channels and data points
    nChannel = np.shape(signal)[1];
    nPts = np.shape(signal)[0];

    # calculate the total log probability for each channel
    jp = np.zeros((nChannel, 1))
    for chan in range(0, nChannel):
        sigTmp = signal[:, chan];

        counts, bin_edges = np.histogram(sigTmp, nBins, (sigTmp.min() - 1, sigTmp.max() + 1))
        datBin = np.digitize(sigTmp, bin_edges) - 1
        probaMap = counts[datBin] / nPts

        jp[chan] = -sum(np.log2(probaMap))

    jp_z_score = (jp - np.mean(jp)) / np.std(jp)
    rej = [x for x in range(0, nChannel) if jp_z_score[x] > threshold]

    return jp_z_score


def data_segmentation(seqs_v_class_map, initTime=0, finTime=700, width=100, nOverlap=0):
    bins = []
    for low in range(initTime, finTime, width - nOverlap):
        bins.append((low, low + width))

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
        [seqs_v_class_map_binned.setdefault(x, np.zeros((nTrials, np.size(bins, 0), width, nChannels))) \
         for x in range(1, nClass + 1)]  # the dictionary is 1 based, not 0 based

        # seed the new dictionary
        for clas in range(1, nClass + 1):
            for trial in range(0, nTrials):
                for b in range(0, np.size(bins, 0)):
                    seqs_v_class_map_binned[clas][trial][b] = seqs_v_class_map[clas][trial][bins[b][0]:bins[b][1]]

    # if the channels is in 2D mesh                
    else:
        # get the mesh of channels
        nX = np.shape(seqs_v_class_map[1])[2]
        nY = np.shape(seqs_v_class_map[1])[3]

        seqs_v_class_map_binned = {}
        # create a new dictionary with same classes and trials as the original one. But bin the times into bins
        [seqs_v_class_map_binned.setdefault(x, np.zeros((nTrials, np.size(bins, 0), width, nX, nY))) \
         for x in range(1, nClass + 1)]  # the dictionary is 1 based, not 0 based

        # seed the new dictionary
        for clas in range(1, nClass + 1):
            for trial in range(0, nTrials):
                for b in range(0, np.size(bins, 0)):
                    seqs_v_class_map_binned[clas][trial][b] = seqs_v_class_map[clas][trial][bins[b][0]:bins[b][1]]

    return seqs_v_class_map_binned


def data_1D_to_2D(seqs_v_class_map, nX=10, nY=11, maps=[(0, 0)]):
    # get the number of classes, trials and dimension of the data
    nClass = len(seqs_v_class_map)
    nTrials = np.shape(seqs_v_class_map[1])[0]
    nTime = np.shape(seqs_v_class_map[1])[1]
    nChannel = np.shape(seqs_v_class_map[1])[2]

    seqs_v_class_map_mesh = {}
    # create a new dictionary with same classes and trials as the original one. But bin the times into bins
    [seqs_v_class_map_mesh.setdefault(x, np.zeros((nTrials, nTime, nX, nY))) \
     for x in range(1, nClass + 1)]  # the dictionary is 1 based, not 0 based

    # seed the new dictionary
    for clas in range(1, nClass + 1):
        for trial in range(0, nTrials):
            for time in range(0, nTime):
                data = seqs_v_class_map[clas][trial][time]

                data_2D = np.zeros([nX, nY])

                # convert from 1D channels to the 2D mesh
                for chan in range(0, np.shape(maps)[0]):
                    data_2D[maps[chan][0]][maps[chan][1]] = data[chan]

                # populate the trial with the 2D mesh
                seqs_v_class_map_mesh[clas][trial][time] = data_2D

    return seqs_v_class_map_mesh
