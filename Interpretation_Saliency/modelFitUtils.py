import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def create_rand_seq_permutations(db,rand):
    seq_idx = {}
    for key in db.keys():
        if key not in seq_idx:
            seq_idx[key] = None
        if rand:
            seq_idx[key] = np.random.permutation(np.arange(len(db[key])))
        else:
            seq_idx[key] = np.arange(len(db[key]))

    return seq_idx

# create sliding window
def sliding_window(a, w = 4, o = 2, copy = False):
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides = st, shape = sh)[0::o]
    if copy:
        return view.copy()
    else:
        return view

# generate data
def genAllDat(db, b, targetClas, nClas=7, nTrials=200, nWin=1, S=16, nX=9, nY=9, rand=True):
    
    inputs = np.zeros((nWin*nTrials*nClas, S, nX, nY))
    targets = np.zeros(nWin*nTrials*nClas, dtype=int)
    wins = np.zeros(nWin*nTrials*nClas, dtype=int)
    seq_perms = create_rand_seq_permutations(db,rand)
    k = 0
    for trial in range(0,nTrials):
        for clas in targetClas:
            perm = seq_perms[clas][trial]
            tmp = db[clas][perm]

            for win in range(0,nWin):
                inputs[k] = tmp[b[win,:]]
                targets[k] = clas
                wins[k] = win
                k=k+1
    return inputs, targets, wins

# generate testing and training datasets for training neural network
def setupDataForNetwork(Data, classes, S=8, overlap=4, nTrials=1):
  nX = Data[1].shape[2]
  nY = Data[1].shape[3]
  nPointsPerTrial = Data[1].shape[1]

  # convert ehd data into sliding windows
  windows = sliding_window(np.arange(nPointsPerTrial), S, overlap)
  nWin = windows.shape[0]
  inputs, targets, wins  = genAllDat(Data, windows, classes, len(classes), nTrials, nWin, S, nX, nY)
  # shuffle the dataset
  inputs, targets = shuffle(inputs, targets)

  # convert target labels to be in ascending order starting from 1
  j = 1
  for c in classes:
    targets[:] = [x if x != c else j for x in targets]
    j = j+1
  # split into training and testing set
  X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2)
  # add an depth dimension
  X_train = np.expand_dims(X_train, axis=-1)
  X_test = np.expand_dims(X_test, axis=-1)

  # set any channel with nans to 0
  where_are_NaNs = np.isnan(X_train)
  X_train[where_are_NaNs] = 0
  where_are_NaNs = np.isnan(X_test)
  X_test[where_are_NaNs] = 0

  return X_train, X_test, y_train, y_test