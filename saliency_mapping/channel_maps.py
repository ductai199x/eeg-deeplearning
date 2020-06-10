import numpy as np

channel_label_map = {
    1: "F3", 2: "F1", 3: "Fz", 4: "F2",
    5: "F4", 6: "FFC5h", 7: "FFC3h", 8: "FFC1h",
    9: "FFC2h", 10: "FFC4h", 11: "FFC6h", 12: "FC5",
    13: "FC3", 14: "FC1", 15: "FCz", 16: "FC2",
    17: "FC4", 18: "FC6", 19: "FTT7h", 20: "FCC5h",
    21: "FCC3h", 22: "FCC1h", 23: "FCC2h", 24: "FCC4h",
    25: "FCC6h", 26: "FTT8h", 27: "C5", 28: "C3",
    29: "C1", 30: "Cz", 31: "C2", 32: "C4",
    33: "C6", 34: "TTP7h", 35: "CCP5h", 36: "CCP3h",
    37: "CCP1h", 38: "CCP2h", 39: "CCP4h", 40: "CCP6h",
    41: "TTP8h", 42: "CP5", 43: "CP3", 44: "CP1",
    45: "CPz", 46: "CP2", 47: "CP4", 48: "CP6",
    49: "CPP5h", 50: "CPP3h", 51: "CPP1h", 52: "CPP2h",
    53: "CPP4h", 54: "CPP6h", 55: "P3", 56: "P1",
    57: "Pz", 58: "P2", 59: "P4", 60: "PPO1h",
    61: "PPO2h", 62: "EOG left", 63: "EOG middle", 64: "EOG right"
}

def channel_loc_map():
    # empirically determined pattern
    startNdx = [0,0,0,0,1,1,2,3,4,6];
    endNdx   = [4,5,6,7,7,8,8,8,8,7];

    # initialize an empty map
    maps = np.zeros((61,2),dtype=int)
    k = 0
    # fill that map with the row and column positions of each electrode
    for i in range(0, np.size(startNdx)):
        tmpCol = np.arange(startNdx[i], endNdx[i]+1, 1).tolist()
        tmpRow = np.arange(endNdx[i], startNdx[i]-1, -1).tolist()

        for j in range(0,np.size(tmpRow)):
            maps[k][:] = [tmpRow[j], tmpCol[j]]
            k = k + 1
    
    return maps