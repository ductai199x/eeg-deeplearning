import numpy as np
import time
import os
import sys
import bz2
import pickle

def get_all_files(dir):
    filelist = []

    for root, dirs, files in os.walk(dir):
        for file in files:
            filelist.append(os.path.join(root,file))

    return filelist


prelim_ME_db = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]}
prelim_MI_db = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]}

pwd = os.getcwd()
processed_data_dir = pwd + "/" + "processed_data"
filelist = get_all_files(processed_data_dir)

print("Processing %d files" % (len(filelist)))

t1 = time.time()
data = None
while len(filelist) > 0:
    with open(filelist.pop(), 'rb') as f:
        data = pickle.load(f)
        f_name = os.path.basename(f.name)
        db = None
        if "ME" in f_name:
            db = prelim_ME_db
        else:
            db = prelim_MI_db
        for key in data.keys():
            db[key].extend(data[key])
            
print("Complete loading and merging maps in %f s" % (time.time()-t1))

t1 = time.time()
f = open("prelim_ME_db.pickle", "wb")
i_str = pickle.dumps(prelim_ME_db)
f_size = sys.getsizeof(i_str)/1048576
f.write(i_str)
f.close()
    
print("Finished writing %.2f MB of data to prelim_ME_db.pickle in %f s" % (f_size, time.time()-t1))

t1 = time.time()
f = open("prelim_MI_db.pickle", "wb")
i_str = pickle.dumps(prelim_MI_db)
f_size = sys.getsizeof(i_str)/1048576
f.write(i_str)
f.close()
    
print("Finished writing %.2f MB of data to prelim_MI_db.pickle in %f s" % (f_size, time.time()-t1))