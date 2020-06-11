import time
import pickle
import sys
import time


def get_all_files(dir):
    filelist = []

    for root, dirs, files in os.walk(dir):
        for file in files:
            if "reject_trials" not in file and "noneeg" not in file:
                filelist.append(os.path.join(root, file))

    return filelist


prelim_ME_db = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
# prelim_MI_db = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]}

reject_ME_db = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}

noneeg_ME_db = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}

pwd = os.getcwd()
processed_data_dir = pwd + "/" + "processed_data"
filelist = get_all_files(processed_data_dir)

print("Processing %d files" % (len(filelist)))

t1 = time.time()
data = None
while len(filelist) > 0:
    seq_v_class_fname = filelist.pop()
    tmp = seq_v_class_fname.split('.')
    reject_trials_fname = tmp[0] + "_reject_trials." + tmp[1]
    noneeg_fname = tmp[0] + "_noneeg." + tmp[1]
    with open(seq_v_class_fname, 'rb') as seq_v_class_file, open(reject_trials_fname, 'rb') as reject_trials_file, open(
            noneeg_fname, 'rb') as noneeg_file:
        seq_v_class_data = pickle.load(seq_v_class_file)
        reject_trials_data = pickle.load(reject_trials_file)
        noneeg_data = pickle.load(noneeg_file)
        for key in seq_v_class_data.keys():
            prelim_ME_db[key].extend(seq_v_class_data[key])
            reject_ME_db[key].extend(reject_trials_data[key])
            noneeg_ME_db[key].extend(noneeg_data[key])

print("Complete loading and merging maps in %f s" % (time.time() - t1))

t1 = time.time()
with open("prelim_ME_db.pickle", "wb") as db, open("reject_ME_db.pickle", 'wb') as rt, open("noneeg_ME_db.pickle",
                                                                                            "wb") as ne:
    i_str = pickle.dumps(prelim_ME_db)
    f_size = sys.getsizeof(i_str) / 1048576
    print("Finished writing %.2f MB of data to prelim_ME_db.pickle in %f s" % (f_size, time.time() - t1))
    db.write(i_str)

    i_str = pickle.dumps(reject_ME_db)
    f_size = sys.getsizeof(i_str) / 1048576
    print("Finished writing %.2f MB of data to reject_ME_db.pickle in %f s" % (f_size, time.time() - t1))
    rt.write(i_str)

    i_str = pickle.dumps(noneeg_ME_db)
    f_size = sys.getsizeof(i_str) / 1048576
    print("Finished writing %.2f MB of data to noneeg_ME_db.pickle in %f s" % (f_size, time.time() - t1))
    ne.write(i_str)

# t1 = time.time()
# f = open("prelim_MI_db.pickle", "wb")
# i_str = pickle.dumps(prelim_MI_db)
# f_size = sys.getsizeof(i_str)/1048576
# f.write(i_str)
# f.close()

# print("Finished writing %.2f MB of data to prelim_MI_db.pickle in %f s" % (f_size, time.time()-t1))
