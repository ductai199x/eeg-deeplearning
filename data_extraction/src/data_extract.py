import numpy
from scipy import signal
import biosig
import time
import pickle
import bz2
import sys

# map event type to event label
# class 1: 0x600 = 1536 (elbow flexion)
# class 2: 0x601 = 1537 (elbow extension)
# class 3: 0x602 = 1538 (supination)
# class 4: 0x603 = 1539 (pronation)
# class 5: 0x604 = 1540 (hand close)
# class 6: 0x605 = 1541 (hand open)
# class 7: 0x606 = 1542 (rest)

# also, there are event types that signify start of trials, computer beep, cross on screen
# 0x300 = 768 (start of trial)
# 0x311 = 785 (beep)
# 0x312 = 786 (cross on screen)
# numbers that are OR-ed with 0x8000 signify end of event.
event_map = {
    1536: 1,
    1537: 2,
    1538: 3,
    1539: 4,
    1540: 5,
    1541: 6,
    1542: 7
}


def read_data(f_name):
    t1 = time.time()
    HDR = biosig.sopen(f_name, 'r')
    # print("Read header of %s in %f s\n" % (f_name, time.time()-t1))

    t1 = time.time()
    data = biosig.sread(HDR, HDR.NRec, 0)
    # print("Read data of %s in %f s\n" % (f_name, time.time()-t1))

    biosig.sclose(HDR)

    return HDR, data


def segregate_data_into_classes(HDR, data):
    t1 = time.time()

    seqs_v_class_map = {}

    for key in event_map:
        seqs_v_class_map[event_map[key]] = []

    event_hit = 0
    start_frame = 0
    end_frame = 0
    for i in range(len(HDR.EVENT.TYP)):
        code = HDR.EVENT.TYP[i]
        if event_hit == 0 and code in event_map:
            event_hit = code
            start_frame = HDR.EVENT.POS[i]

        if code == event_hit + 32768:
            end_frame = HDR.EVENT.POS[i]
            seqs_v_class_map[event_map[event_hit]].append(
                signal_processing(data[start_frame:end_frame+1, 0:64]))
            event_hit = 0

    # print("Finished segregating data into classes in %f s\n" % (time.time()-t1))

    return seqs_v_class_map


def signal_processing(data):
    downsampled = signal.decimate(data, 2, 30, axis=0)

    return downsampled


# compress segregated data into zip file
def compress_segregated_data(data, f_name):
    t1 = time.time()
    i_str = pickle.dumps(data)
    compressed_data = bz2.compress(i_str)
    f_size = sys.getsizeof(compressed_data)/1048576
    with bz2.open(f_name, "wb") as f:
        f.write(compressed_data)

    # print("Finished writing %.2f MB of data to %s in %f s\n" %
    #       (f_size, f_name, time.time()-t1))
