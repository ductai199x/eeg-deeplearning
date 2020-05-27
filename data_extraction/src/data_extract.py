#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import time
from data_extract_utils import *
from channel_maps import *
import multiprocessing
import os


def get_all_files(dir):
    filelist = []

    for root, dirs, files in os.walk(dir):
        for file in files:
            filelist.append(os.path.join(root, file))

    return filelist


def process_file(fname):
    print("Worker %d is processing file %s\n" % (os.getpid(), fname))
    compressed_fname = fname[len(database_dir):].split('/')
    run_idx = compressed_fname[1].split('.')
    run_idx = run_idx[0].split('_')
    compressed_fname = compressed_fname[0] + "_" + run_idx[-1] + ".pickle"

    pwd = os.getcwd()
    processed_file_dir = pwd + "/processed_data"
    try:
        if not os.path.isdir(processed_file_dir):
            os.mkdir(processed_file_dir)
    except OSError as error:
        print("Cannot create directory. Exiting...")
        print(error)

    compressed_fname = processed_file_dir + "/" + compressed_fname

    t1 = time.time()
    HDR, data = read_data(fname)
    seqs_v_class_map = segregate_data_into_classes(HDR, data)
    # seqs_v_class_map = reject_trials_from_map(seqs_v_class_map)
    CLM = channel_loc_map()
    seqs_v_class_map = data_1D_to_2D(seqs_v_class_map, 9, 9, CLM)
    compress_segregated_data(seqs_v_class_map, compressed_fname)
    print("Worker %d is done processing file in %f s\n" %
          (os.getpid(), time.time() - t1))


database_dir = "/home/sweet/1-workdir/eeg001-2017/"
filelist = get_all_files(database_dir)
MAX_NPROCESS = multiprocessing.cpu_count()//2

if __name__ == "__main__":
    print("There are a total of %d files in %s\n" % (len(filelist), database_dir))
    print("**********START PROCESSING ALL FILES**********")
    print("MAX NUMBER OF PROCESS = %d" % (MAX_NPROCESS))

    p = multiprocessing.Pool(MAX_NPROCESS)
    p.map(process_file, filelist)
