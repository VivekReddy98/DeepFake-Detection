#!/bin/python3

from src.video2tfrecordCustom import *
import cv2 as cv2
import numpy as np
import math, os, json, time, sys
import tensorflow as tf
import horovod.tensorflow as hvd

#/mnt/beegfs/ppatel27/data/dfdc_train_part_0

#/mnt/beegfs/vkarri/tfrecords

def partitionList(filenames, rank, size):
    numFiles = len(filenames)
    numfilesPRank = int(numFiles/size)
    low = rank*numfilesPRank
    max = rank*numfilesPRank+numfilesPRank

    if(rank == size-1):
        max = max + numFiles%size
    return filenames[low:max]


if __name__ == "__main__":

    hvd.init()

    if len(sys.argv) != 4:
    	print("Usage:", sys.argv[0], "src_path dest_path file_suffix")
    	sys.exit()

    src_path = sys.argv[1]
    dest_path = sys.argv[2]
    file_suffix = sys.argv[3]

    with open(os.path.join(src_path,"metadata.json")) as f:
        data = json.load(f)

    filenames = gfile.Glob(os.path.join(src_path, "*."+ file_suffix))
    filenames =  [name.split(os.path.sep)[-1] for name in filenames]

    myfiles = partitionList(filenames, hvd.rank(), hvd.size())

    V2TF = Video2TFRecord(src_path, dest_path, data, "weights/InceptionV3_Non_Trainable.h5")

    try:
        V2TF.convert_videos_to_tfrecordv2(myfiles)
    except:
        print("Error")
