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

def getSplits(filenames, splits=(90,5,5)):
    numFiles = len(filenames)
    train_end = int(numFiles*splits[0]/100)
    val_end = int(numFiles*(splits[0]+splits[1])/100)
    return (filenames[0:train_end], filenames[train_end:val_end], filenames[val_end:])

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

    (train_split, val_split, test_split) = getSplits(myfiles)

    assert len(myfiles) == len(train_split) + len(val_split) + len(test_split)

    try:
        V2TF.convert_videos_to_tfrecordv2(train_split, split='train')
        V2TF.convert_videos_to_tfrecordv2(val_split, split='val')
        V2TF.convert_videos_to_tfrecordv2(test_split, split='test')
    except:
        time.sleep(5)
        print("Error in Conversion")
