from src.video2tfrecord import *
from multiprocessing import Process
from tensorflow.python.platform import gfile
from tensorflow.python.platform import flags
from tensorflow.python.platform import app
import cv2 as cv2
import numpy as np
import math
import os
import tensorflow as tf
import time

'''
Useful Links:
https://github.com/tomrunia/TF_VideoInputPipeline/blob/master/kinetics/input_pipeline.py
https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564
'''

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def give_indices(numframes, n_frames_per_training_sample):
    n_records_by_video = int(numframes/n_frames_per_training_sample)
    remainder = int((numframes%n_frames_per_training_sample)/2)
    for i in range(0, n_records_by_video):
        ind_min = remainder + i*n_frames_per_training_sample
        ind_max = ind_min + n_frames_per_training_sample
        if (i < n_records_by_video-1):
            split = "train"
        else:
            split = "val"
        yield (ind_min, ind_max, split)
    yield None

def _float32_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

class Video2TFRecord:
    def __init__(self, source_path, destination_path, label_dict, n_frames_per_training_sample=80,
                file_suffix = "*.mp4", width=299, height=299):
        """
        source_path: directory where video videos are stored
        destination_path: directory where tfrecords should be stored
        n_frames_per_training_sample: HyperParameter one of [20, 40, 80]
        file_suffix: defines the video file type, e.g. *.mp4
        width: the width of the videos in pixels (299 Default for InceptionV3)
        height: the height of the videos in pixels (299 Default for InceptionV3)
        """
        assert n_frames_per_training_sample in [20, 40, 80]

        self.source_path = source_path
        self.destination_path = destination_path
        self.n_frames_per_training_sample = n_frames_per_training_sample
        self.width = width
        self.height = height
        self.file_suffix = file_suffix
        self.df = label_dict #pd.read_json(json_path).transpose()

    def convert_videos_to_tfrecordv2(self):
        filenames = gfile.Glob(os.path.join(self.source_path, self.file_suffix))
        print('Total videos found: ' + str(len(filenames)))
        # print(filenames)
        self.__save_data_to_tfrecords(filenames)


    def __save_data_to_tfrecords(self, filenames):

        jobPool = []

        for i, file in enumerate(filenames):

            name = file.split("/")[-1]

            if self.df[name]["label"] == "FAKE":
                y_label = 1;
            else:
                y_label = 0;

            p = Process(target=save_video_as_tf_records, args=(file, self.destination_path, self.width, self.height,
                                                               self.n_frames_per_training_sample, y_label))
            p.start()
            jobPool.append(p)

            break

        for process in jobPool:
            process.join()


def save_video_as_tf_records(file, out_path, width, height, n_frames_per_training_sample, label):
    cap = cv2.VideoCapture(file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    image = np.empty((frameHeight, frameWidth, 3), np.dtype('float32'))

    buf = np.empty((frameCount, width, height, 3), np.dtype('float32'))

    fc = 0
    ret = True

    while (fc < frameCount  and ret):
        ret, image = cap.read()
        image = image.astype("float32")
        image -= image.mean()
        image /= image.std()
        buf[fc] = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
        fc += 1

    buf.astype("float32")
    itr = give_indices(frameCount, n_frames_per_training_sample)
    record_count = 0
    writer = None
    feature = {}

    tup = next(itr)

    while(itr != None):
        filename = file.split("/")[-1].split(".")[0]+"_"+str(record_count)+"_"+tup[2]+".tfrecords"
        print('Writing', filename)
        feature["image"] = bytes_feature(tf.compat.as_bytes(buf[tup[0]:tup[1],:,:,:].tobytes()))
        feature["label"] = int64_feature(label)
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer = tf.python_io.TFRecordWriter(out_path+filename)
        writer.write(example.SerializeToString())
        writer.close()
        tup = next(itr)
        record_count += 1

    return 1
