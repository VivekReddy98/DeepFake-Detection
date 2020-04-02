
from src.video2tfrecord import *

# from tensorflow.python.platform import gfile
# from tensorflow.python.platform import flags
# from tensorflow.python.platform import app
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

class Video2TFRecord:
    def __init__(self, source_path, destination_path, n_frames_per_training_sample=80,
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

    def convert_videos_to_tfrecordv2(self):
        filenames = gfile.Glob(os.path.join(self.source_path, self.file_suffix))
        print('Total videos found: ' + str(len(filenames)))
        self.__save_data_to_tfrecords(filenames)

    def __save_data_to_tfrecords(self, filenames):
        n_records_by_video = 0 # This depends of int(num_frames_per_video/n_frames_per_training_sample)

        for i, file in enumerate(filenames):

            data = convert_video_to_numpy(filenames=[file], width=self.width, height=self.height,
                                          n_frames_per_video='all',
                                          n_channels=3,
                                          dense_optical_flow=False)

            data = data[-1]

            n_frames = data[0]

            # n_records_by_video = int(n_frames/)

            break;

            # if n_videos_in_record > len(filenames):
            #   total_batch_number = 1
            # else:
            #   total_batch_number = int(math.ceil(len(filenames) / n_videos_in_record))
            # print('Batch ' + str(i + 1) + '/' + str(total_batch_number) + " completed")
            # assert data.size != 0, 'something went wrong during video to numpy conversion'
            # save_numpy_to_tfrecords(data, destination_path, 'batch_',
            #                         n_videos_in_record, i + 1, total_batch_number,
            #                         color_depth=color_depth)
