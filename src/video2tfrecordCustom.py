from src.video2tfrecord import *
from multiprocessing import Process
from tensorflow.python.platform import gfile
from tensorflow.python.platform import flags
from tensorflow.python.platform import app
import cv2 as cv2
import numpy as np
import math, os, json, time
import tensorflow as tf
import multiprocessing
import random

'''
Useful Links:
https://github.com/tomrunia/TF_VideoInputPipeline/blob/master/kinetics/input_pipeline.py
https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564
http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
'''

NUMFRAMES = 80


class TfRecordDecoder:
    def __init__(self, NUMFRAMES):
        self.NUMFRAMES = NUMFRAMES

    # Can Feed this Iterator to Training
    def _make_batch_iterator(self, tfrecord_files, batch_size, num_epochs):
        dataset = tf.data.TFRecordDataset(tfrecord_files, compression_type="GZIP")
        dataset = dataset.map(self.decode_tfrecord)
        dataset = dataset.apply(tf.data.experimental.unbatch())
        dataset = dataset.batch(batch_size)  # Because of TF 1.12, For New Versions, unbatch is a direct method for datset object
        # dataset = dataset.repeat(num_epochs)
        # dataset = dataset.apply(tf.data.experimental.prefetch_to_device())
        # dataset = dataset.shuffle(len(tfrecord_files)*3, seed=random.randint(0, 100))
        # dataset = tf.data.Dataset.zip((dataset))
        # dataset = dataset.prefetch(5)
        return dataset #.make_one_shot_iterator()

    def decode_tfrecord(self, serialized_example):

        parsed_data = tf.parse_single_example(serialized_example, features={
                                                      'vector_size': tf.FixedLenFeature([], tf.int64),
                                                      'batch_size' : tf.FixedLenFeature([], tf.int64),
                                                      'image_raw': tf.FixedLenFeature([], tf.string),
                                                      'labels_raw': tf.FixedLenFeature([], tf.string),
                                                  })

        image = tf.decode_raw(parsed_data['image_raw'], tf.float32)
        annotation = tf.decode_raw(parsed_data['labels_raw'], tf.int64)

        vector_dimensions = tf.cast(parsed_data['vector_size'], tf.int32)
        batch_dim = tf.cast(parsed_data['batch_size'], tf.int32)

        print(tf.shape(annotation), tf.shape(image))
        image = tf.reshape(image, [batch_dim/self.NUMFRAMES, self.NUMFRAMES, vector_dimensions])
        print(tf.shape(annotation), tf.shape(image))

        annotation = tf.reshape(annotation, (1,2))
        annotation = tf.ones([batch_dim/NUMFRAMES, 1], tf.int64) * annotation

        annotation = tf.cast(annotation, dtype=tf.int8)
        # print(tf.shape(annotation), tf.shape(image))

        return image, annotation


class Video2TFRecord:
    def __init__(self, source_path, destination_path, label_dict, inception_path, n_frames_per_training_sample=240,
                file_suffix = "*.mp4", width=299, height=299):
        """
        source_path: directory where video videos are stored
        destination_path: directory where tfrecords should be stored
        file_suffix: defines the video file type, e.g. *.mp4
        width: the width of the videos in pixels (299 Default for InceptionV3)
        height: the height of the videos in pixels (299 Default for InceptionV3)
        """
        # CNN_VECTORIZER = tf.keras.models.load_model(inception_path)
        self.MP = TFRecordGenerator(source_path, destination_path, inception_path, n_frames_per_training_sample,
                                    file_suffix, width, height)
        self.df = label_dict #pd.read_json(json_path).transpose()

    def convert_videos_to_tfrecordv2(self, filenames, split='train'):
        assert split in ['train', 'val', 'test']
        print('Total videos found: ' + str(len(filenames)) + " Split: " + split)
        args_process = []
        count = 0
        for file in filenames:
            # args_process.append((file,self.df[file]["label"],split))
            self.MP.save_video_as_tf_records_ylabels(file,self.df[file]["label"],split)
            count += 1
            if (count > 50):
                break
        # num_cores = multiprocessing.cpu_count()
        # p = multiprocessing.Pool(num_cores)
        # p.starmap(self.MP.save_video_as_tf_records_ylabels, args_process)
        return 1;


class TFRecordGenerator:
    def __init__(self, source_path, destination_path, inception_path, n_frames_per_training_sample,
                file_suffix, width, height):
        self.inception_path = inception_path
        self.N_FRMS_PER_SAMPLE = n_frames_per_training_sample
        self.WIDTH = width
        self.HEIGHT = height
        self.SRC_PATH = source_path
        self.OUT_PATH = destination_path
        self.FILE_SUFF = file_suffix
        self.CNN_VECTORIZER = tf.keras.models.load_model(inception_path)

    def save_video_as_tf_records_ylabels(self, file, label, split):

        start_time = time.time()
        file_path = os.path.join(self.SRC_PATH, file)
        cap = cv2.VideoCapture(file_path)

        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        buf = np.empty((self.N_FRMS_PER_SAMPLE, self.WIDTH, self.HEIGHT, 3), np.dtype('float32'))

        fc = 0
        while (fc < self.N_FRMS_PER_SAMPLE):
            ret, frame = cap.read()
            frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.central_crop(frame, 0.875)
            frame = cv2.resize(frame, (299, 299), interpolation = cv2.INTER_AREA)
            buf[fc] = tf.keras.applications.inception_v3.preprocess_input(frame)
            fc += 1

        buf.astype("float32")
        start_cnn = time.time()
        predictions = self.CNN_VECTORIZER.predict(buf)
        predictions = predictions.astype('float32')

        # print(np.sum(predictions), np.sum(predictions, axis=1))

        # print("--- Model Inference Took: %s seconds ---" % (time.time() - start_cnn))

        tfrecords_filename = os.path.join(self.OUT_PATH, file.split('.')[0] + "_" + split + '.tfrecords')

        # print(tfrecords_filename)

        # Gzip Compression
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        writer = tf.python_io.TFRecordWriter(tfrecords_filename, options=options)

        if label == "FAKE":
            y_label = np.array([1, 0], dtype=np.int64)
        else:
            y_label = np.array([0, 1], dtype=np.int64)


        img_raw = predictions.tostring()
        labels_raw = y_label.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={ 'vector_size': int64_feature(predictions.shape[1]),
                                                                        'batch_size': int64_feature(predictions.shape[0]),
                                                                        'image_raw': bytes_feature(img_raw),
                                                                        'labels_raw': bytes_feature(labels_raw)}))

        writer.write(example.SerializeToString())
        writer.close()
        print("--- Record Generated for file {0}, in {1} seconds --- \n".format(file, str(time.time() - start_time)))
        return 1


    # Borowed from https://github.com/kentsommer/keras-inception-resnetV2/blob/14994dfca36ca09725998a1abf81ea2027dbcb76/evaluate_image.py#L16
    def central_crop(self, image, central_fraction):
        """Crop the central region of the image.
        Remove the outer parts of an image but retain the central region of the image
        along each dimension. If we specify central_fraction = 0.5, this function
        returns the region marked with "X" in the below diagram.
         --------
        |        |
        |  XXXX  |
        |  XXXX  |
        |        | where "X" is the central 50% of the image.
         --------
        Args:
        image: 3-D array of shape [height, width, depth]
        central_fraction: float (0, 1], fraction of size to crop
        Raises:
        ValueError: if central_crop_fraction is not within (0, 1].
        Returns:
        3-D array
        """
        if central_fraction <= 0.0 or central_fraction > 1.0:
            raise ValueError('central_fraction must be within (0, 1]')
        if central_fraction == 1.0:
            return image

        img_shape = image.shape
        depth = img_shape[2]
        fraction_offset = int(1 / ((1 - central_fraction) / 2.0))
        bbox_h_start = np.divide(img_shape[0], fraction_offset)
        bbox_w_start = np.divide(img_shape[1], fraction_offset)

        bbox_h_size = img_shape[0] - bbox_h_start * 2
        bbox_w_size = img_shape[1] - bbox_w_start * 2

        image = image[int(bbox_h_start):int(bbox_h_start+bbox_h_size), int(bbox_w_start):int(bbox_w_start+bbox_w_size), :]
        return image


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

if __name__ == "__main__":

    FRAME_COUNT_PER_EXAMPLE = 80

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    tfrecord_files = gfile.Glob(os.path.join("../data/records/", "*.tfrecords"))
    decoder = TfRecordDecoder(FRAME_COUNT_PER_EXAMPLE)
    iterator = tf.data.Iterator.from_structure((tf.float32, tf.int8), (tf.TensorShape([None, FRAME_COUNT_PER_EXAMPLE, 2048]), tf.TensorShape([None, 2])))
    dataset = decoder._make_batch_iterator(tfrecord_files, 10, 4)

    # iterator = dataset.make_one_shot_iterator()
    data_initializer_op = iterator.make_initializer(dataset)
    next_batch = iterator.get_next()
    sess.run(data_initializer_op)
    for _ in range(0,10):
        batch_videos, batch_labels = sess.run(next_batch)
        print(batch_videos.shape, batch_labels.shape, np.sum(batch_videos), batch_labels)

    '''                                              Create TFRecords
    with open('../data/metadata.json') as f:
        data = json.load(f)

    V2TF = Video2TFRecord("../data/train", "../data/records", data, "../weights/InceptionV3_Non_Trainable.h5")

    filenames = gfile.Glob(os.path.join("../data/train", "*.mp4"))

    filenames =  [name.split(os.path.sep)[-1] for name in filenames]

    V2TF.convert_videos_to_tfrecordv2(filenames)
    '''

    '''                                              Decode Records
    sess = tf.Session()

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer())

    sess.run(init_op)

    tfrecord_files = gfile.Glob(os.path.join("../data/", "*.tfrecords"))
    # sess = tf.Session()
    dataset = tf.data.TFRecordDataset(tfrecord_files, compression_type="GZIP")

    print(dataset)

    decode = TfRecordDecoder(80)

    dataset = dataset.map(decode.decode_tfrecord)
    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()

    batch_videos, batch_labels = sess.run(next_batch)

    print(np.sum(batch_videos),  np.sum(batch_videos, axis=2))

    print(batch_videos.shape, batch_labels.shape)
    '''
