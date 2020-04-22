from src.architecture import DeepFakeDetector, DeefFakeDetectorTF, precision_m, f1_m, recall_m
from src.video2tfrecordCustom import TfRecordDecoder, Video2TFRecord
import json, math, os
from tensorflow.python.platform import gfile
# import horovod.tensorflow.keras as hvd
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

if __name__ == "__main__":

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    FRAME_COUNT_PER_EXAMPLE = 80
    BATCH_SIZE = 16
    NUM_EPOCHS = 5

    # Get the dataset from tfrecords
    tfrecord_files = gfile.Glob(os.path.join("../data/records/", "*.tfrecords"))
    num_files = len(tfrecord_files)
    # print("Num Records Found: {0}".format(num_files))
    decoder = TfRecordDecoder(FRAME_COUNT_PER_EXAMPLE)
    iterator = tf.data.Iterator.from_structure((tf.float32, tf.int8), (tf.TensorShape([None, FRAME_COUNT_PER_EXAMPLE, 2048]), tf.TensorShape([None, 2])))
    dataset = decoder._make_batch_iterator_keras(tfrecord_files, BATCH_SIZE, NUM_EPOCHS)
    data_initializer_op = iterator.make_initializer(dataset)

    # Setup input and output placeholders
    (input, labels) = iterator.get_next()
    labels = tf.dtypes.cast(labels, tf.float32) # Labels are originally int32

    # Initialize the model
    DF = DeefFakeDetectorTF(FRAME_COUNT_PER_EXAMPLE)
    model = DF.build(input)
    opt = tf.keras.optimizers.Adam(lr=1e-05)
    met = ['acc', f1_m, precision_m, recall_m]

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics = met, target_tensors=[labels])

    sess.run(data_initializer_op)
    model.fit(steps_per_epoch=int(num_files*(240/FRAME_COUNT_PER_EXAMPLE)/BATCH_SIZE), epochs=NUM_EPOCHS, verbose=1)
