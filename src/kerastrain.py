

# Usage: python3 keratrain.py /mnt/beegfs/vkarri/tfrecords


from src.architecture import DeepFakeDetector, DeefFakeDetectorTF, precision_m, f1_m, recall_m
from src.video2tfrecordCustom import TfRecordDecoder, Video2TFRecord
import json, math, os, sys
from tensorflow.python.platform import gfile
# import horovod.tensorflow.keras as hvd
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


if __name__ == "__main__":

    if len(sys.argv) != 2:
    	print("Usage:", sys.argv[0], "src_path")
    	sys.exit()

    src_path = sys.argv[1]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    FRAME_COUNT_PER_EXAMPLE = 80
    BATCH_SIZE = 128
    NUM_EPOCHS = 1

    try:
        # Get the dataset from tfrecords
        #    --------------------------------------------------------------- Train Data --------------------------------------------
        tfrecord_train_files = gfile.Glob(os.path.join(src_path, "*_train.tfrecords"))
        num_files_train = len(tfrecord_train_files)
        # print("Num Records Found: {0}".format(num_files))
        decoder_train = TfRecordDecoder(FRAME_COUNT_PER_EXAMPLE)
        train_iterator = tf.data.Iterator.from_structure((tf.float32, tf.int8), (tf.TensorShape([None, FRAME_COUNT_PER_EXAMPLE, 2048]), tf.TensorShape([None, 2])))
        dataset_train = decoder_train._make_batch_iterator_keras(tfrecord_train_files, BATCH_SIZE, NUM_EPOCHS)
        data_initializer_train_op = train_iterator.make_initializer(dataset_train)

        # Setup input and output placeholders
        (input_train, labels_train) = train_iterator.get_next()
        labels_train = tf.dtypes.cast(labels_train, tf.float32) # Labels are originally int32
        # ----------------------------------------------------------------------------------------------------------------------------

        #    --------------------------------------------------------------- Val Data --------------------------------------------
        tfrecord_val_files = gfile.Glob(os.path.join(src_path, "*_val.tfrecords"))
        num_files_val = len(tfrecord_val_files)
        # print("Num Records Found: {0}".format(num_files))
        decoder_val = TfRecordDecoder(FRAME_COUNT_PER_EXAMPLE)
        val_iterator = tf.data.Iterator.from_structure((tf.float32, tf.int8), (tf.TensorShape([None, FRAME_COUNT_PER_EXAMPLE, 2048]), tf.TensorShape([None, 2])))
        dataset_val = decoder_val._make_batch_iterator_keras(tfrecord_val_files, BATCH_SIZE, NUM_EPOCHS)
        data_initializer_val_op = val_iterator.make_initializer(dataset_val)

        # Setup input and output placeholders
        (input_val, labels_val) = val_iterator.get_next()
        labels_val = tf.dtypes.cast(labels_val, tf.float32) # Labels are originally int32
        # ----------------------------------------------------------------------------------------------------------------------------

        # Initialize the model
        DF = DeefFakeDetectorTF(FRAME_COUNT_PER_EXAMPLE)
        model = DF.build(input_train)
        opt = tf.keras.optimizers.Adam(lr=1e-05)
        met = [tf.keras.metrics.binary_accuracy]
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics = met, target_tensors=[labels_train])

        # Train the Model
        sess.run(data_initializer_train_op)
        sess.run(data_initializer_val_op)
        model.fit(steps_per_epoch=int(num_files_train*(240/FRAME_COUNT_PER_EXAMPLE)/BATCH_SIZE), epochs=NUM_EPOCHS, verbose=1, validation_data=(input_val, labels_val),
                  validation_steps=int(num_files_val*(240/FRAME_COUNT_PER_EXAMPLE)/BATCH_SIZE))

    except Exception as e:
        print(e + "Wating for 5 Seconds")
        time.sleep(5)
