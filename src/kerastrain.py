# Usage: python3 keratrain.py /mnt/beegfs/vkarri/tfrecords

from src.architecture import DeepFakeDetector, DeefFakeDetectorTF
from src.video2tfrecordCustom import TfRecordDecoder
import json, math, os, sys, time, math, glob
from tensorflow.python.platform import gfile
import tensorflow as tf
from trainingUtils import EvaluateCallback, getDatasetStatistics, TP_m, FP_m, FN_m, TN_m
from tensorflow.keras import backend as K
import numpy as np
# import horovod.tensorflow.keras as hvd
from tensorflow.keras.models import load_model


if __name__ == "__main__":

    OUTPUT_PATH = "weights"

    if len(sys.argv) != 2:
    	print("Usage:", sys.argv[0], "src_path")
    	sys.exit()

    src_path = sys.argv[1]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    FRAME_COUNT_PER_EXAMPLE = 80
    BATCH_SIZE_TRAIN = 64
    BATCH_SIZE_VAL = 512
    NUM_EPOCHS = 10


    #                                                            Get the dataset object from tfrecords object
    #    --------------------------------------------------------------- Train Data --------------------------------------------
    tfrecord_train_files = gfile.Glob(os.path.join(src_path, "*_val.tfrecords"))
    steps_per_epoch_train, class_weights = getDatasetStatistics(tfrecord_train_files, FRAME_COUNT_PER_EXAMPLE, BATCH_SIZE_TRAIN, 'train')
    decoder_train = TfRecordDecoder(FRAME_COUNT_PER_EXAMPLE)
    train_iterator = tf.data.Iterator.from_structure((tf.float32, tf.int8), (tf.TensorShape([None, FRAME_COUNT_PER_EXAMPLE, 2048]), tf.TensorShape([None, 2])))
    dataset_train = decoder_train._make_batch_iterator_keras(tfrecord_train_files, BATCH_SIZE_TRAIN, NUM_EPOCHS, 512)
    data_initializer_train_op = train_iterator.make_initializer(dataset_train)

    # Setup input and output placeholders
    (input_train, labels_train) = train_iterator.get_next()
    labels_train = tf.dtypes.cast(labels_train, tf.float32) # Labels are originally int32
    # ----------------------------------------------------------------------------------------------------------------------------

    #    --------------------------------------------------------------- Val Data --------------------------------------------
    tfrecord_val_files = gfile.Glob(os.path.join(src_path, "*_test.tfrecords"))
    steps_per_epoch_validation, clas_weights_val = getDatasetStatistics(tfrecord_val_files, FRAME_COUNT_PER_EXAMPLE, BATCH_SIZE_VAL, 'validation')
    decoder_val = TfRecordDecoder(FRAME_COUNT_PER_EXAMPLE)
    val_iterator = tf.data.Iterator.from_structure((tf.float32, tf.int8), (tf.TensorShape([None, FRAME_COUNT_PER_EXAMPLE, 2048]), tf.TensorShape([None, 2])))
    dataset_val = decoder_val._make_batch_iterator_keras(tfrecord_val_files, BATCH_SIZE_VAL, NUM_EPOCHS, 512)
    data_initializer_val_op = val_iterator.make_initializer(dataset_val)

    # Setup input and output placeholders
    (input_val, labels_val) = val_iterator.get_next()
    labels_val = tf.dtypes.cast(labels_val, tf.float32) # Labels are originally int32
    # ----------------------------------------------------------------------------------------------------------------------------

    # tfrecord_test_files = gfile.Glob(os.path.join(src_path, "*_test.tfrecords"))
    # steps_per_epoch_test = getDatasetStatistics(tfrecord_test_files, FRAME_COUNT_PER_EXAMPLE, BATCH_SIZE, 'test')

    # Initialize the model
    DF = DeefFakeDetectorTF(FRAME_COUNT_PER_EXAMPLE)
    model = DF.build(name="INP_PLACEHOLDER")
    print(model.summary())
    opt = tf.keras.optimizers.Adam(lr=1e-05)
    met = ['acc'] #, recall, f1] TP_m, FP_m, FN_m, TN_m
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics = met, target_tensors=[labels_train])

    # Train the Model
    # init_g = tf.global_variables_initializer()
    # init_l = tf.local_variables_initializer()
    sess.run([data_initializer_train_op, data_initializer_val_op])

    # Create Callbacks
    #filepath = os.path.join(OUTPUT_PATH, "model_{0}_{1}_weights_".format(FRAME_COUNT_PER_EXAMPLE, BATCH_SIZE) + "{epoch:02d}-{val_loss:.2f}.h5")
    #chkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='acc', save_weights_only=True)

    json_log = open(os.path.join(OUTPUT_PATH,'loss_log_{0}_{1}.json'.format(FRAME_COUNT_PER_EXAMPLE, BATCH_SIZE_TRAIN)), mode='wt', buffering=1)
    EvaluateCB = EvaluateCallback(json_log, input_val, labels_val, steps_per_epoch_validation,
                                  OUTPUT_PATH, FRAME_COUNT_PER_EXAMPLE, BATCH_SIZE_VAL, sess)

    try:
        history = model.fit(x = {"INP_PLACEHOLDER" : input_train}, steps_per_epoch=steps_per_epoch_train, epochs=NUM_EPOCHS, verbose=1,
                            callbacks=[EvaluateCB], class_weight=class_weights)
    except Exception as e:
        print(e)
        time.sleep(5)
