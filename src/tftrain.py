from src.architecture import DeepFakeDetector, DeefFakeDetectorTF, cross_entropy_loss, f1_m, acc
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

    FRAME_COUNT_PER_EXAMPLE = 80
    BATCH_SIZE = 16
    NUM_EPOCHS = 1

    # Get the dataset from tfrecords
    tfrecord_files = gfile.Glob(os.path.join("../data/records/", "*.tfrecords"))
    num_files = len(tfrecord_files)
    print("Num Records Found: {0}".format(num_files))
    decoder = TfRecordDecoder(FRAME_COUNT_PER_EXAMPLE)
    iterator = tf.data.Iterator.from_structure((tf.float32, tf.int8), (tf.TensorShape([None, FRAME_COUNT_PER_EXAMPLE, 2048]), tf.TensorShape([None, 2])))
    dataset = decoder._make_batch_iterator(tfrecord_files, BATCH_SIZE, NUM_EPOCHS)
    data_initializer_op = iterator.make_initializer(dataset)

    # Initialize the model
    DF = DeefFakeDetectorTF(FRAME_COUNT_PER_EXAMPLE)

    # Setup input and output placeholders
    (input, labels) = iterator.get_next()
    preds = DF.buildTF(input, scope_name="LSTM")

    labels = tf.dtypes.cast(labels, tf.float32) # Cast to Float 32
    # Set Up the loss and metrics
    cross_entropy = cross_entropy_loss(labels, preds)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    loss_op = cross_entropy_loss(labels, preds)
    acc_op = acc(labels, preds)
    f1_op = f1_m(labels, preds)

    metrics_op = [loss_op, acc_op, f1_op]

    variable_initializer_op = tf.global_variables_initializer()

    # https://stackoverflow.com/questions/44624648/tensorflow-attempting-to-use-uninitialized-value-in-variable-initialization/44630421
    local_variable_initializer_op = tf.local_variables_initializer()
    # print(local_variable_initializer_op )

    # Training step
    Fake_labels = 0
    True_labels = 0
    with sess.as_default():
        sess.run(variable_initializer_op)
        sess.run(local_variable_initializer_op)
        for step in range(NUM_EPOCHS):
            sess.run(data_initializer_op)
            batch_step = 0
            while True:
                try:
                    batch_videos, batch_labels = sess.run([input, labels])
                    Fake_labels += np.sum(batch_labels[:,0])
                    True_labels += np.sum(batch_labels[:,1])
                    print(batch_step+1, batch_videos.shape, batch_labels.shape, np.sum(batch_videos), np.sum(batch_labels[:,0]), np.sum(batch_labels[:,1]))
                    # train_step.run(feed_dict={K.learning_phase(): 1}) #feed_dict={input: train_x, labels: train_y, K.learning_phase(): 1}
                    # [loss_val, acc_op, f1_op] = sess.run(metrics_op, feed_dict={K.learning_phase(): 0})
                    # print("Epoch: %d, Batch Step: %d, Loss: %f, Accuracy: %f, F1: %f" % (step+1, batch_step, loss_val, acc_op, f1_op))
                except tf.errors.OutOfRangeError:
                    # print("Breaking off\n")
                    break
                batch_step += 1
    print(Fake_labels, True_labels, True_labels+Fake_labels)
