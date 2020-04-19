from src.architecture import DeepFakeDetector, DeefFakeDetectorTF, cross_entropy_loss, f1_m, acc
from src.video2tfrecordCustom import TfRecordDecoder, Video2TFRecord
import json, math, os
from tensorflow.python.platform import gfile
import horovod.tensorflow.keras as hvd
import tensorflow as tf
from tensorflow.keras import backend as K

if __name__ == "__main__":

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    FRAME_COUNT_PER_EXAMPLE = 80
    BATCH_SIZE = 8
    NUM_EPOCHS = 4

    # Get the dataset from tfrecords
    tfrecord_files = gfile.Glob(os.path.join("../data/records/", "*.tfrecords"))
    num_files = len(tfrecord_files)
    decoder = TfRecordDecoder(FRAME_COUNT_PER_EXAMPLE)
    iterator = tf.data.Iterator.from_structure((tf.float32, tf.int8), (tf.TensorShape([None, FRAME_COUNT_PER_EXAMPLE, 2048]), tf.TensorShape([None, 2])))
    dataset = decoder._make_batch_iterator(tfrecord_files, BATCH_SIZE, NUM_EPOCHS)
    data_initializer_op = iterator.make_initializer(dataset)

    # Initialize the model
    DF = DeefFakeDetectorTF(FRAME_COUNT_PER_EXAMPLE)

    # Setup input and output placeholders
    (input, labels) = iterator.get_next()
    preds = DF.build(input, scope_name="LSTM")

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
    with sess.as_default():
        sess.run(variable_initializer_op)
        sess.run(local_variable_initializer_op)
        for step in range(10):
            sess.run(data_initializer_op)
            while True:
                try:
                    batch_videos, batch_labels = sess.run([input, labels])
                    print(batch_videos.shape, batch_labels.shape)
                    # train_step.run(feed_dict={K.learning_phase(): 1}) #feed_dict={input: train_x, labels: train_y, K.learning_phase(): 1}
                    # [loss_val, acc_op, f1_op] = sess.run(metrics_op, feed_dict={K.learning_phase(): 0})
                    # print("Step: %d, Loss: %f, Accuracy: %f, F1: %f\n" % (step+1, loss_val, acc_op, f1_op))
                except tf.errors.OutOfRangeError:
                    print("Breaking off\n")
                    break
