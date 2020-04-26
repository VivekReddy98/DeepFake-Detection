import json, math, os, sys, time
from tensorflow.python.platform import gfile
# import horovod.tensorflow.keras as hvd
import tensorflow as tf
from tensorflow.keras import backend as K
from src.architecture import DeepFakeDetector, DeefFakeDetectorTF, cross_entropy_loss, f1_m, acc
from src.video2tfrecordCustom import TfRecordDecoder, Video2TFRecord
from src.tf_metrics import precision, recall, f1
import numpy as np
import functools
from pathlib import Path

def model_fn(features, labels, mode, params):
    # Define the inference graph
    graph_outputs = some_tensorflow_applied_to(features)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Extract the predictions
        predictions = some_dict_from(graph_outputs)
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        # Compute loss, metrics, tensorboard summaries
        loss = compute_loss_from(graph_outputs, labels)
        metrics = compute_metrics_from(graph_outputs, labels)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            # Get train operator
            train_op = compute_train_op_from(graph_outputs, labels)
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)

        else:
            raise NotImplementedError('Unknown mode {}'.format(mode))



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
    BATCH_SIZE = 16
    NUM_EPOCHS = 1

    # Initialize the model
    DF = DeefFakeDetectorTF(FRAME_COUNT_PER_EXAMPLE)
    estimator_model = DF.build()

    # Compile the model
    estimator_model.compile(optimizer=tf.train.AdamOptimizer(1e-4), loss='categorical_crossentropy',
                    metrics=[precision, recall, f1, 'acc'])

    # Generate the Estimator
    estimator = tf.keras.estimator.model_to_estimator(keras_model=estimator_model, model_dir="/home/vkarri/DeepFake_Detection/tfmeta")


    #    --------------------------------------------------------------- Train Data --------------------------------------------
    tfrecord_train_files = gfile.Glob(os.path.join(src_path, "*_train.tfrecords"))
    num_files_train = len(tfrecord_train_files)
    decoder_train = TfRecordDecoder(FRAME_COUNT_PER_EXAMPLE)
    # train_iterator = tf.data.Iterator.from_structure((tf.float32, tf.int8), (tf.TensorShape([None, FRAME_COUNT_PER_EXAMPLE, 2048]), tf.TensorShape([None, 2])))
    # dataset_train = decoder_train._make_batch_iterator_keras(tfrecord_train_files, BATCH_SIZE, NUM_EPOCHS)
    # data_initializer_op = train_iterator.make_initializer(dataset_train)
    # print(dataset_train)
    # ----------------------------------------------------------------------------------------------------------------------------

    #    --------------------------------------------------------------- Val Data --------------------------------------------
    tfrecord_val_files = gfile.Glob(os.path.join(src_path, "*_val.tfrecords"))
    num_files_val = len(tfrecord_val_files)
    decoder_val = TfRecordDecoder(FRAME_COUNT_PER_EXAMPLE)
    # dataset_val = decoder_val._make_batch_iterator_keras(tfrecord_val_files, BATCH_SIZE, 1)
    # print(dataset_val)
    # ----------------------------------------------------------------------------------------------------------------------------


    train_inpf = functools.partial(decoder_train._make_batch_iterator_keras, tfrecord_train_files, BATCH_SIZE, NUM_EPOCHS)
    eval_inpf = functools.partial(decoder_val._make_batch_iterator_keras, tfrecord_train_files, BATCH_SIZE, NUM_EPOCHS)

    # 2. Create a hook
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    # hook = tf.contrib.estimator.stop_if_no_increase_hook(
        # estimator, 'f1', 500, min_steps=8000, run_every_secs=120)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=None)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # estimator.train(input_fn = train_inpf)
    # estimator.evaluate(input_fn = lambda: decoder_val._make_batch_iterator_keras(tfrecord_train_files, BATCH_SIZE, 1))
