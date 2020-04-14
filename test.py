#!/bin/python3

# from src.DataGenerator import MetaData, DataGenerator
from src.architecture import DeepFakeDetector, DeefFakeDetectorTF
from src.videp2tfrecordCustom import TfRecordDecoder, Video2TFRecord
import json, math
import horovod.tensorflow.keras as hvd
import tensorflow as tf
from tensorflow.keras import backend as K

if __name__ == "__main__":

    FRAME_COUNT_PER_EXAMPLE = 60
    BATCH_SIZE = 8
    NUM_SETS_PER_VIDEO = 4

    inception_path = 'weights/InceptionV3_Non_Trainable.h5'

    with open('data/metadata.json') as f:
        data = json.load(f)

    MD = MetaData("data/train", data, FRAME_COUNT_PER_EXAMPLE, NUM_SETS_PER_VIDEO) #sourcePath, labels, numFrames, numSetsPerVideo
    trainDataGenerator = DataGenerator("data/train", BATCH_SIZE, MD)

    DF = DeepFakeDetector(FRAME_COUNT_PER_EXAMPLE)
    DF.build(inception_path, verbose=True)

    numSteps = len(trainDataGenerator)

    hvd.init()

    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.Session(config=config))

    epochs = int(math.ceil(numSteps / hvd.size()))

    opt = tf.keras.optimizers.Adadelta(1.0 * hvd.size())

    opt = hvd.DistributedOptimizer(opt)

    DF.compile(optimizer=opt)

    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    ]

    # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
    if hvd.rank() == 0:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint('./weights/checkpoint-{epoch}.h5'))

    # opt = tf.keras.optimizers.Adadelta(1.0)
    # DF.compile(optimizer=opt)
    # epochs = numSteps
    DF.model.fit_generator(trainDataGenerator, steps_per_epoch=epochs, epochs=1, verbose=1, use_multiprocessing=False, workers=1) #callbacks=callbacks)


    # train(train_data_generator=, val_data_generator=None, steps_per_epoch=epochs, callbacks=callbacks, use_multiprocessing=False)
