# Usage: python3 keratrain.py /mnt/beegfs/vkarri/tfrecords

from src.architecture import DeepFakeDetector, DeefFakeDetectorTF, precision_m, f1_m, recall_m
from src.video2tfrecordCustom import TfRecordDecoder, Video2TFRecord
import json, math, os, sys, time, math, glob
from tensorflow.python.platform import gfile
import tensorflow as tf
from src.tf_metrics import precision, recall, f1
from tensorflow.keras import backend as K
import numpy as np
# import horovod.tensorflow.keras as hvd

class BatchHistory(tf.keras.callbacks.Callback):
    def __init__(self, json_writer):
        super(BatchHistory, self).__init__()
        self.epoch = 0
        self.writer = json_writer

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_batch_end(self, batch, logs={}):
        self.writer.write(json.dumps({'epoch': self.epoch, 'batch': batch, 'loss': str(logs['loss']),
                        'precision': str(logs['precision_m']), 'recall': str(logs['precision_m']), 'f1': str(logs['f1_m']), 'acc': str(logs['acc'])}) + ',')

    def on_train_begin(self, logs):
        self.writer.write("[")

    def on_train_end(self, logs):
        self.writer.write("]")
        self.writer.close()

def getDatasetStatistics(tfrecord_train_files, FRAME_COUNT_PER_EXAMPLE, BATCH_SIZE, split):
    Total = 0
    RecordType = {"Plain": 0, "REAL" : 0, "FAKE": 0}

    for file in tfrecord_train_files:
        list_parts = file.split("_")
        if (len(list_parts) == 2 or list_parts[1] == "REAL"):
            if (list_parts[1] == "REAL"):
                RecordType["REAL"] += 1
            else:
                RecordType["Plain"] += 1
            Total += int(240/FRAME_COUNT_PER_EXAMPLE)
        else:
            RecordType["FAKE"] += 1
            label = list_parts[-1].split(".")[0]
            Total += int(80/FRAME_COUNT_PER_EXAMPLE)

    print("\n----------------------------Batch Statistics for {0} -------------------------------------".format(split))
    print("\n Total Examples found for {0} split: {1}".format(split, Total))
    print("\n Record Frequency => Plain: {0}, REAL Marked: {1}, FAKE Marked: {2}".format(RecordType["Plain"], RecordType["REAL"], RecordType["FAKE"]))
    print("\n Total Example Found: {0}, Steps Per Epoch: {1}".format(Total, math.ceil(Total/BATCH_SIZE)))

    try:
        assert Total == (RecordType["Plain"]+RecordType["REAL"])*int(240/FRAME_COUNT_PER_EXAMPLE) + (RecordType["FAKE"])*int(80/FRAME_COUNT_PER_EXAMPLE)
    except AssertionError as msg:
        print("Total didn't match: your FRAME_COUNT_PER_EXAMPLE should divide 240 & 80 or function implemetation is wrong")

    return math.ceil(Total/BATCH_SIZE)

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
    BATCH_SIZE = 64
    NUM_EPOCHS = 2

    # try:
    # Get the dataset from tfrecords
    #    --------------------------------------------------------------- Train Data --------------------------------------------
    tfrecord_train_files = gfile.Glob(os.path.join(src_path, "*_train.tfrecords"))
    steps_per_epoch_train = getDatasetStatistics(tfrecord_train_files, FRAME_COUNT_PER_EXAMPLE, BATCH_SIZE, 'train')
    decoder_train = TfRecordDecoder(FRAME_COUNT_PER_EXAMPLE)
    train_iterator = tf.data.Iterator.from_structure((tf.float32, tf.int8), (tf.TensorShape([None, FRAME_COUNT_PER_EXAMPLE, 2048]), tf.TensorShape([None, 2])))
    dataset_train = decoder_train._make_batch_iterator_keras(tfrecord_train_files, BATCH_SIZE, NUM_EPOCHS, 512)
    data_initializer_train_op = train_iterator.make_initializer(dataset_train)

    # Setup input and output placeholders
    (input_train, labels_train) = train_iterator.get_next()
    labels_train = tf.dtypes.cast(labels_train, tf.float32) # Labels are originally int32
    # ----------------------------------------------------------------------------------------------------------------------------

    #    --------------------------------------------------------------- Val Data --------------------------------------------
    tfrecord_val_files = gfile.Glob(os.path.join(src_path, "*_val.tfrecords"))
    steps_per_epoch_validation = getDatasetStatistics(tfrecord_val_files, FRAME_COUNT_PER_EXAMPLE, BATCH_SIZE, 'validation')
    decoder_val = TfRecordDecoder(FRAME_COUNT_PER_EXAMPLE)
    val_iterator = tf.data.Iterator.from_structure((tf.float32, tf.int8), (tf.TensorShape([None, FRAME_COUNT_PER_EXAMPLE, 2048]), tf.TensorShape([None, 2])))
    dataset_val = decoder_val._make_batch_iterator_keras(tfrecord_val_files, BATCH_SIZE, NUM_EPOCHS, 512)
    data_initializer_val_op = val_iterator.make_initializer(dataset_val)

    # Setup input and output placeholders
    (input_val, labels_val) = val_iterator.get_next()
    labels_val = tf.dtypes.cast(labels_val, tf.float32) # Labels are originally int32
    # ----------------------------------------------------------------------------------------------------------------------------

    # Initialize the model
    DF = DeefFakeDetectorTF(FRAME_COUNT_PER_EXAMPLE)
    model = DF.build(input_train)
    print(model.summary())
    opt = tf.keras.optimizers.Adam(lr=1e-05)
    met = ['acc', precision_m, f1_m, recall_m] #, recall, f1]
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics = met, target_tensors=[labels_train])

    # Train the Model
    # init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    sess.run([init_l, data_initializer_train_op, data_initializer_val_op])

    # Create Callbacks
    filepath = os.path.join(OUTPUT_PATH, "model_{0}_{1}_weights_".format(FRAME_COUNT_PER_EXAMPLE, BATCH_SIZE) + "{epoch:02d}-{val_loss:.2f}.h5")
    chkpoint = tf.keras.callbacks.ModelCheckpoint(filepath)

    json_log = open(os.path.join(OUTPUT_PATH,'loss_log_{0}_{1}.json'.format(FRAME_COUNT_PER_EXAMPLE, BATCH_SIZE)), mode='wt', buffering=1)
    history = BatchHistory(json_log)

    # try:
    #     history = model.fit(steps_per_epoch=steps_per_epoch_train, epochs=NUM_EPOCHS, verbose=1,
    #               validation_data=(input_val, labels_val), callbacks=[history, chkpoint], class_weight={0.25, 0.75},
    #               validation_steps=steps_per_epoch_validation)
    # finally:
    #     if not(json_log.closed):
    #         json_log.write("]")
    #         json_log.close()

    # except Exception as e:
    #     print(e)
    #     time.sleep(5)
