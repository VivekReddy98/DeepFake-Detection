import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.platform import gfile
import json, math, os, sys, time, math, glob
import numpy as np
from sklearn.metrics import confusion_matrix
class DS:
    def __init__(self):
        precision = np.float32(0.)
        recall = np.float32(0.)
        accuracy = np.float32(0.)
        f1 = np.float32(0.)
        loss = np.float32(0.)

class ValBatchHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super(ValBatchHistory, self).__init__()
        self.ds = DS()
        self.f1_dict = {}

    def on_batch_begin(self, batch, logs={}):
        self.f1_dict = {"TP":0, "FP":0, "FN":0}

    def on_batch_end(self, batch, logs={}):
        self.f1_dict["TP"] += logs['TP_m']
        self.f1_dict["FP"] += logs['FP_m']
        self.f1_dict["FN"] += logs['FN_m']

    def on_train_end(self, logs):
        ds.precision = self.f1_dict["TP"] / (self.f1_dict["TP"] + self.f1_dict["FP"])
        ds.recall = self.f1_dict["TP"] / (self.f1_dict["TP"] + self.f1_dict["FN"])
        ds.accuracy = logs['acc']
        ds.loss = logs['loss']
        ds.f1 = (2*ds.precision*ds.recall) / (ds.recall+ds.precision)


class EvaluateCallback(tf.keras.callbacks.Callback):
    def __init__(self, json_writer, input_tensors_val,
                label_tensors_val,
                steps_per_epoch_val,
                filepath,
                FRAME_COUNT_PER_EXAMPLE,
                BATCH_SIZE,
                sess):

        super(EvaluateCallback, self).__init__()
        self.epoch = 0
        self.input_tensors_val = input_tensors_val
        self.label_tensors_val = label_tensors_val
        self.writer = json_writer
        self.f1_dict = {}
        # self.evaluatecallback = ValBatchHistory()
        self.steps = steps_per_epoch_val

        self.filepath = filepath
        self.NUMFRMS = FRAME_COUNT_PER_EXAMPLE
        self.BTH_SIZE = BATCH_SIZE

        self.sess = sess

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.f1_dict = {"TP":0, "FP":0, "FN":0}
        self.f1_dict_eval = {"TP":0, "FP":0, "FN":0}

    def on_batch_begin(self, batch, logs={}):
        self.writer.write(',')

    def on_batch_end(self, batch, logs={}):
        self.writer.write(json.dumps({'epoch': self.epoch, 'batch': batch, 'loss': str(logs['loss']),
                        'TP': str(logs['TP_m']), 'FP': str(logs['FP_m']), 'FN': str(logs['FN_m']), 'acc': str(logs['acc'])}))
        self.f1_dict["TP"] += logs['TP_m']
        self.f1_dict["FP"] += logs['FP_m']
        self.f1_dict["FN"] += logs['FN_m']
        self.__printf1()

    def on_epoch_end(self, epoch, logs=None):
        loss_list = []
        acc_list = []
        for _ in range(0, self.steps):
            try:
                batch_videos, batch_labels = self.sess.run([self.input_tensors_val, self.label_tensors_val])
                batch_predict = self.model.predict(batch_videos)
                batch_predict = np.round(batch_predict)

                # print(batch_predict.shape, batch_labels.shape, batch_labels.dtype, batch_predict.dtype)

                batch_labels = batch_labels.astype(int)
                batch_predict = batch_predict.astype(int)

                # print(batch_predict.shape, batch_labels.shape, batch_labels.dtype, batch_predict.dtype)
                # print(batch_labels[:,0].shape, batch_videos[:,0].shape, batch_labels[:,0].dtype, batch_predict[:,0].dtype)

                #[loss, acc, _, _, _] = self.model.evaluate(batch_videos, batch_labels, verbose=0)
                tn, fp, fn, tp = confusion_matrix(batch_labels[:,0], batch_predict[:,0]).ravel()
                self.f1_dict_eval["TP"] += tp
                self.f1_dict_eval["FN"] += fn
                self.f1_dict_eval["FP"] += fp
                #loss_list.append(loss)
                #acc_list.append(acc)

            except tf.errors.OutOfRangeError:
                break

        self.ds.precision = self.f1_dict_eval["TP"] / (self.f1_dict_eval["TP"] + self.f1_dict_eval["FP"])
        self.ds.recall = self.f1_dict_eval["TP"] / (self.f1_dict_eval["TP"] + self.f1_dict_eval["FN"])
        #ds.accuracy = np.float32(sum(acc_list) / len(acc_list))
        #ds.loss = np.float32(sum(loss_list) / len(loss_list))
        self.ds.f1 = (2*ds.precision*ds.recall) / (ds.recall+ds.precision)

        print("Validation Metrics: Precision: {0:.4f}, Recall: {1:.4f}, F1: {2:.4f}, Accuracy: {3:.4f}, Loss: {4:.4f}".format(self.ds.precision.item(),
                                                                                                                              self.ds.recall.item(),
                                                                                                                              self.ds.f1.item(),
                                                                                                                              self.ds.accuracy.item(),
                                                                                                                              self.ds.loss.item()))

        self.model.save_weights(os.path.join(self.filepath, "model_{0}_{1}_weights_{2}".format(self.NUMFRMS, self.BTH_SIZE, epoch)))

    def __printf1(self):
        precision = self.f1_dict["TP"] / (self.f1_dict["TP"] + self.f1_dict["FP"])
        recall = self.f1_dict["TP"] / (self.f1_dict["TP"] + self.f1_dict["FN"])
        f1 = (2*precision*recall)/(precision+recall)
        print(" Precision: {0:.4f} - Recall: {1:.4f} - F1: {2:.4f} ".format(precision.item(), recall.item(), f1.item()))

    def on_train_begin(self, logs):
        self.writer.write("[")

    def on_train_end(self, logs):
        self.writer.write("]")
        self.writer.close()


# Get Datset Statistics
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
    print("\n Total Example Found: {0}, Steps Per Epoch: {1}".format(Total, math.floor(Total/BATCH_SIZE)))

    try:
        assert Total == (RecordType["Plain"]+RecordType["REAL"])*int(240/FRAME_COUNT_PER_EXAMPLE) + (RecordType["FAKE"])*int(80/FRAME_COUNT_PER_EXAMPLE)
    except AssertionError as msg:
        print("Total didn't match: your FRAME_COUNT_PER_EXAMPLE should divide 240 & 80 or function implemetation is wrong")

    return math.floor(Total/BATCH_SIZE)



# Metrics and Loss Function
def TP_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    return true_positives

def FP_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    FP = predicted_positives - true_positives
    return FP

def FN_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    FN = possible_positives - true_positives
    return FN

def cross_entropy_loss(y_true, y_pred):
    return tf.reduce_mean(categorical_crossentropy(y_true, y_pred))

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def acc(y_true, y_pred):
    (val, op) = tf.metrics.accuracy(y_true, y_pred)
    return op
