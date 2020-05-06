import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.platform import gfile
import json, math, os, sys, time, math, glob
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

class DS:
    def __init__(self):
        self.metrics = {}
        self.metrics['P_FK'] = []
        self.metrics['R_FK'] = []
        self.metrics['f1_FK'] = []
        self.metrics['P_RL'] = []
        self.metrics['R_RL'] = []
        self.metrics['f1_RL'] = []
        self.metrics['acc'] = []
        self.metrics['loss'] = []

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
                sess,
                optimizer):

        super(EvaluateCallback, self).__init__()
        self.epoch = 0
        self.input_tensors_val = input_tensors_val
        self.label_tensors_val = label_tensors_val
        self.writer = json_writer
        self.f1_dict = {}
        # self.evaluatecallback = ValBatchHistory()
        self.steps = steps_per_epoch_val
        self.ds = DS()

        self.filepath = filepath
        self.NUMFRMS = FRAME_COUNT_PER_EXAMPLE
        self.BTH_SIZE = BATCH_SIZE
        self.sess = sess
        self.optimizer = optimizer

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.f1_dict = {"TP":0, "FP":0, "FN":0, "TN":0}
        self.f1_dict_eval = {"TP":0, "FP":0, "FN":0, "TN":0}

    def on_batch_begin(self, batch, logs={}):
        self.writer.write(',')

    def on_batch_end(self, batch, logs={}):
        self.writer.write(json.dumps({'epoch': self.epoch, 'batch': batch, 'loss': str(logs['loss'])}))

    def on_epoch_end(self, epoch, logs=None):
        loss_list = []
        acc_list = []
        for _ in range(0, self.steps):
            try:
                batch_videos, batch_labels = self.sess.run([self.input_tensors_val, self.label_tensors_val])
                batch_predict = self.model.predict(batch_videos)
                loss_list.append(log_loss(batch_labels,batch_predict))

                batch_predict = np.round(batch_predict)
                batch_labels = batch_labels.astype(int)
                batch_predict = batch_predict.astype(int)

                #[loss, acc, _, _, _] = self.model.evaluate(batch_videos, batch_labels, verbose=0)
                tn, fp, fn, tp = confusion_matrix(batch_labels[:,0], batch_predict[:,0]).ravel()
                self.f1_dict_eval["TP"] += tp
                self.f1_dict_eval["FN"] += fn
                self.f1_dict_eval["FP"] += fp
                self.f1_dict_eval["TN"] += tn

                acc_list.append((tn+tp)/(fp+fn+tn+tp))

            except tf.errors.OutOfRangeError:
                print("Range Exceeded")
                break

        precision_F = self.f1_dict_eval["TP"] / (self.f1_dict_eval["TP"] + self.f1_dict_eval["FP"])
        recall_F = self.f1_dict_eval["TP"] / (self.f1_dict_eval["TP"] + self.f1_dict_eval["FN"])
        f1_F = (2*precision_F*recall_F) / (precision_F+recall_F)

        precision_R = self.f1_dict_eval["TN"] / (self.f1_dict_eval["TN"] + self.f1_dict_eval["FN"])
        recall_R = self.f1_dict_eval["TN"] / (self.f1_dict_eval["TN"] + self.f1_dict_eval["FP"])
        f1_R = (2*precision_R*recall_R) / (precision_R+recall_R)

        accuracy = np.float32(sum(acc_list) / len(acc_list))
        loss = np.float32(sum(loss_list) / len(loss_list))

        Total = self.f1_dict_eval["TP"].item() + self.f1_dict_eval["TN"].item() + self.f1_dict_eval["FP"].item() + self.f1_dict_eval["FN"].item()
        print("\nEpoch " +  str(epoch) + " Validation Metrics: Accuracy: {0:.4f}, Loss: {1:.4f}".format(accuracy.item(), loss.item()))
        print("                     Class FAKE: Precision : {0:.4f}, Recall: {1:.4f}, F1: {2:.4f}".format(precision_F.item(), recall_F.item(), f1_F.item()))
        print("                     Class REAL: Precision : {0:.4f}, Recall: {1:.4f}, F1: {2:.4f}".format(precision_R.item(), recall_R.item(), f1_R.item()))
        print("               TN: {0}, TP: {1}, FP: {2} FN: {3}".format(self.f1_dict_eval["TP"].item()/Total, self.f1_dict_eval["TN"].item()/Total,
                                                                                        self.f1_dict_eval["FP"].item()/Total, self.f1_dict_eval["FN"].item()/Total))
        if epoch == 0:
            self.model.save_weights(os.path.join(self.filepath, "model_{0}_{1}_{2}_weights.h5".format(self.NUMFRMS, self.BTH_SIZE, self.optimizer)))
        else:
            print( max(self.ds.metrics['f1_FK']), max(self.ds.metrics['f1_RL']), f1_F, f1_R)
            if f1_R >= max(self.ds.metrics['f1_RL']):
                self.model.save_weights(os.path.join(self.filepath, "model_{0}_{1}_{2}_weights.h5".format(self.NUMFRMS, self.BTH_SIZE), self.optimizer))
            else:
                print("F1 Score not improved and thus not saving model")

        self.ds.metrics['P_FK'].append(precision_F.item())
        self.ds.metrics['R_FK'].append(recall_F.item())
        self.ds.metrics['f1_FK'].append(f1_F.item())
        self.ds.metrics['P_RL'].append(precision_R.item())
        self.ds.metrics['R_RL'].append(recall_R.item())
        self.ds.metrics['f1_RL'].append(f1_R.item())
        self.ds.metrics['acc'].append(accuracy.item())
        self.ds.metrics['loss'].append(loss.item())

    def on_train_begin(self, logs):
        self.writer.write("[")
        print(self.ds.metrics)

    def on_train_end(self, logs):
        self.writer.write("]")
        self.writer.close()
        with open(os.path.join(self.filepath,"model_{0}_{1}_{2}_valhist.json".format(self.NUMFRMS, self.BTH_SIZE, self.optimizer)), 'w') as f:
            json.dump(self.ds.metrics, f)

    def on_train_end_interrupt(self):
        with open(os.path.join(self.filepath,"model_{0}_{1}_{2}_valhist.json".format(self.NUMFRMS, self.BTH_SIZE, self.optimizer)), 'w') as f:
            json.dump(self.ds.metrics, f)

# Get Datset Statistics
def getDatasetStatistics(tfrecord_train_files, FRAME_COUNT_PER_EXAMPLE, BATCH_SIZE, split):
    Total = 0
    RecordType = {"Plain": 0, "REAL" : 0, "FAKE": 0}

    for file in tfrecord_train_files:
        filename = file.split(os.path.sep)[-1]
        list_parts = filename.split("_")
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

    print("\n----------Batch Statistics for {0} -------------".format(split))
    print("Frequency of TF Records => REAL Marked: {0}, FAKE Marked: {1}".format(RecordType["REAL"], RecordType["FAKE"]))
    print("Total Examples Extracted: {0}".format(Total))
    print("Steps Per Epoch: {0} Batch Size: {1}".format(math.floor(Total/BATCH_SIZE), BATCH_SIZE))

    try:
        assert Total == (RecordType["Plain"]+RecordType["REAL"])*int(240/FRAME_COUNT_PER_EXAMPLE) + (RecordType["FAKE"])*int(80/FRAME_COUNT_PER_EXAMPLE)
    except AssertionError as msg:
        print("Total didn't match: your FRAME_COUNT_PER_EXAMPLE should divide 240 & 80 or function implemetation is wrong")

    # weights = {(RecordType["FAKE"])*int(80/FRAME_COUNT_PER_EXAMPLE)/Total, (RecordType["Plain"]+RecordType["REAL"])*int(240/FRAME_COUNT_PER_EXAMPLE)/Total}

    weights = {0.40, 0.60}

    # print(weights)

    return math.floor(Total/BATCH_SIZE), weights

# Get Datset Statistics
def getDatasetStatisticsTest(tfrecord_train_files, FRAME_COUNT_PER_EXAMPLE, BATCH_SIZE, split):
    Total = 0
    RecordType = {"Plain": 0, "REAL" : 0, "FAKE": 0}

    for file in tfrecord_train_files:
        filename = file.split(os.path.sep)[-1]
        list_parts = filename.split("_")
        if (list_parts[1] == "REAL"):
            RecordType["REAL"] += 1
            Total += int(240/FRAME_COUNT_PER_EXAMPLE)
        else:
            RecordType["FAKE"] += 1
            Total += int(240/FRAME_COUNT_PER_EXAMPLE)

    print("\n----------Batch Statistics for {0} -------------".format(split))
    print("Frequency of TF Records => REAL Marked: {0}, FAKE Marked: {1}".format(RecordType["REAL"], RecordType["FAKE"]))
    print("Total Examples Extracted: {0}".format(Total))
    print("Steps Per Epoch: {0} Batch Size: {1}".format(math.floor(Total/BATCH_SIZE), BATCH_SIZE))

    try:
        assert Total == (RecordType["Plain"]+RecordType["REAL"])*int(240/FRAME_COUNT_PER_EXAMPLE) + (RecordType["FAKE"])*int(240/FRAME_COUNT_PER_EXAMPLE)
    except AssertionError as msg:
        print("Total didn't match: your FRAME_COUNT_PER_EXAMPLE should divide 240 & 80 or function implemetation is wrong")

    # weights = {(RecordType["FAKE"])*int(80/FRAME_COUNT_PER_EXAMPLE)/Total, (RecordType["Plain"]+RecordType["REAL"])*int(240/FRAME_COUNT_PER_EXAMPLE)/Total}

    weights = {0.40, 0.60}

    # print(weights)

    return math.ceil(Total/BATCH_SIZE), weights

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

def TN_m(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    true_negatives = K.sum(y_neg * y_pred_neg)
    return true_negatives

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

if __name__ == "__main__":
    ds = DS()
    attrs = vars(ds)
    print(attrs)
