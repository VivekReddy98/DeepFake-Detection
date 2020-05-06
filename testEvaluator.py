from src.architecture import DeepFakeDetector, DeefFakeDetectorTF
from src.video2tfrecordCustom import TfRecordDecoder
import json, math, os, sys, time, math, glob
from tensorflow.python.platform import gfile
import tensorflow as tf
from src.trainingUtils import EvaluateCallback, getDatasetStatistics, getDatasetStatisticsTest
from tensorflow.keras import backend as K
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

if __name__ == "__main__":

    if len(sys.argv) != 3:
    	print("Usage:", sys.argv[0], "file_path model_path")
    	sys.exit()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    src_path = sys.argv[1]
    model_path = sys.argv[2]

    print(model_path.split(os.path.sep)[-1])

    FRAME_COUNT_PER_EXAMPLE = 20

    BATCH_SIZE_TEST = 512
    # print(FRAME_COUNT_PER_EXAMPLE)

    DF = DeefFakeDetectorTF(FRAME_COUNT_PER_EXAMPLE)
    model = DF.buildnew(name="INP_PLACEHOLDER")
    model.load_weights(model_path)


    # ----------------------------------------  Extract Test Files  -------------------------------------------------------------------------------------------
    tfrecord_test_files = gfile.Glob(os.path.join(src_path, "*.tfrecords"))
    steps_per_epoch_test, clas_weights_test = getDatasetStatisticsTest(tfrecord_test_files, FRAME_COUNT_PER_EXAMPLE, BATCH_SIZE_TEST, 'test')
    decoder_test = TfRecordDecoder(FRAME_COUNT_PER_EXAMPLE)
    test_iterator = tf.data.Iterator.from_structure((tf.float32, tf.int8), (tf.TensorShape([None, FRAME_COUNT_PER_EXAMPLE, 2048]), tf.TensorShape([None, 2])))
    dataset_test = decoder_test._make_batch_iterator_keras(tfrecord_test_files, BATCH_SIZE_TEST, 1, 512)
    data_initializer_test_op = test_iterator.make_initializer(dataset_test)

    # Setup input and output placeholders
    (input_tensors_test, label_tensors_test) = test_iterator.get_next()
    label_tensors_test = tf.dtypes.cast(label_tensors_test, tf.float32) # Labels are originally int32

    # Initialize the data op
    sess.run([data_initializer_test_op])

    f1_dict_eval = {"TP":0, "FP":0, "FN":0, "TN":0}
    loss_list = []
    acc_list = []
    for _ in range(0, steps_per_epoch_test):
        try:
            batch_videos, batch_labels = sess.run([input_tensors_test, label_tensors_test])
            batch_predict = model.predict(batch_videos)
            loss_list.append(log_loss(batch_labels, batch_predict))

            batch_predict = np.round(batch_predict)
            batch_labels = batch_labels.astype(int)
            batch_predict = batch_predict.astype(int)

            #[loss, acc, _, _, _] = model.evaluate(batch_videos, batch_labels, verbose=0)
            tn, fp, fn, tp = confusion_matrix(batch_labels[:,0], batch_predict[:,0]).ravel()

            f1_dict_eval["TP"] += tp
            f1_dict_eval["FN"] += fn
            f1_dict_eval["FP"] += fp
            f1_dict_eval["TN"] += tn

            # print(tn+fp+fn+tp, f1_dict_eval["TP"]+f1_dict_eval["FN"]+f1_dict_eval["FP"]+f1_dict_eval["TN"])

            acc_list.append((tn+tp)/(fp+fn+tn+tp))

        except tf.errors.OutOfRangeError:
            print("Range Exceeded")
            break

    precision_F = f1_dict_eval["TP"] / (f1_dict_eval["TP"] + f1_dict_eval["FP"])
    recall_F = f1_dict_eval["TP"] / (f1_dict_eval["TP"] + f1_dict_eval["FN"])
    f1_F = (2*precision_F*recall_F) / (precision_F+recall_F)

    precision_R = f1_dict_eval["TN"] / (f1_dict_eval["TN"] + f1_dict_eval["FN"])
    recall_R = f1_dict_eval["TN"] / (f1_dict_eval["TN"] + f1_dict_eval["FP"])
    f1_R = (2*precision_R*recall_R) / (precision_R+recall_R)
    accuracy = np.float32(sum(acc_list) / len(acc_list))
    loss = np.float32(sum(loss_list) / len(loss_list))

    Total = f1_dict_eval["TP"].item() + f1_dict_eval["TN"].item() + f1_dict_eval["FP"].item() + f1_dict_eval["FN"].item()
    print(" Test Metrics: Accuracy: {0:.4f}, Loss: {1:.4f}".format(accuracy.item(), loss.item()))
    print("               Class FAKE: Precision : {0:.4f}, Recall: {1:.4f}, F1: {2:.4f}".format(precision_F.item(), recall_F.item(), f1_F.item()))
    print("               Class REAL: Precision : {0:.4f}, Recall: {1:.4f}, F1: {2:.4f}".format(precision_R.item(), recall_R.item(), f1_R.item()))
    print("               TN: {0}, TP: {1}, FP: {2} FN: {3}".format(f1_dict_eval["TP"].item()/Total, f1_dict_eval["TN"].item()/Total,
                                                                                    f1_dict_eval["FP"].item()/Total, f1_dict_eval["FN"].item()/Total))
