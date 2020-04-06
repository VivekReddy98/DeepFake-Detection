import tensorflow
import tensorflow.keras as keras
from tensorflow.python.platform import gfile
import numpy as np
import pandas as pd
import json
import os, cv2, random

class MetaData:
    def __init__(self, sourcePath, labels, numFrames, numSetsPerVideo, fileSuffix="*.mp4"):
        self.sourcePath = sourcePath
        self.labels = labels
        self.numVideos = 0
        self.totalFrames = 0   # Total Multiples of numFrames present
        self.train = []
        self.val = []
        self.numFrames = numFrames
        self.numSetsPerVideo = numSetsPerVideo
        filenames = gfile.Glob(os.path.join(self.sourcePath, fileSuffix))
        self.__extract_metainfo(filenames)

    def __extract_metainfo(self, filenames):
        for file in filenames:
            if os.path.sep in file:
                file = file.split(os.path.sep)[-1]
            cap = cv2.VideoCapture(os.path.join(self.sourcePath, file))
            frameCount = 0
            if hasattr(cv2, 'cv'):
              frameCount = int(cap.get(cv2.cv.CAP_PROP_FRAME_COUNT))
            else:
              frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            # print(file + " : " + str(frameCount))
            numfrm = int(frameCount/self.numFrames)
            if numfrm == 0:
                print("Sample " + file + " is skipped from traning because of low frame count\n")
            else:
                self.numVideos += 1
                if numfrm > 4:
                    numfrm = 4
                self.labels[file]["frame"] = numfrm
                self.train.append(file)
        return None

class DataGenerator(keras.utils.Sequence):
    def __init__(self, sourcePath, batchSize, metadata, dim=(299,299,3), fileSuffix="*.mp4"):
        self.dim = dim
        self.sourcePath = sourcePath
        self.file_suffix = fileSuffix
        self.batchSize = batchSize
        self.labels = metadata.labels
        self.totalFrames = metadata.totalFrames
        self.numVideos = metadata.numVideos
        self.train = metadata.train
        self.numFrames = metadata.numFrames
        self.numSetsPerVideo = metadata.numSetsPerVideo
        self.numBathcesPerEpoch = int(np.floor(self.numVideos*self.numSetsPerVideo)/self.batchSize)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.numBathcesPerEpoch

    def __getitem__(self, index):
        'Generate one batch of data'
        numfilestoprocess = int(self.batchSize/self.numSetsPerVideo)
        ind_min = index*numfilestoprocess
        ind_max = (index+1)*numfilestoprocess

        if (ind_max > len(self.train)):
            ind_max = len(self.train)

        outArr = np.empty(shape = (self.batchSize, self.numFrames, self.dim[0], self.dim[1], self.dim[2]), dtype = np.float32)
        ylabels = []

        batchPtr = 0

        for ind in range(ind_min, ind_max):
            cap = cv2.VideoCapture(os.path.join(self.sourcePath, self.train[ind]))

            # print(batchPtr, os.path.join(self.sourcePath, self.train[ind]))

            framePtr = 0
            setCount = 0
            while (1):
                ret, image = cap.read()
                # print(type(image), framePtr, batchPtr, setCount)
                image = image.astype("float32")
                image -= image.mean()
                image /= image.std()

                if (framePtr < self.numFrames and batchPtr < self.batchSize):
                    outArr[batchPtr, framePtr, :, :, :] = cv2.resize(image, (self.dim[0], self.dim[1]), interpolation = cv2.INTER_AREA)
                else:
                    raise Exception("batchPtr and framePtr are out of range, Error in calculating indexes")

                framePtr += 1
                if (framePtr >= self.numFrames):
                    if (self.labels[self.train[ind]]["label"] == "FAKE"):
                        ylabels.append([1,0])
                    else:
                        ylabels.append([0,1])
                    framePtr = 0
                    batchPtr += 1
                    setCount += 1

                if (setCount >= self.numSetsPerVideo):
                    break

            cap.release()

        ylabels_np  = np.asarray(ylabels, dtype=np.float32)
        # print(ylabels., (ind_max-ind_min)*self.numSetsPerVideo)
        assert (ylabels_np.shape[0] == (ind_max-ind_min)*self.numSetsPerVideo)

        print("Batch Extraction with index " + str(index) + " completed \n")
        return (outArr, ylabels_np)

    def on_epoch_end(self):
        random.shufle(self.train)
