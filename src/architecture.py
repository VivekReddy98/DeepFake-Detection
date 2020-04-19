import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import TimeDistributed, Dense, LSTM, Dropout
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.backend import categorical_crossentropy



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
    (val, _) = tf.metrics.accuracy(y_true, y_pred)
    return val


class DeepFakeDetector:
    def __init__(self, frames):
        '''
        Frames: According to the Paper the Frame size is set to 20, 40 or 80.
        Issue Regarding BatchSize and Frames: https://github.com/keras-team/keras/issues/8364
        Very Important: While Evaluating or Predicting use the batch size as specified in the Training Phase
        '''
        self.FRAMES = frames
        self.model = Sequential()

    def compile(self, loss='categorical_crossentropy', optimizer=None, metrics=[]):
        if optimizer==None:
            opt = tf.keras.optimizers.Adam(lr=1e-05)
        else:
            opt = optimizer

        if len(metrics) == 0:
            met = ['acc',f1_m, precision_m, recall_m]
        else:
            met = metrics
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics = met )

    def train(self, train_data_generator, steps_per_epoch, val_data_generator = None, callbacks = [], use_multiprocessing=False):
        self.model.fit_generator(generator = train_data_generator, steps_per_epoch=steps_per_epoch, verbose=1, callbacks = callbacks,
                                 use_multiprocessing=use_multiprocessing, validation_data=val_data_generator)

    def predict(self):
        pass

    def build(self, inception_path, verbose=False):
        '''
        --------------------------------------------------------------------------
        Takes a frame of size 299*299*3 and returns a vector of shape 1*2048
        Pre-trained InceptionV3 is used as the CNN BackBone
        ---------------------------------------------------------------------------
                             To save InceptionV3 weights
        #https://github.com/fchollet/deep-learning-models/blob/master/inception_v3.py
        model = InceptionV3(include_top=False, weights='imagenet', pooling='avg')
        model.trainable = False
        for layer in model.layers:
            layer.trainable = False
        model.save(inception_path)
        ---------------------------------------------------------------------------
                             Trainable Model Architecture

        Layer (type)                 Output Shape              Param #
        =================================================================
        time_distributed (TimeDistri (None, 80, 2048)          21802784
        _________________________________________________________________
        lstm (LSTM)                  (None, 2048)              33562624
        _________________________________________________________________
        dense (Dense)                (None, 512)               1049088
        _________________________________________________________________
        dropout (Dropout)            (None, 512)               0
        _________________________________________________________________
        dense_1 (Dense)              (None, 1)                 513
        =================================================================
        Total params: 56,415,009
        Trainable params: 34,612,225
        Non-trainable params: 21,802,784 (Inception V3 Parameters)
        _________________________________________________________________

        '''

        CNN = tf.keras.models.load_model(inception_path)

        '''
        Distribute a batch of frames in time and process them sequentially in LSTM Layer followed by Dense Layers.
        '''
        # input_model = Input(shape=(self.FRAMES, 299, 299, 3), name='video_input')
        self.model.add(TimeDistributed(CNN, input_shape=(self.FRAMES, 299, 299, 3)))
        self.model.add(LSTM(units=512, dropout=0.5))
        self.model.add(Dense(units=256, activation='relu'))
        self.model.add(Dropout(rate=0.5))
        self.model.add(Dense(2, activation='softmax'))

        if verbose:
            # CNN.summary()
            self.model.summary()

        return None


'''
TF Compatible Model Defintion
'''
class DeefFakeDetectorTF:
    def __init__(self, frames):
        self.FRAMES = frames

    def build(self, input, scope_name="LSTM"):
        '''
        Run this under a session
        '''
        with tf.variable_scope(scope_name):
            x = LSTM(units=512, input_shape=(None, self.FRAMES, None), dropout=0.5)(input)
            x = Dense(units=256, activation='relu')(x)
            x = Dropout(rate=0.5)(x)
            preds = Dense(2, activation='softmax')(x)

        return preds





    # inception_path = '../weights/InceptionV3_Non_Trainable.h5'
    # DF = DeepFakeDetector(80)
    # model = DF.build(inception_path, verbose=True)



    '''
    FRAME_COUNT_PER_EXAMPLE = 60
    BATCH_SIZE = 8
    NUM_SETS_PER_VIDEO = 4

    inception_path = 'weights/InceptionV3_Non_Trainable.h5'

    with open('data/metadata.json') as f:
        data = json.load(f)

    MD = MetaData("data/train", data, FRAME_COUNT_PER_EXAMPLE, NUM_SETS_PER_VIDEO) #sourcePath, labels, numFrames, numSetsPerVideo
    trainDataGenerator = DataGenerator("data/train", BATCH_SIZE, MD)

    DF = DeepFakeDetector(FRAME_COUNT_PER_EXAMPLE)
    # opt = tf.keras.optimizers.Adadelta(1.0)
    # DF.compile(optimizer=opt)
    # epochs = numSteps
    DF.model.fit_generator(trainDataGenerator, steps_per_epoch=epochs, epochs=1, verbose=1, use_multiprocessing=False, workers=1) #callbacks=callbacks)
    '''


if __name__ == "__main__":
    DF = DeefFakeDetectorTF(80)
    DF.build()
