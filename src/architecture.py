import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import TimeDistributed, Dense, LSTM, Dropout
from tensorflow.keras.applications import InceptionV3

class DeepFakeDetector:
    def __init__(self, frames):
        '''
        Frames: According to the Paper the Frame size is set to 20, 40 or 80.
        Issue Regarding BatchSize and Frames: https://github.com/keras-team/keras/issues/8364
        Very Important: While Evaluating or Predicting use the batch size as specified in the Training Phase
        '''
        self.FRAMES = frames
        self.model = Sequential()

    def compile(self):
        opt = tf.keras.optimizers.Adam(lr=1e-05)
        self.model.compile(optimizer=opt, loss='sparse_categorical_crossentropy')

    def train(self, train_data_generator, val_data_generator = None):
        self.model.fit_generator(generator = train_data_generator, verbose=1,
                                 use_multiprocessing=True, validation_data=val_data_generator)

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
        self.model.add(LSTM(units=2048, dropout=0.5))
        self.model.add(Dense(units=512, activation='relu'))
        self.model.add(Dropout(rate=0.5))
        self.model.add(Dense(1, activation='sigmoid'))

        if verbose:
            # CNN.summary()
            self.model.summary()

        return None



if __name__ == "__main__":

    inception_path = '../weights/InceptionV3_Non_Trainable.h5'
    DF = DeepFakeDetector(80)
    model = DF.build(inception_path, verbose=True)
