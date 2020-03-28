import tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.applications import InceptionV3

'''                  To save InceptionV3 weights                          '''
#https://github.com/fchollet/deep-learning-models/blob/master/inception_v3.py
model = InceptionV3(include_top=False, weights='imagenet', pooling='avg')
model.trainable = False
for layer in model.layers:
    layer.trainable = False
model.save('weights/InceptionV3_Non_Trainable.h5')


model.summary()

# model = tensorflow.keras.models.load_model("InceptionV3_Non_Trainable.h5")
# model.summary()
