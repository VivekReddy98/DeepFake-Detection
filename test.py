# import tensorflow
# from tensorflow.keras import backend as K
# from tensorflow.keras.applications import InceptionV3


# df = pd.read_json("data/metadata.json").transpose()
#
#
# model.summary()

# model = tensorflow.keras.models.load_model("InceptionV3_Non_Trainable.h5")
# model.summary()

from src.video2tfrecordCustom import Video2TFRecord

V2TF = Video2TFRecord("data/train", "data/train")
V2TF.convert_videos_to_tfrecordv2()
