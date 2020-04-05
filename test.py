from src.DataGenerator import MetaData, DataGenerator
from src.architecture import DeepFakeDetector
import json

if __name__ == "__main__":

    BATCH_SIZE = 60

    inception_path = 'weights/InceptionV3_Non_Trainable.h5'

    with open('data/metadata.json') as f:
        data = json.load(f)

    MD = MetaData("data/train", data, 60, 4)
    trainDataGenerator = DataGenerator("data/train", 4, MD)

    DF = DeepFakeDetector(BATCH_SIZE)
    DF.build(inception_path, verbose=True)
    DF.compile()
    DF.train(trainDataGenerator)
