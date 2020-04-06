from src.DataGenerator import MetaData, DataGenerator
from src.architecture import DeepFakeDetector
import json

if __name__ == "__main__":

    FRAME_COUNT_PER_EXAMPLE = 60
    BATCH_SIZE = 8
    NUM_SETS_PER_VIDEO = 4

    inception_path = 'weights/InceptionV3_Non_Trainable.h5'

    with open('data/metadata.json') as f:
        data = json.load(f)

    MD = MetaData("data/train", data, FRAME_COUNT_PER_EXAMPLE, NUM_SETS_PER_VIDEO) #sourcePath, labels, numFrames, numSetsPerVideo
    trainDataGenerator = DataGenerator("data/train", BATCH_SIZE, MD)


    # for i in range(0, 2):
    #     (X, y) = trainDataGenerator[i]
    #     print(X.shape, y.shape)

    DF = DeepFakeDetector(FRAME_COUNT_PER_EXAMPLE)
    DF.build(inception_path, verbose=True)
    DF.compile()
    DF.train(trainDataGenerator)
