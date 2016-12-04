from keras.preprocessing.image import ImageDataGenerator


def preprocess(mode="train"):
    if mode == "train":
        datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True)
        gen = datagen.flow_from_directory(
            "data/train",
            target_size=(128, 128),
            batch_size=128,
            class_mode="binary")
    else:
        datagen = ImageDataGenerator(
            rescale=1./255)
        gen = datagen.flow_from_directory(
            "data/test",
            target_size=(128, 128),
            batch_size=128,
            class_mode="binary")
    return gen
