from keras.preprocessing.image import ImageDataGenerator


def get_gen():
    datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True)
    train_gen = datagen.flow_from_directory(
        "data/train",
        target_size=(128, 128),
        batch_size=128,
        class_mode="binary")
    val_gen = datagen.flow_from_directory(
        "data/val",
        target_size=(128, 128),
        batch_size=128,
        class_mode="binary")
    return train_gen, val_gen
