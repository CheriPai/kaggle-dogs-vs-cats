from keras.preprocessing.image import ImageDataGenerator


def get_gen():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_gen = train_datagen.flow_from_directory(
        "data/train",
        target_size=(150, 150),
        batch_size=128,
        class_mode="binary")
    val_gen = test_datagen.flow_from_directory(
        "data/val",
        target_size=(150, 150),
        batch_size=128,
        class_mode="binary")
    return train_gen, val_gen
