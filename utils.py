from keras.preprocessing.image import ImageDataGenerator

img_size = 200

def get_gen():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_gen = train_datagen.flow_from_directory(
        "data/train",
        target_size=(img_size, img_size),
        batch_size=32,
        class_mode="binary")
    val_gen = test_datagen.flow_from_directory(
        "data/val",
        target_size=(img_size, img_size),
        batch_size=32,
        class_mode="binary")
    return train_gen, val_gen
