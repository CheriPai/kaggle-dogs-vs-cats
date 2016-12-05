from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from utils import get_gen

nb_epoch = 100

base_model = InceptionV3(weights="imagenet", include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

model = Model(input=base_model.input, output=predictions)
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy", "binary_crossentropy"])

train_gen, val_gen = get_gen()
model.fit_generator(train_gen, nb_epoch=nb_epoch, verbose=1, samples_per_epoch=20000, validation_data=val_gen, nb_val_samples=5000)
model.save("data/model.h5")
