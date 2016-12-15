from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from utils import get_gen

nb_epoch = 5

# base_model = InceptionV3(weights="imagenet", include_top=False)
base_model = ResNet50(weights="imagenet", include_top=False)


model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:]))
model.add(Dense(1024, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

checkpointer = ModelCheckpoint(filepath="data/weights.hdf5", monitor="val_loss", verbose=1, save_best_only=True, mode="min")

train_gen, val_gen = get_gen()
history = model.fit_generator(train_gen, nb_epoch=nb_epoch, verbose=1, samples_per_epoch=20000, validation_data=val_gen, nb_val_samples=5000, callbacks=[checkpointer])

model_json = model.to_json()
with open("data/model.json", "w") as json_file:
    json_file.write(model_json)
