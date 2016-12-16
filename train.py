from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras import backend as K
from utils import get_gen

nb_epoch = 100

base_model = InceptionV3(weights="imagenet", include_top=False)
for layer in base_model.layers:
    layer.trainable = False

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:]))
model.add(Dense(1024, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

train_gen, val_gen = get_gen()
model.fit_generator(train_gen, nb_epoch=3, verbose=1, samples_per_epoch=20000)

for layer in model.layers:
    layer.trainable = True

model.compile(optimizer=RMSprop(lr=0.001), loss="binary_crossentropy", metrics=["accuracy"])
checkpointer = ModelCheckpoint(filepath="data/weights.hdf5", monitor="val_loss", verbose=1, save_best_only=True, mode="min")
history = model.fit_generator(train_gen, nb_epoch=nb_epoch, verbose=1, samples_per_epoch=20000, validation_data=val_gen, nb_val_samples=5000, callbacks=[checkpointer])

model_json = model.to_json()
with open("data/model.json", "w") as json_file:
    json_file.write(model_json)
