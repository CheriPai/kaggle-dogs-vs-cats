from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K
from utils import get_gen

nb_epoch = 100

base_model = VGG16(weights="imagenet", include_top=False)
for layer in base_model.layers:
    layer.trainable = False

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:]))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer=Adam(lr=0.001), loss="binary_crossentropy", metrics=["accuracy"])

train_gen, val_gen = get_gen()

earlystopper = EarlyStopping(monitor="val_loss", patience=20, mode="min")
checkpointer = ModelCheckpoint(filepath="data/weights.hdf5", monitor="val_loss", verbose=1, save_best_only=True, mode="min")
history = model.fit_generator(train_gen, nb_epoch=nb_epoch, verbose=1, samples_per_epoch=train_gen.nb_sample, validation_data=val_gen, nb_val_samples=val_gen.nb_sample, callbacks=[checkpointer, earlystopper])

model_json = model.to_json()
with open("data/model.json", "w") as json_file:
    json_file.write(model_json)
