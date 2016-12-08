from keras.models import model_from_json
from utils import get_gen

with open("data/model.json", "r") as json_file:
    json_model = json_file.read()

model = model_from_json(json_model)
model.load_weights("data/model.h5")
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy", "binary_crossentropy"])

test_gen = get_gen(mode="test")

model.evaluate_generator(test_gen, val_samples=12500)
