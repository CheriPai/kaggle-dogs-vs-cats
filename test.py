import numpy as np
import os
import pandas as pd
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from utils import get_gen


test_dir = "data/test"


with open("data/model.json", "r") as json_file:
    json_model = json_file.read()
model = model_from_json(json_model)
model.load_weights("data/model.h5")
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

imgs = np.zeros((12500, 150, 150, 3))
for i, f in enumerate(os.listdir(test_dir)):
    img_path = os.path.join(test_dir, f)
    img = load_img(img_path, target_size=((150,150)))
    imgs[i] = img_to_array(img) / 255

predictions = model.predict(imgs, batch_size=256)
