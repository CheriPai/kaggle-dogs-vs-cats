import numpy as np
import os
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from utils import get_gen


test_dir = "data/test"


with open("data/model.json", "r") as json_file:
    json_model = json_file.read()
model = model_from_json(json_model)
model.load_weights("data/model.h5")
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy", "binary_crossentropy"])


imgs = np.zeros((12500, 128, 128, 3))
for i, f in enumerate(os.listdir(test_dir)):
    img_path = os.path.join(test_dir, f)
    img = load_img(img_path, target_size=((128,128)))
    imgs[i] = img_to_array(img)

np.savetxt("data/output", model.predict(imgs, batch_size=256))
