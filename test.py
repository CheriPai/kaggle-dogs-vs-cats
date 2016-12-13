import numpy as np
import os
import pandas as pd
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from utils import get_gen


test_dir = "data/test"
num_test = 12500
img_size = 150


with open("data/model.json", "r") as json_file:
    json_model = json_file.read()
model = model_from_json(json_model)
model.load_weights("data/model.h5")
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

imgs = np.zeros((num_test, img_size, img_size, 3))
for f in os.listdir(test_dir):
    img_path = os.path.join(test_dir, f)
    img_id = int(img_path.split("/")[-1].split(".")[0])
    img = load_img(img_path, target_size=((img_size,img_size)))
    imgs[img_id-1] = img_to_array(img) / 255

df = pd.DataFrame()
df["id"] = np.arange(1, num_test+1)
df["label"] = model.predict(imgs, batch_size=256)
df.to_csv("data/output", index=False)
