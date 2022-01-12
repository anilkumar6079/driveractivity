import os
import json
from keras.models import load_model
import pandas as pd
import pickle
import numpy as np
import shutil

from keras.preprocessing import image                  
from tqdm.notebook import tqdm
from PIL import ImageFile                            

BASE_MODEL_PATH = "./model"
PICKLE_DIR = "./pickle_files"
JSON_DIR = "./json_files"


BEST_MODEL = os.path.join(BASE_MODEL_PATH,"distracted-17-1.00.hdf5")
model = load_model(BEST_MODEL)

with open(os.path.join(PICKLE_DIR,"labels_list.pkl"),"rb") as handle:
    labels_id = pickle.load(handle)
print(labels_id)

imageTest = "img_555.jpg"
img = image.load_img(imageTest, target_size=(128, 128))
x = image.img_to_array(img)
imageProcessed = np.expand_dims(x, axis=0)
ImageFile.LOAD_TRUNCATED_IMAGES = True  
test_tensors = imageProcessed.astype('float32')/255 - 0.5
ypred_class = np.argmax(ypred_test,axis=1)
print(ypred_class)



id_labels = dict()
for class_name,idx in labels_id.items():
    id_labels[idx] = class_name
    
data_test = id_labels[ypred_class[0]]

with open(os.path.join(JSON_DIR,'class_name_map.json')) as secret_input:
    info = json.load(secret_input)

info[data_test]