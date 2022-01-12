import os
import json
from tensorflow.keras.models import load_model
import pandas as pd
import pickle
import numpy as np
import shutil

from tensorflow.keras.preprocessing import image                  
from tqdm.notebook import tqdm
from PIL import ImageFile                            

BASE_MODEL_PATH = "./model"
PICKLE_DIR = "./pickle_files"
JSON_DIR = "./json_files"

info = {
    "c0": "SAFE_DRIVING",
    "c1": "TEXTING_RIGHT",
    "c2": "TALKING_PHONE_RIGHT",
    "c3": "TEXTING_LEFT",
    "c4": "TALKING_PHONE_LEFT",
    "c5": "OPERATING_RADIO",
    "c6": "DRINKING",
    "c7": "REACHING_BEHIND",
    "c8": "HAIR_AND_MAKEUP",
    "c9": "TALKING_TO_PASSENGER"
}
BEST_MODEL = os.path.join(BASE_MODEL_PATH,"distracted-17-1.00.hdf5")
model = load_model(BEST_MODEL)

with open(os.path.join(PICKLE_DIR,"labels_list.pkl"),"rb") as handle:
    labels_id = pickle.load(handle)
print(labels_id)


def predict_img(imageTest):
    img = image.load_img(imageTest, target_size=(128, 128))
    x = image.img_to_array(img)
    imageProcessed = np.expand_dims(x, axis=0)
    ImageFile.LOAD_TRUNCATED_IMAGES = True  
    test_tensors = imageProcessed.astype('float32')/255 - 0.5
    ypred_test = model.predict(test_tensors,verbose=1)
    ypred_class = np.argmax(ypred_test,axis=1)
    print(ypred_class)



    id_labels = dict()
    for class_name,idx in labels_id.items():
        id_labels[idx] = class_name
        
    data_test = id_labels[ypred_class[0]]

    return info[data_test]