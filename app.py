""""
Created on 29-03-2021

@author: Sharath Chandra B
"""

from __future__ import division, print_function
import os
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Defining the flask app
app = Flask(__name__)

# Load the model
MODEL_PATH = "inception.h5"
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path,target_size=(224, 224))

    # processing the image
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = "This is Bacterial_spot disease"
    elif preds == 1:
        preds = "This is Early_blight disease"
    elif preds == 2:
        preds = "This is healthy leaf"
    elif preds == 3:
        preds = "This is Late_blight disease"
    elif preds == 4:
        preds = "This is Leaf_Mold disease"
    elif preds == 5:
        preds = "This is Septoria_leaf_spot disease"
    elif preds == 6:
        preds = "This is Spider_mites Two-spotted_spider_mite disease"
    elif preds == 7:
        preds = "This is Target_Spot disease"
    elif preds == 8:
        preds = "This is Tomato_mosaic_virus disease"
    elif preds == 9:
        preds = "This is Tomato_Yellow_Leaf_Curl_Virus disease"

    return preds

@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        # save file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None

if __name__ == '__main__':
    app.run(port = 5001, debug = True)