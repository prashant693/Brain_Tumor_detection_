from __future__ import division, print_function
from keras.applications.resnet import preprocess_input as preprocess_input_0
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import preprocess_input as preprocess_input_2
from keras.applications.vgg19 import preprocess_input as preprocess_input_1
import os
from PIL import Image  
import shutil
from tensorflow import keras

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)


MODEL_PATH_0 = "D:/Downloads/model_00.h5"
MODEL_PATH_1 = "D:/Downloads/model_01.h5"
MODEL_PATH_2 = "D:/Downloads/model_02.h5"


model_0 = keras.models.load_model(MODEL_PATH_0)
model_1 = keras.models.load_model(MODEL_PATH_1)
model_2 = keras.models.load_model(MODEL_PATH_2)
       

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    shutil.rmtree('C:/Users/dell/Desktop/abc')
    os.mkdir('C:/Users/dell/Desktop/abc')
    os.mkdir('C:/Users/dell/Desktop/abc/yes')
    picture = Image.open(img_path)  
    picture = picture.save("C:/Users/dell/Desktop/abc/yes/1.jpg")
    test_datagen_0 = ImageDataGenerator( preprocessing_function=preprocess_input_0)
    test_generator_0 = test_datagen_0.flow_from_directory("C:/Users/dell/Desktop/abc",target_size=(224,224),batch_size=32,shuffle=False,class_mode='binary')
    test_datagen_1 = ImageDataGenerator( preprocessing_function=preprocess_input_1)
    test_generator_1 = test_datagen_0.flow_from_directory("C:/Users/dell/Desktop/abc",target_size=(224,224),batch_size=32,shuffle=False,class_mode='binary')
    test_datagen_2 = ImageDataGenerator( preprocessing_function=preprocess_input_2)
    test_generator_2 = test_datagen_0.flow_from_directory("C:/Users/dell/Desktop/abc",target_size=(224,224),batch_size=32,shuffle=False,class_mode='binary')
    prediction_0 = model_0.predict(test_generator_0)
    pre_0 = prediction_0[0][0]
    prediction_1 = model_1.predict(test_generator_1)
    pre_1 = prediction_1[0][0]
    prediction_2 = model_2.predict(test_generator_2)
    pre_2 = prediction_2[0][0]
    if(pre_0<0.80):
        pree_0 = 0
    else:
        pree_0 = 1
    if(pre_1<0.80):
        pree_1 = 0
    else:
        pree_1 = 1
    if(pre_2<0.80):
        pree_2 = 0
    else:
        pree_2 = 1        
    result = pree_0 + pree_1 + pree_2
    if(result<2):
        return 'No'
    else:
        return 'Yes'


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        file_path = "C:/Users/dell/Demo/1.jpg"
        f.save(file_path)
        preds = model_predict(file_path, model_0)
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)
