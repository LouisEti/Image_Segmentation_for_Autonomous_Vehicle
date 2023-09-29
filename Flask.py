# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 12:05:13 2022

@author: Loulou
"""
# Libraries importation
from flask import Flask, request, send_file, Response
from PIL import Image 
import cv2
import numpy as np
import requests
from flask import Flask,jsonify,request,json
import jsonpickle
import base64
import tensorflow as tf

# Flask app creation
app = Flask(__name__)

img_size = (256,256) #Size of image to resize

#Importation of IoU metric (Intersection over Union)
class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
  def __init__(self,
               y_true=None,
               y_pred=None,
               num_classes=None,
               ignore_class=None,
               sparse_y_true: bool = True,
               sparse_y_pred: bool = True,
               axis: int = -1,
               name=None,
               dtype=None):
    super(UpdatedMeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.math.argmax(y_pred, axis=-1)
    return super().update_state(y_true, y_pred, sample_weight)


def gray_to_color(img_array):
    """Assign color for each class to convert grayscale into RGB image"""
    x = np.where(img_array==[0],(0,0,0), img_array)
    x = np.where(x==[1], (128,64,128), x)
    x = np.where(x==[2], (70,70,70), x)
    x = np.where(x==[3], (220,220,0), x)
    x = np.where(x==[4], (107,142,35), x)
    x = np.where(x==[5], (70,130,180), x)
    x = np.where(x==[6], (220,20,60), x)
    x = np.where(x==[7], (0,0,142), x)
    return x

#Load the model for prediction    
model = tf.keras.models.load_model('Unet_resnet34_256x256.h5', 
                                   custom_objects={'UpdatedMeanIoU': UpdatedMeanIoU})


# Endpoint to call to run API
@app.route('/image', methods=['POST'])
def image():
    """ Send BGR image as input and send mask under JSON file"""
    r = request
    nparr = np.frombuffer(r.data, np.uint8)
    
    image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR) #decode image
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB) #True RGB color and not BGR
    image_array = cv2.resize(image_array, img_size)
    
    if len(image_array.shape) == 3:
        image_array = np.expand_dims(image_array, 0)
        pred = model.predict(image_array)
        image_array = np.argmax(pred, axis=-1)
        image_array = np.expand_dims(image_array, axis=-1)
        image_array = gray_to_color(image_array)
        image_array = np.squeeze(image_array, axis=0)
    
    
    response = {'message': 'image received. size={}x{}'.format(image_array.shape[2], image_array.shape[1]), 'img':image_array
                }
    
    response_pickled = jsonpickle.encode(response)
    
    return Response(response=response_pickled, status=200, mimetype="application/json")
    

app.run(host='127.0.0.1', port=5000)

