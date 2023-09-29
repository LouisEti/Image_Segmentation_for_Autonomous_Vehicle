from distutils.command.upload import upload
import streamlit as st
from PIL import Image
import numpy as np
from tables import UnImplemented
import requests
import json
import cv2
import jsonpickle
import numpy as np 
from PIL import Image
import cv2

# Quelques importation d'image RGB et de leur masque annoté pour l'exemple du fonctionnement de l'API
image_path = [r"D:\Mes Documents\Downloads\leftImg8bit\val\frankfurt\frankfurt_000000_000294_leftImg8bit.png",
                r"D:\Mes Documents\Downloads\leftImg8bit\val\frankfurt\frankfurt_000000_000576_leftImg8bit.png",
                r"D:\Mes Documents\Downloads\leftImg8bit\val\frankfurt\frankfurt_000000_001016_leftImg8bit.png"]

true_image = [r"D:\Mes Documents\Downloads\P8_Cityscapes_gtFine_trainvaltest\gtFine\val\frankfurt\frankfurt_000000_000294_gtFine_color.png",
                r"D:\Mes Documents\Downloads\P8_Cityscapes_gtFine_trainvaltest\gtFine\val\frankfurt\frankfurt_000000_000576_gtFine_color.png",
                r"D:\Mes Documents\Downloads\P8_Cityscapes_gtFine_trainvaltest\gtFine\val\frankfurt\frankfurt_000000_001016_gtFine_color.png"]               

# URL vers laquelle on envoie une image grâce à la méthode 'POST' 
url = 'http://51.83.78.142:5000/image'


list_img = []
for  path in image_path:
    img = Image.open(path)
    list_img.append(img)

def display_img(path):
    img = Image.open(path)
    img = img.resize((256,256))
    st.image(img)

def display_fromarray(image_array):
    img = Image.fromarray(image_array.astype(np.uint8))
    st.image(img)

# Streamlit objects
image_choosen = st.selectbox(label='Select the image', options=image_path) #Boîte de sélection des images 
index = image_path.index(image_choosen)

image = open(image_choosen, 'rb').read()
reponse = requests.post(url, data=image)
image = jsonpickle.decode(reponse.text)['img']


if st.button(label="Predict"): # bouton predict
    col1, col2, col3 = st.columns(3)
    
    with col1: #Image 1
        st.subheader('Image RGB')
        display_img(image_choosen)

    with col2: #Image 2
        st.subheader('Masque annoté')
        display_img(true_image[index])

    with col3: #Image 3
        st.subheader('Masque prédit')
        display_fromarray(image)
    
else : 
    pass
