import gradio as gr

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models import *

import warnings
warnings.filterwarnings("ignore")


# preparing paths for functions
current_directory = os.getcwd()
all_images_path = '.'

# get model directory
model_name = '0508_saved_model_small_sample_vgg16_base.h5'
model_folder = 'models'
model_path = os.path.join(current_directory, model_folder, model_name)


def model_init():
    # set model
    mode = 'predict' #'train' or 'predict'
    model_type = 'vgg16'
    model_tag = 'base'
    model_id = '{:s}_{:s}'.format(model_type, model_tag)
    # set params
    bs = 8
    epochs = 20
    freeze_backbone = True
    
    # init model
    model = FacePrediction(img_dir=all_images_path, model_type=model_type)
    model.define_model(freeze_backbone=freeze_backbone)
    # model.model.summary()
    
    # use our own load model function to load
    model.load_weights(model_path)
    return model

def model_predict_bmi(tmp_img_path):
    # # Save the uploaded image to a temporary file
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
    #     tmp.write(image.read())
    #     tmp_path = tmp.name

    model = model_init()
    bmi = model.predict_external(tmp_img_path, show_img=False)
    output_bmi = float(bmi[0][0])
    output_bmi = np.round(output_bmi, 3)
    output_results = 'Your BMI is: {}'.format(output_bmi)
    return output_results


# Create the Gradio interface
iface = gr.Interface(
    fn=model_predict_bmi,
    #inputs=[gr.Image(type="filepath", label="Input Image")],
    inputs=[gr.Image(source="webcam", streaming=True, 
                     type="filepath", label="Input Image")],
    outputs="text",
    title="BMI Prediction",
    description="Use webcam to capture your face and get the predicted BMI.",
    live=False
)

# Run the interface
iface.launch(server_port=8888, share=True)
