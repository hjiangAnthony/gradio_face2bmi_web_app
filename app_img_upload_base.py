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
from models_base import *

import warnings
warnings.filterwarnings("ignore")


# preparing paths for functions
current_directory = os.getcwd()
all_images_path = '.'

# get model directory
#model_name = '0520_saved_model_full_sample_uniPredict_vgg16_base_extra_cov_layer.h5'
model_name = '0520_saved_model_full_sample_uniPredict_vgg16_base_no_layer.h5'

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

    model = model_init()
    bmi = model.predict_external(tmp_img_path, show_img=False)
    output_bmi = float(bmi[0][0])
    output_bmi = float(np.round(output_bmi, 3))
    #output_results = 'Your BMI is: {}'.format(output_bmi)
    
    return output_bmi

def get_weight_status(bmi):
    '''
    Standards come from U.S. CDC: https://www.cdc.gov/healthyweight/assessing/bmi/adult_bmi/index.html
    '''
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi <= 24.9:
        return "Healthy Weight"
    elif 25.0 <= bmi <= 29.9:
        return "Overweight"
    else:
        return "Obesity"

def give_advice(bmi):
    status = get_weight_status(bmi)
    if type(bmi) == float:
        if status == None:
            assert 'Input is not valid'
        elif status == 'Healthy Weight':
            advice = 'Your BMI is:{}. '.format(bmi) + 'Great job! (◍•ᴗ•◍)'
            return advice
        elif status != 'Healthy Weight':
            advice = 'Your BMI is:{}. '.format(bmi) + '\nAccording to the U.S. CDC BMI recommendation, your status is {}. Perhaps you could try on a new diet? ‎(·•᷄ࡇ•᷅ ）'.format(status)
            return advice
        else:
            return 'Something went wrong. 😢'

def main(tmp_img_path):
    bmi_result = model_predict_bmi(tmp_img_path)
    health_advice = give_advice(bmi_result)
    return health_advice

# Create the Gradio interface
iface = gr.Interface(
    fn=main,
    inputs=[gr.Image(type="filepath", label="Input Image")],
    outputs="text",
    title="BMI Prediction",
    description="Upload an image and get the predicted BMI.")

# Run the interface
iface.launch(server_port=8888, share=True)
