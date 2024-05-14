from django.shortcuts import render
import tensorflow as tf
from tensorflow import keras
from PIL import Image as pilimage
import numpy as np
import os
from django.core.files.storage import FileSystemStorage

#Deep learning libs
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
from keras.models import Sequential, Model
from keras.layers import GlobalAveragePooling2D, Dense, Layer
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam , Adamax
from keras import regularizers
from keras.applications import EfficientNetB3, MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input

# Create your views here.
media = 'media'
import keras.backend as K

def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

class SEBlock(Layer):
    def __init__(self, ratio=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        super(SEBlock, self).build(input_shape)
        channels = input_shape[-1]
        self.global_pooling = GlobalAveragePooling2D()
        self.dense1 = Dense(channels // self.ratio, activation='relu')
        self.dense2 = Dense(channels, activation='sigmoid')

    def call(self, inputs):
        se = self.global_pooling(inputs)
        se = K.expand_dims(K.expand_dims(se, axis=1), axis=1)
        se = self.dense1(se)
        se = self.dense2(se)
        
        # Scale the input features
        return inputs * se

    def compute_output_shape(self, input_shape):
        return input_shape

import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import os
from PIL import Image
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf

media = 'media'
model = load_model("CNN&SE&LSTM.h5", custom_objects={'f1_score': f1_score, 'SEBlock': SEBlock})

def makepredictions(path):
    img = pilimage.open(path)
    img_d = img.resize((256, 256))  # Resize the image to match the model input shape
    if len(np.array(img_d).shape) < 4:
        rgb_img = pilimage.new("RGB", img_d.size)
        rgb_img.paste(img_d)
    else:
        rgb_img = img_d

    rgb_img = np.array(rgb_img, dtype=np.float64)
    rgb_img = np.expand_dims(rgb_img, axis=0)  # Add batch dimension
    predictions = model.predict(rgb_img)
    a = int(np.argmax(predictions))
    
    if a == 0:
        a = "  Glioma Tumor"
    elif a == 1:
        a = "  Meningioma Tumor"
    elif a == 2:
        a = "  No Tumor"
    else:
        a = "  Pituitary Tumor"
    return a


import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.models import load_model

def generate_grad_cam(img_path):
    # Load and preprocess an image
    img = image.load_img(img_path, target_size = (256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Normalize the pixel values

    # Get the output tensor of the last convolutional layer in your custom model
    last_conv_layer = model.get_layer('conv2d_34') 

    # Create a model that maps the input image to the activations of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

    # Compute the gradient of the predicted class with respect to the output feature map of the last conv layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        loss = predictions[:, np.argmax(predictions)]

    grads = tape.gradient(loss, conv_outputs)[0]

    # Compute the CAM
    cam = np.mean(conv_outputs[0], axis=-1)
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (256, 256))  # Resize CAM to match the size of the original image
    cam = cam / cam.max()

    # Generate heatmap overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Convert the input image to uint8
    x_uint8 = (x[0] * 255).astype(np.uint8)

    # Superimpose the heatmap on the original image
    superimposed_img = cv2.addWeighted(
        x_uint8, 0.6,
        cv2.resize(heatmap, (256, 256)), 0.4, 0  # Resize heatmap to match the size of the original image
    )

    return superimposed_img, heatmap


def index(req):
    if req.method == "POST" and req.FILES['upload']:
        f = req.FILES['upload']
        if f == '':
            err = 'No files selected'
            return render(req, 'result.html', {'err' : err})
        upload = req.FILES['upload']
        fss = FileSystemStorage()
        file = 'upload.jpg'  # Set the filename to 'upload.jpg'
        fss.delete(file)  # Delete the existing 'upload.jpg' if it exists
        file = fss.save(file, upload)
        file_url = fss.url(file)
        imgPath = os.path.join(media, file)

        # Read the image using PIL (to keep it in RGB format)
        img = Image.open(imgPath)
        img = np.array(img)

        predictions = makepredictions(imgPath)
        
        # Superimposed image
        superimposed_img, heatmap = generate_grad_cam(imgPath)
        superimposed_path = os.path.join(media, 'superimposed_image3.jpg')
        cv2.imwrite(superimposed_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
        
        heatmap_path = os.path.join(media, 'GradCamHeatmap.jpg')
        cv2.imwrite(heatmap_path, heatmap)
        return render(req, 'result.html', {'pred' : predictions, 'file_url' : file_url, 'imgPath' : imgPath, 'heatmap_path': heatmap_path, 'superimposed_path': superimposed_path})
    else:
        return render(req, 'result.html', {'returned_image_path': None, 'heatmap_path': None})
