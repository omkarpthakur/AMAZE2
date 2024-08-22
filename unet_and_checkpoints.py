import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the pre-trained DeepLab model
model = hub.load('https://tfhub.dev/tensorflow/deeplab/cityscapes/1')

def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [513, 513])
    return img

def run_model(image):
    input_image = tf.expand_dims(image, axis=0)
    result = model(input_image)
    segmentation_map = tf.argmax(result['logits'], axis=-1)
    segmentation_map = tf.squeeze(segmentation_map)
    return segmentation_map

def visualize(image, segmentation_map):
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.imshow(image)
    plt.title('Input Image')
    plt.subplot(122)
    plt.imshow(segmentation_map, cmap='viridis')
    plt.title('Segmentation Map')
    plt.colorbar()
    plt.show()

# Usage example
image_path = r"C:\Users\omkar\OneDrive\Desktop\sample 1.jpg"  # Replace with your image path
image = load_image(image_path)
segmentation_map = run_model(image)

visualize(image, segmentation_map)

# You can now use this segmentation map for further processing in your self-driving car application