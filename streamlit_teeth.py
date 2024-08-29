import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

target_shape = (224,224)
model = tf.keras.models.load_model('mobilenet_1.00_224_0.h5')

uploaded_file = st.file_uploader("choose a image file",type = "jpg")

map_dict = {
    0: 'CaS',
    1: 'CoS',
    2: 'Gum',
    3: 'MC',
    4: 'OC',
    5: 'OLP',
    6: 'OT'
}

img_ = ImageDataGenerator(rescale=1 / 255.)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()),dtype=np.uint8)
    cv_img = cv2.imdecode(file_bytes,1)
    cv_img = cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)
    resized = cv2.resize(cv_img,target_shape)
    resized = resized / 255.
    img_reshape = resized[np.newaxis, ...]

    st.image(cv_img, channels="RGB")

    genrate_pred = st.button("Generate Prediction")
    if genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("predicted label for the image is {}".format(map_dict [prediction]))



