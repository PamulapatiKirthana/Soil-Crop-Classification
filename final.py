import streamlit as st
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the saved model
saved_model = load_model('model.h5')

# Function for crop prediction based on soil type
def crop_prediction(soil_type):
    crop_mapping = {
        'Black Soil': ['Cotton','Wheat','Groundnut','Tobacco','Chillies','Oilseeds','Rice','Ragi','Maize','Sugarcane','Citrus','Linseed','Sunflower','Millets'],
        'Cinder Soil': ['Roses','Succulents','Cactus','Adenium','Snake Plant','Orchids','Bonsai'],
        'Laterite Soil': ['Tea','Coffee','Rubber','Cashew','Coconut','Wheat','Rice','Pulses','Cotton'],
        'Peat Soil': ['Potatoes','Sugar Beets','Lettuce','Onions','Carrots','Celery'],
        'Yellow Soil': ['Wheat','Cotton','Oilseeds','Millets','Tobacco','Pulses','Maize','Groundnut','Rice','Mango','Orange','Potato']
    }
    return crop_mapping.get(soil_type,[])

# Function to display images of crops in a table format with consistent size
def display_crops_images(crops_list):
    st.write("Crops that can be grown on this soil:")
    num_cols = 3  # Number of columns for the table layout
    num_images = len(crops_list)
    num_rows = -(-num_images // num_cols)  # Calculate number of rows needed

    for i in range(num_rows):
        cols = st.columns(num_cols)
        for j in range(num_cols):
            index = i * num_cols + j
            if index < num_images:
                crop = crops_list[index]
                image_path = f'C://SRET//Soil-Crop Classification//Crops//{crop}.jpeg'  # Update with actual path to crop images

                # Read and resize the image to a consistent size (e.g., 150x150 pixels)
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img, (150, 150))

                cols[j].image(img_resized, caption=crop)

# Define the Streamlit app and its functionality
def app():
    st.title('Soil Crop Prediction')
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            st.image(image, caption='Uploaded Image')

            # Make predictions and get soil type
            image = np.expand_dims(image, axis=0)
            result = saved_model.predict(image)

            # Define train_set for class indices mapping
            train_datagen = ImageDataGenerator(rescale=1./255)
            train_generator = train_datagen.flow_from_directory(
                'C://SRET//Soil-Crop Classification//Soil types',
                target_size=(224, 224),
                batch_size=32,
                class_mode='categorical')

            # Assign the training set to the variable
            train_set = train_generator

            # Map predicted indices to class names
            class_names = {v: k for k, v in train_set.class_indices.items()}
            prediction = class_names[np.argmax(result)]

            st.write('Predicted soil type:', prediction)

            crops_list = crop_prediction(prediction)

            if crops_list:
                display_crops_images(crops_list)
            else:
                st.write("No information available for crops on this soil type.")

if __name__ == '__main__':
    app()