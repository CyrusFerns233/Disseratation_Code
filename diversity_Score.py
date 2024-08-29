import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import os

# Loading the saved models (Adjust paths if needed)
cpgan_generator = load_model('E:/cpgan_models/res_64/model_28000/generator', compile=False)
cgan_generator = load_model('E:/cganmodaug_model/ac_gan_generator.h5', compile=False)
stylegan_generator = load_model('E:/stylegan_model/sg_gan_generator.h5', compile=False)

# Loading the InceptionV3 model
inception_model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

# Function to preprocess images for Inception model
def preprocess_for_inception(images):
    images = tf.image.resize(images, (299, 299))
    return images

# Function to generate images using a model
def generate_images(generator, num_images, latent_dim=100, condition=None):
    noise = np.random.normal(0, 1, (num_images, latent_dim))
    
    if condition is not None:
        condition_input = np.array(condition)  # Adjust as per your condition format
        generated_images = generator.predict([noise, condition_input])
    else:
        if generator.input_shape[0] is not None and len(generator.input_shape) == 2:
            # If the model expects two inputs but condition is None, create a dummy condition
            condition_input = np.zeros((num_images, 1))  # or any appropriate dummy input
            generated_images = generator.predict([noise, condition_input])
        else:
            generated_images = generator.predict(noise)
    
    generated_images = (generated_images * 127.5 + 127.5).astype(np.uint8)
    return generated_images

# Function to calculate Diversity Score
def calculate_diversity_score(images, model, num_pairs=10000):
    # Preprocessing images for the model
    images = preprocess_for_inception(images)
    
    # Extracting features using InceptionV3
    features = model.predict(images)
    
    # Randomly selecting pairs of features
    num_images = features.shape[0]
    indices = np.random.choice(num_images, (num_pairs, 2), replace=True)
    
    # Calculating pairwise Euclidean distances
    distances = np.linalg.norm(features[indices[:, 0]] - features[indices[:, 1]], axis=1)
    
    # Calculating the diversity score as the mean of these distances
    diversity_score = np.mean(distances)
    
    return diversity_score

# Defining the number of images to generate
num_images = 1000  # Number of images to generate for evaluation

# My usage with a condition 
condition = np.random.randint(0, 6, size=(num_images, 1))  # Assuming 6 classes

# Generating images for each model
cpgan_images = generate_images(cpgan_generator, num_images=num_images, condition=condition)
cgan_images = generate_images(cgan_generator, num_images=num_images, condition=condition)
stylegan_images = generate_images(stylegan_generator, num_images=num_images, condition=None)

# Calculating Diversity Score for CPGAN
cpgan_diversity_score = calculate_diversity_score(cpgan_images, inception_model)
print(f"CPGAN Diversity Score: {cpgan_diversity_score}")

# Calculating Diversity Score for CGAN
cgan_diversity_score = calculate_diversity_score(cgan_images, inception_model)
print(f"CGAN Diversity Score: {cgan_diversity_score}")

# Calculating Diversity Score for StyleGAN
stylegan_diversity_score = calculate_diversity_score(stylegan_images, inception_model)
print(f"StyleGAN Diversity Score: {stylegan_diversity_score}")
