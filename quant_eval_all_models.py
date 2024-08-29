import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm
from numpy import cov
from numpy import trace
from keras.models import load_model
from PIL import Image
import os

# Loading the saved models
cpgan_generator = load_model('E:/cpgan_models/res_64/model_28000/generator', compile=False)
cgan_generator = load_model('E:/cganmodaug_model/ac_gan_generator.h5', compile=False)
stylegan_generator = load_model('E:/stylegan_model/sg_gan_generator.h5', compile=False)


# Loading the InceptionV3 model
inception_model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

# Function to preprocess images for Inception model
def preprocess_for_inception(images):
    images = tf.image.resize(images, (299, 299))
    return images

# Function to calculate Inception Score (IS)
def calculate_inception_score(images, n_split=10, eps=1E-16):
    images = np.array(images)
    images = preprocess_for_inception(images)
    preds = inception_model.predict(images)
    scores = []
    n_part = int(images.shape[0] / n_split)
    for i in range(n_split):
        part = preds[i * n_part:(i + 1) * n_part]
        kl_div = part * (np.log(part + eps) - np.log(np.expand_dims(np.mean(part, axis=0), 0) + eps))
        sum_kl_div = np.sum(kl_div, axis=1)
        avg_kl_div = np.mean(sum_kl_div)
        avg_kl_div = np.clip(avg_kl_div, -700, 700)  # Clip to prevent overflow
        is_score = np.exp(avg_kl_div)
        scores.append(is_score)
    return np.mean(scores), np.std(scores)


# Function to calculate Frechet Inception Distance (FID)
def calculate_fid(real_images, generated_images):
    real_images = preprocess_for_inception(real_images)
    generated_images = preprocess_for_inception(generated_images)
    
    act1 = inception_model.predict(real_images)
    act2 = inception_model.predict(generated_images)
    
    mu1, sigma1 = np.mean(act1, axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), cov(act2, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

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

# Function to load and preprocess real images
def load_real_images(directory, num_images):
    image_files = [os.path.join(directory, img) for img in os.listdir(directory) if img.endswith('.jpg') or img.endswith('.png')]
    images = []
    for img_path in image_files[:num_images]:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((299, 299))
        img = np.array(img)
        images.append(img)
    images = np.array(images)
    return images

# Defining the number of images to generate
num_images = 1000  # Number of images to generate for evaluation

# Loading real images
real_images = load_real_images('E:/oral disease dataset/Calculus/Calculus', num_images=num_images)  # Adjust path and number as needed

# My usage with a condition
condition = np.random.randint(0, 6, size=(num_images, 1))  # Assuming 6 classes

# Generating images for each model
cpgan_images = generate_images(cpgan_generator, num_images=num_images, condition=condition)
cgan_images = generate_images(cgan_generator, num_images=num_images, condition=condition)
stylegan_images = generate_images(stylegan_generator, num_images=num_images, condition=None)

# Calculating IS and FID for CPGAN
cpgan_is, cpgan_is_std = calculate_inception_score(cpgan_images)
cpgan_fid = calculate_fid(real_images, cpgan_images)
print(f"CPGAN IS: {cpgan_is} ± {cpgan_is_std}, FID: {cpgan_fid}")

# Calculating IS and FID for CGAN
cgan_is, cgan_is_std = calculate_inception_score(cgan_images)
cgan_fid = calculate_fid(real_images, cgan_images)
print(f"CGAN IS: {cgan_is} ± {cgan_is_std}, FID: {cgan_fid}")

# Calculating IS and FID for StyleGAN
stylegan_is, stylegan_is_std = calculate_inception_score(stylegan_images)
stylegan_fid = calculate_fid(real_images, stylegan_images)
print(f"StyleGAN IS: {stylegan_is} ± {stylegan_is_std}, FID: {stylegan_fid}")
