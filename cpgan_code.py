import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU, BatchNormalization, Embedding, Concatenate, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Constants
CHANNELS = 3
NOISE_DIM = 100
BATCH_SIZE = 32
EPOCHS_PER_RES = 50000  # Number of epochs at each resolution
SAVE_INTERVAL = 200
IMAGE_SAVE_INTERVAL = 1000
MODEL_SAVE_INTERVAL = 2000
NUM_CLASSES = 6  # Number of disease classes
IMAGE_SIZE = 64  # Define IMAGE_SIZE to match the maximum resolution used

# Directories for dataset and models
original_dirs = {
    'Calculus': 'E:/oral disease dataset/Calculus/Calculus',
    'Caries': 'E:/oral disease dataset/Data caries/Data caries/caries augmented data set/preview',
    'Gingivitis': 'E:/oral disease dataset/Gingivitis/Gingivitis',
    'Ulcers': 'E:/oral disease dataset/Mouth Ulcer/Mouth Ulcer/Mouth_Ulcer_augmented_DataSet/preview',
    'Tooth Discoloration': 'E:/oral disease dataset/Tooth Discoloration/Tooth Discoloration/Tooth_discoloration_augmented_dataser/preview',
    'Hypodontia': 'E:/oral disease dataset/hypodontia/hypodontia'
}

# Load and preprocess the dataset
def load_images_and_labels(original_dirs, image_size):
    images = []
    labels = []
    image_extensions = ['.png', '.jpg', '.jpeg']
    for label, (class_name, class_path) in enumerate(original_dirs.items()):
        print(f"Loading images from {class_path}...")
        if not os.path.exists(class_path):
            print(f"Directory does not exist: {class_path}")
            continue
        for root, dirs, files in os.walk(class_path):
            for img_name in files:
                img_path = os.path.join(root, img_name)
                if os.path.isfile(img_path) and any(img_name.lower().endswith(ext) for ext in image_extensions):
                    try:
                        img = load_img(img_path, target_size=(image_size, image_size))
                        img_array = img_to_array(img)
                        images.append(img_array)
                        labels.append(label)
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
                else:
                    print(f"Unsupported file format or not a file: {img_path}")
    images = np.array(images, dtype='float32')
    labels = np.array(labels)
    if images.size > 0:
        images = (images - 127.5) / 127.5  # Normalize images to [-1, 1]
    print(f"Loaded {len(images)} images.")
    return images, labels

def build_discriminator(resolution):
    img = Input(shape=(resolution, resolution, CHANNELS))
    
    # Convolutional layers
    x = Conv2D(64, kernel_size=3, strides=2, padding='same')(img)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    
    if resolution > 8:
        x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)
    
    if resolution > 16:
        x = Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)
    
    if resolution > 32:
        x = Conv2D(512, kernel_size=3, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)
    
    x = Flatten()(x)
    
    conv_output_size = int(x.shape[-1])
    print(f"Conv output size: {conv_output_size} at resolution {resolution}")
    
    x = Dense(1024)(x)

    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(NUM_CLASSES, 1024)(label))
    
    concatenated = Concatenate()([x, label_embedding])
    
    x = Dense(512)(concatenated)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model([img, label], x)
    return model


def build_generator(resolution):
    noise = Input(shape=(NOISE_DIM,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(NUM_CLASSES, NOISE_DIM)(label))
    
    model_input = Concatenate()([noise, label_embedding])
    
    init_size = resolution // 8
    
    x = Dense(128 * init_size * init_size)(model_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((init_size, init_size, 128))(x)
    x = BatchNormalization(momentum=0.8)(x)
    
    if resolution >= 16:
        x = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
    
    if resolution >= 32:
        x = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
    
    if resolution >= 64:
        x = Conv2DTranspose(32, kernel_size=4, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
    
    img = Conv2D(CHANNELS, kernel_size=3, padding='same', activation='tanh')(x)
    
    # Ensure the output image is exactly the resolution we expect
    img = tf.image.resize(img, (resolution, resolution))
    
    model = Model([noise, label], img)
    return model


def train_cpgan(images, labels, max_resolution, epochs_per_res, batch_size, save_interval, image_save_interval, model_save_interval):
    for res in [8, 16, 32, max_resolution]:
        print(f"Training with resolution: {res}x{res}")
        
        discriminator = build_discriminator(res)
        generator = build_generator(res)
        
        optimizer = Adam(0.0002, 0.5)
        discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        discriminator.trainable = False
        
        noise = Input(shape=(NOISE_DIM,))
        label = Input(shape=(1,), dtype='int32')
        generated_img = generator([noise, label])
        
        validity = discriminator([generated_img, label])
        
        combined = Model([noise, label], validity)
        combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        
        for epoch in range(epochs_per_res):
            # Select a random half batch of images
            idx = np.random.randint(0, images.shape[0], batch_size // 2)
            imgs, labels_real = images[idx], labels[idx]
            
            # Resize real images to match the current resolution
            imgs_resized = tf.image.resize(imgs, (res, res))
            
            noise = np.random.normal(0, 1, (batch_size // 2, NOISE_DIM))
            gen_labels = np.random.randint(0, NUM_CLASSES, batch_size // 2)
            gen_imgs = generator.predict([noise, gen_labels])

            # Resize generated images to match the current resolution
            gen_imgs_resized = tf.image.resize(gen_imgs, (res, res))
            
            # Train the discriminator
            d_loss_real = discriminator.train_on_batch([imgs_resized, labels_real], np.ones((batch_size // 2, 1)))
            d_loss_fake = discriminator.train_on_batch([gen_imgs_resized, gen_labels], np.zeros((batch_size // 2, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train the generator
            noise = np.random.normal(0, 1, (batch_size, NOISE_DIM))
            valid_y = np.ones((batch_size, 1))
            gen_labels = np.random.randint(0, NUM_CLASSES, batch_size)
            g_loss = combined.train_on_batch([noise, gen_labels], valid_y)
            
            if epoch % save_interval == 0:
                print(f"{epoch}/{epochs_per_res} [D loss: {d_loss[0]}] [G loss: {g_loss}]")
            
            if epoch % image_save_interval == 0:
                save_imgs(epoch, generator, res)
            
            if epoch % model_save_interval == 0:
                save_model(epoch, generator, discriminator, combined, res)




# Save generated images
def save_imgs(epoch, generator, resolution):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, NOISE_DIM))
    gen_labels = np.random.randint(0, NUM_CLASSES, r * c)
    gen_imgs = generator.predict([noise, gen_labels])
    gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale images 0 - 1

    fig, axs = plt.subplots(r, c, figsize=(10, 10))
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, :])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(f"E:/cpgan_images/gan_{resolution}_{epoch}.png")
    plt.close()

# Save the models
def save_model(epoch, generator, discriminator, combined, resolution):
    model_path = f"E:/cpgan_models/res_{resolution}/model_{epoch}"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    try:
        generator.save(os.path.join(model_path, 'generator'))
        discriminator.save(os.path.join(model_path, 'discriminator'))
        combined.save(os.path.join(model_path, 'combined'))
        print(f"Saved models at epoch {epoch} for resolution {resolution}")
    except Exception as e:
        print(f"Failed to save models at epoch {epoch} for resolution {resolution}: {e}")

# Load dataset
images, labels = load_images_and_labels(original_dirs, IMAGE_SIZE)

# Verify that images have been loaded
if images.size == 0:
    raise ValueError("No images loaded. Please check the dataset directories and ensure images are in the correct format.")

# Train the model
train_cpgan(images, labels, max_resolution=64, epochs_per_res=EPOCHS_PER_RES, batch_size=BATCH_SIZE, save_interval=SAVE_INTERVAL, image_save_interval=IMAGE_SAVE_INTERVAL, model_save_interval=MODEL_SAVE_INTERVAL)
