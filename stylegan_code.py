import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, Embedding, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

# Constants
IMAGE_SIZE = 64  # Image size should match in generator, discriminator, and data processing
LATENT_DIM = 100
BATCH_SIZE = 32
EPOCHS = 50000
SAVE_INTERVAL = 2000

# Load and preprocess the dataset
original_dirs = {
    'Calculus': 'E:/oral disease dataset/Calculus/Calculus',
    'Caries': 'E:/oral disease dataset/Data caries/Data caries/caries augmented data set/preview',
    'Gingivitis': 'E:/oral disease dataset/Gingivitis/Gingivitis',
    'Ulcers': 'E:/oral disease dataset/Mouth Ulcer/Mouth Ulcer/Mouth_Ulcer_augmented_DataSet/preview',
    'Tooth Discoloration': 'E:/oral disease dataset/Tooth Discoloration/Tooth Discoloration/Tooth_discoloration_augmented_dataser/preview',
    'Hypodontia': 'E:/oral disease dataset/hypodontia/hypodontia'
}

NUM_CLASSES = len(original_dirs)

def data_generator(original_dirs, batch_size, image_size):
    class_names = list(original_dirs.keys())
    while True:
        images = []
        labels = []
        for _ in range(batch_size):
            class_name = np.random.choice(class_names)
            class_path = original_dirs[class_name]
            class_label = class_names.index(class_name)
           
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_file = np.random.choice(image_files)
            image_path = os.path.join(class_path, image_file)
           
            try:
                img = load_img(image_path, target_size=(image_size, image_size))  # Ensure the size matches IMAGE_SIZE
                img_array = img_to_array(img)
                img_array = (img_array - 127.5) / 127.5  # Normalize to [-1, 1]
                images.append(img_array)
                labels.append(class_label)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
       
        yield np.array(images), np.array(labels)

# Define the mapping network (from latent space to style vectors)
def build_mapping_network():
    latent = Input(shape=(LATENT_DIM,))
    x = Dense(256)(latent)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    style = Dense(256)(x)
    return Model(latent, style)

# Define the generator
def build_generator():
    noise = Input(shape=(LATENT_DIM,))
    label = Input(shape=(1,), dtype='int32')

    label_embedding = Embedding(NUM_CLASSES, 50)(label)
    label_embedding = Flatten()(label_embedding)

    x = Concatenate()([noise, label_embedding])

    x = Dense(8 * 8 * 512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((8, 8, 512))(x)

    def style_block(x, style):
        gamma = Dense(K.int_shape(x)[-1])(style)  # Match the channels of x
        beta = Dense(K.int_shape(x)[-1])(style)   # Match the channels of x
        gamma = Reshape((1, 1, K.int_shape(x)[-1]))(gamma)
        beta = Reshape((1, 1, K.int_shape(x)[-1]))(beta)
        return x * gamma + beta

    style = build_mapping_network()(noise)
    x = style_block(x, style)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)  # 8x8 to 16x16
    x = LeakyReLU(alpha=0.2)(x)
    x = style_block(x, style)

    x = Conv2DTranspose(128, 4, strides=2, padding='same')(x)  # 16x16 to 32x32
    x = LeakyReLU(alpha=0.2)(x)
    x = style_block(x, style)

    x = Conv2DTranspose(64, 4, strides=2, padding='same')(x)  # 32x32 to 64x64
    x = LeakyReLU(alpha=0.2)(x)
    x = style_block(x, style)

    x = Conv2D(3, 3, padding='same', activation='tanh')(x)  # Output layer
   
    model = Model([noise, label], x)
    return model

# Define the discriminator
def build_discriminator():
    img = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))  # Ensure this matches IMAGE_SIZE
   
    x = Conv2D(64, 3, strides=2, padding='same')(img)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
   
    x = Conv2D(128, 3, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
   
    x = Conv2D(256, 3, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(512, 3, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
   
    x = Flatten()(x)
   
    validity = Dense(1, activation='sigmoid')(x)
    label = Dense(NUM_CLASSES, activation='softmax')(x)
   
    model = Model(img, [validity, label])
    return model

# Build and compile the models
generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
                      optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
                      metrics=['accuracy'])

discriminator.trainable = False
noise = Input(shape=(LATENT_DIM,))
label = Input(shape=(1,))
img = generator([noise, label])
validity, aux = discriminator(img)
combined = Model([noise, label], [validity, aux])
combined.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
                 optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

# Training loop
def train(epochs, batch_size, save_interval):
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
   
    data_gen = data_generator(original_dirs, batch_size, IMAGE_SIZE)
   
    for epoch in range(epochs):
        # Train discriminator
        real_imgs, real_labels = next(data_gen)
        print(f"Real images shape: {real_imgs.shape}")  # Debugging line
       
        noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
        fake_labels = np.random.randint(0, NUM_CLASSES, (batch_size, 1))
        gen_imgs = generator.predict([noise, fake_labels])
        print(f"Generated images shape: {gen_imgs.shape}")  # Debugging line
       
        d_loss_real = discriminator.train_on_batch(real_imgs, [real, real_labels])
        d_loss_fake = discriminator.train_on_batch(gen_imgs, [fake, fake_labels.reshape(-1)])
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
       
        # Train generator
        noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
        fake_labels = np.random.randint(0, NUM_CLASSES, (batch_size, 1))
        g_loss = combined.train_on_batch([noise, fake_labels], [real, fake_labels.reshape(-1)])
       
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, D loss: {d_loss[0]}, G loss: {g_loss[0]}")
       
        if epoch % save_interval == 0:
            save_imgs(epoch)
   
    # Save the final model
    generator.save('E:/stylegan_model/sg_gan_generator.h5')
    discriminator.save('E:/stylegan_model/sg_gan_discriminator.h5')

# Function to save generated images
def save_imgs(epoch):
    r, c = 2, 3
    noise = np.random.normal(0, 1, (r * c, LATENT_DIM))
    labels = np.arange(0, NUM_CLASSES).reshape(-1, 1)
    gen_imgs = generator.predict([noise, labels])
    gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to [0, 1]
   
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt])
            axs[i, j].set_title(f"Class: {labels[cnt][0]}")
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(f"E:/stylegan_images/sg_gan_generated_{epoch}.png")
    plt.close()

# Start training
train(EPOCHS, BATCH_SIZE, SAVE_INTERVAL)