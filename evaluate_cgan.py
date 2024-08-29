import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras import backend as K
import gc

# Forcing TensorFlow to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Constants
IMAGE_SIZE = 32  # Image size set to 32x32
CHANNELS = 3
NOISE_DIM = 100
NUM_CLASSES = 6
BATCH_SIZE = 2  # Further reducing batch size to lower memory consumption
NUM_SAMPLES = 100  # Reducing the number of samples to generate for combined dataset

# Clearing any previous sessions and manually trigger garbage collection
K.clear_session()
gc.collect()

# Loading the saved CPGAN generator model (replace with your actual path)
cpgan_generator_path = 'E:/cganmodaug_model/ac_gan_generator.h5'  # Example path
generator = load_model(cpgan_generator_path)

# Loading and preprocessing the dataset in smaller batches
def load_images_and_labels(original_dirs, batch_size=1000):
    images = []
    labels = []
    image_extensions = ['.png', '.jpg', '.jpeg']
    
    for label, (class_name, class_path) in enumerate(original_dirs.items()):
        print(f"Loading images from {class_path}...")
        image_files = []
        for root, dirs, files in os.walk(class_path):
            for img_name in files:
                if any(img_name.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, img_name))
        
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i+batch_size]
            for img_path in batch_files:
                try:
                    img = load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
                    img_array = img_to_array(img)
                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
            gc.collect()  # Collect garbage after each batch to free up memory
    
    images = np.array(images, dtype='float32')
    labels = np.array(labels)
    images = (images - 127.5) / 127.5  # Normalize images to [-1, 1]
    print(f"Loaded {len(images)} images.")
    return images, labels

# Defining directories
original_dirs = {
    'Calculus': 'E:/oral disease dataset/Calculus/Calculus',
    'Caries': 'E:/oral disease dataset/Data caries/Data caries/caries augmented data set/preview',
    'Gingivitis': 'E:/oral disease dataset/Gingivitis/Gingivitis',
    'Ulcers': 'E:/oral disease dataset/Mouth Ulcer/Mouth Ulcer/Mouth_Ulcer_augmented_DataSet/preview',
    'Tooth Discoloration': 'E:/oral disease dataset/Tooth Discoloration/Tooth Discoloration/Tooth_discoloration_augmented_dataser/preview',
    'Hypodontia': 'E:/oral disease dataset/hypodontia/hypodontia'
}

# Loading images and labels in smaller batches
images, labels = load_images_and_labels(original_dirs, batch_size=500)

# Function to build the classifier
def build_classifier():
    classifier = Sequential([
        Flatten(input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
        Dense(32, activation='relu'),  # Reduce the number of neurons to lower memory usage
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')  # Use softmax for multi-class classification
    ])
    classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return classifier

# Training and evaluating on the original dataset
def evaluate_on_original_dataset(images, labels):
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    classifier = build_classifier()
    classifier.fit(X_train, y_train, epochs=20, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))
    y_pred = np.argmax(classifier.predict(X_test), axis=1)
    
    # Calculating accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Classifier Accuracy on Original Dataset: {accuracy * 100:.2f}%')
    
    # Calculating F1 Score (macro average)
    f1 = f1_score(y_test, y_pred, average='macro')  # Use 'macro' for multi-class
    print(f'Classifier F1 Score on Original Dataset: {f1:.4f}')
    
    return accuracy, f1

# Generating the combined dataset
def generate_combined_dataset(generator, images, labels, num_samples):
    real_images = images
    real_labels = labels  # Keep original labels for real images

    # Generating fake images
    noise = np.random.normal(0, 1, (num_samples, NOISE_DIM))
    gen_labels = np.random.randint(0, NUM_CLASSES, num_samples)
    fake_images = generator.predict([noise, gen_labels], batch_size=BATCH_SIZE)
    fake_images_resized = np.array([img_to_array(array_to_img(img).resize((IMAGE_SIZE, IMAGE_SIZE))) for img in fake_images])
    fake_labels = gen_labels  # Use the generated labels for fake images

    # Combining and creating labels
    X_combined = np.concatenate([real_images, fake_images_resized])
    y_combined = np.concatenate([real_labels, fake_labels])

    return X_combined, y_combined

# Training and evaluating on the combined dataset
def evaluate_on_combined_dataset(generator, images, labels, num_samples):
    X_combined, y_combined = generate_combined_dataset(generator, images, labels, num_samples)
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)
    classifier = build_classifier()
    classifier.fit(X_train, y_train, epochs=20, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))
    y_pred = np.argmax(classifier.predict(X_test), axis=1)
    
    # Calculating accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Classifier Accuracy on Combined Dataset: {accuracy * 100:.2f}%')
    
    # Calculating F1 Score (macro average)
    f1 = f1_score(y_test, y_pred, average='macro')  # Use 'macro' for multi-class
    print(f'Classifier F1 Score on Combined Dataset: {f1:.4f}')
    
    return accuracy, f1

# Running the evaluations
original_accuracy, original_f1 = evaluate_on_original_dataset(images, labels)
combined_accuracy, combined_f1 = evaluate_on_combined_dataset(generator, images, labels, num_samples=NUM_SAMPLES)

# Comparing results
print(f'Accuracy Improvement: {combined_accuracy - original_accuracy:.2f}%')
print(f'F1 Score Improvement: {combined_f1 - original_f1:.4f}')


