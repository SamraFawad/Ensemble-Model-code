#import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Average
from tensorflow.keras.applications import NASNetMobile, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator , load_img, img_to_array

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, f1_score
from sklearn.model_selection import train_test_split, train_test_split
from sklearn.metrics import classification_report
import os
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load and preprocess the Kvasir dataset
# Data augmentation for training images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Data preprocessing for test images (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)


batch_size = 16
train_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/Colab Notebooks/hyper-kvasir',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
)

val_generator = test_datagen.flow_from_directory(
    '/content/drive/MyDrive/Colab Notebooks/kvasir-dataset',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
)

# Define the directory where your hyper Kvasir dataset is located
dataset_dir = '/content/drive/MyDrive/Colab Notebooks/hyper-kvasir'

# Define the list of class labels based on your dataset
class_labels = sorted(os.listdir(dataset_dir))

# Initialize empty lists to store images and labels
images = []
labels = []

# Loop through each class label folder
for label in class_labels:
    label_dir = os.path.join(dataset_dir, label)
    file_list = os.listdir(label_dir)

    # Loop through each image in the class label folder
    for filename in file_list:
        img = load_img(os.path.join(label_dir, filename), target_size=(224, 224))
        img = img_to_array(img)
        img = img / 255.0  # Normalize pixel values to the range [0, 1]
        images.append(img)
        labels.append(class_labels.index(label))

# Convert the lists to NumPy arrays
X = np.array(images)
y = to_categorical(labels, num_classes=len(class_labels))

# Optionally, you can use data augmentation to further preprocess your training data
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the data augmentation generator on your training data
train_datagen.fit(train_generator)

# Apply data augmentation during training
batch_size = 16
train_generator = train_datagen.flow(train_generator, train_generator, batch_size=batch_size)

# Define the path to your KVASIR v2 dataset directory
dataset_dir = "/content/drive/MyDrive/Colab Notebooks/kvasir-dataset"

# Define the image size for resizing
image_size = (224, 224)

# Load images and labels
images = []
labels = []

class_labels = os.listdir(dataset_dir)

for label in class_labels:
    class_dir = os.path.join(dataset_dir, label)
    if os.path.isdir(class_dir):
        for filename in os.listdir(class_dir):
            if filename.endswith(".jpg"):
                img_path = os.path.join(class_dir, filename)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
                img = cv2.resize(img, image_size)  # Resize the image to the desired size
                images.append(img)
                labels.append(label)

# Encode class labels into numerical values

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Define the NASNet Mobile model
nasnet_input = Input(shape=(224, 224, 3))
nasnet_base_model = NASNetMobile(weights='imagenet', include_top=False, input_tensor=nasnet_input)
nasnet_x = GlobalAveragePooling2D()(nasnet_base_model.output)
nasnet_output = Dense(len(class_labels), activation='softmax')(nasnet_x)
nasnet_model = Model(inputs=nasnet_input, outputs=nasnet_output)

# Define the EfficientNet B0 model
efficientnet_input = Input(shape=(224, 224, 3))
efficientnet_base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=efficientnet_input)
efficientnet_x = GlobalAveragePooling2D()(efficientnet_base_model.output)
efficientnet_output = Dense(len(class_labels), activation='softmax')(efficientnet_x)
efficientnet_model = Model(inputs=efficientnet_input, outputs=efficientnet_output)


# Define the weights for the ensemble
nasnet_weight = 0.6
efficientnet_weight = 0.4

# Combine the two models using a weighted average
ensemble_output = Average()([nasnet_output * nasnet_weight, efficientnet_output * efficientnet_weight])

# Create the ensemble model
ensemble_model = Model(inputs=[nasnet_input, efficientnet_input], outputs=ensemble_output)
#ALTERNATE EXECUTION
# Compile the ensemble model
#ensemble_model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Define uncertainty quantification
class GaussianNoise(tf.keras.layers.Layer):
    def __init__(self, stddev, **kwargs):
        super(GaussianNoise, self).__init__(**kwargs)
        self.stddev = stddev

    def call(self, inputs, training=None):
        if training:
            noise = tf.random.normal(tf.shape(inputs), stddev=self.stddev)
            return inputs + noise
        return inputs

# Add Gaussian noise for uncertainty quantification
ensemble_model_output = GaussianNoise(stddev=0.1)(ensemble_model.output)
ensemble_model_with_uncertainty = Model(inputs=ensemble_model.input, outputs=ensemble_model_output)

# Compile the model
ensemble_model_with_uncertainty.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Train the ensemble model
ensemble_model_with_uncertainty.fit([np.array(train_generator), np.array(train_generator)], np.array(val_generator), epochs=30, batch_size=32)

# Evaluate the ensemble model on the test set
test_loss, test_accuracy = ensemble_model_with_uncertainty.evaluate([np.array(val_generator), np.array(val_generator)], np.array(val_generator))

