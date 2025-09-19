import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from pillow import Image
import os

# Load the Hateful Memes dataset
dataset = load_dataset("neuralcatcher/hateful_memes")

# Create directories for dataset storage
base_dir = "dataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")

bullying_dir_train = os.path.join(train_dir, "bullying")
non_bullying_dir_train = os.path.join(train_dir, "non_bullying")
bullying_dir_val = os.path.join(val_dir, "bullying")
non_bullying_dir_val = os.path.join(val_dir, "non_bullying")

os.makedirs(bullying_dir_train, exist_ok=True)
os.makedirs(non_bullying_dir_train, exist_ok=True)
os.makedirs(bullying_dir_val, exist_ok=True)
os.makedirs(non_bullying_dir_val, exist_ok=True)


# Function to save images to respective folders
def save_images(dataset_split, save_dir, split_ratio=0.8):
    for i, data in enumerate(dataset_split):
        image = Image.open(data["img"]).convert("RGB")  # Open the image
        label = data["label"]  # 1 = bullying, 0 = non-bullying

        # Define save path
        if label == 1:
            save_path = os.path.join(bullying_dir_train if i < len(dataset_split) * split_ratio else bullying_dir_val,
                                     f"{i}.jpg")
        else:
            save_path = os.path.join(
                non_bullying_dir_train if i < len(dataset_split) * split_ratio else non_bullying_dir_val, f"{i}.jpg")

        image.save(save_path)


# Save images from dataset
save_images(dataset["train"], train_dir)
save_images(dataset["validation"], val_dir)

# Image preprocessing for ResNet50
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Load pre-trained ResNet50 model + a custom classifier on top
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base layers

# Custom model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")


# Plot accuracy/loss
def plot_history(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()


plot_history(history)

# Save the trained model
model.save('bullying_detection_model.h5')
