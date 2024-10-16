# Get list of [input numpy array, output numpy array] pairs
from datasets import load_dataset
import numpy as np

ds = load_dataset("ylecun/mnist")

def preprocess_image(example):
    # Convert the image to a numpy array
    image_array = np.array(example['image'])
    # Flatten the image to a 1D array of 784 elements (28*28 pixels)
    image_array = image_array.flatten()
    # Normalize the pixel values to be between 0 and 1 (divide by max value of a pixel)
    image_array = image_array / 255.0
    example['image'] = image_array
    return example

ds = ds.map(preprocess_image)


train_images = np.array(ds["train"]["image"])
train_labels = np.array(ds["train"]["label"])
test_images = np.array(ds["test"]["image"])
test_labels = np.array(ds["test"]["label"])

print("Training images shape:", train_images.shape)
print("Training labels shape:", train_labels.shape)
print("Test images shape:", test_images.shape)
print("Test labels shape:", test_labels.shape)

# Save the preprocessed data to .npy files
np.save('data/train_images.npy', train_images)
np.save('data/train_labels.npy', train_labels)
np.save('data/test_images.npy', test_images)
np.save('data/test_labels.npy', test_labels)