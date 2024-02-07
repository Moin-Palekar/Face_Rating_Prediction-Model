#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[19]:


import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from PIL import Image, ImageOps
import os
from keras.regularizers import l2


# # Image renaming (one time run)

# In[4]:


def rename_images(folder_path):
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Iterate through each image file
    for idx, image_file in enumerate(image_files, 1):
        # Display the image using matplotlib
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)
        plt.imshow(image)
        plt.title(f"Image {idx}/{len(image_files)}: {image_file}")
        plt.show()
        
        # Prompt for the number
        number = int(input(f"Enter the number for image {idx}: "))
        
        # Get the file extension
        _, ext = os.path.splitext(image_file)
        
        # Rename the image file
        new_name = f"{number}_{idx}{ext}"
        os.rename(image_path, os.path.join(folder_path, new_name))
        
    print("Image renaming completed.")

# Example usage:
folder_path = "face_rating_prediction_model_v1"
rename_images(folder_path)


# # Image Padding (one time run)

# In[20]:


from PIL import Image, ImageOps
import os

# Specify the path to your dataset
dataset_path = "testfolder"

# Get a list of all image files in the dataset
image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Initialize variables to store maximum height and width
max_height = 0
max_width = 0

# Find the maximum height and width among all images
for image_file in image_files:
    image_path = os.path.join(dataset_path, image_file)
    with Image.open(image_path) as img:
        width, height = img.size
        max_width = max(max_width, width)
        max_height = max(max_height, height)

# Padding function
def add_padding(image, target_width, target_height):
    left_padding = (target_width - image.width) // 2
    top_padding = (target_height - image.height) // 2
    right_padding = target_width - image.width - left_padding
    bottom_padding = target_height - image.height - top_padding

    padding = (left_padding, top_padding, right_padding, bottom_padding)
    return ImageOps.expand(image, padding, fill=0)  # 'fill' can be adjusted as needed

# Add padding to each image
for image_file in image_files:
    image_path = os.path.join(dataset_path, image_file)
    with Image.open(image_path) as img:
        padded_img = add_padding(img, max_width, max_height)
        
        # Save or overwrite the padded image
        padded_img.save(image_path)

print(f"All images padded to {max_width}x{max_height}")


# # CSV file Conversion (one time run)

# In[22]:


import os
import csv

# Path to the folder containing images
image_folder = "testfolder"

# Path to save the CSV file
csv_file_path = "testfolder/output.csv"

# Open CSV file in write mode
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write header to CSV file
    csv_writer.writerow(['Image_Path', 'Label'])  # Add more columns as needed

    # Iterate through each image in the folder
    for image_file in os.listdir(image_folder):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            # Assuming the image filename includes the label (e.g., cat_01.jpg)
            label = image_file.split('_')[0]  # Extract label from filename
            image_path = os.path.join(image_folder, image_file)
            # Write image path and label to CSV file
            csv_writer.writerow([image_path, label])


# # Creating numpy arrays for training 

# In[24]:


import numpy as np
import csv
from sklearn.model_selection import train_test_split
from PIL import Image

# Load CSV file into a list of lists
csv_file_path = "testfolder/output.csv"

data = []
with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    header = next(csv_reader)  # Skip the header
    for row in csv_reader:
        data.append(row)

# Convert the list of lists into a NumPy array
data_array = np.array(data)

# Assuming the 'Image_Path' column is the first column and 'Label' is the second column
image_paths = data_array[:, 0]  # Image paths
labels = data_array[:, 1]  # Labels

# Load and preprocess images
images = []

for image_path in image_paths:
    img = Image.open(image_path)
    img_array = np.array(img)
    images.append(img_array)

# Convert the list of images into a NumPy array
x = np.array(images)
y = np.array(labels)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= None, random_state=42)

y_train = y_train.astype(float)
y_test = y_test.astype(float)


# Print the shapes of the resulting arrays
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)


# In[11]:


plt.imshow(x_train[3, :])


# # The Model

# In[3]:


model = Sequential([
    Conv2D(22, (3, 3), activation='relu', input_shape=(975, 978, 4), kernel_regularizer=l2(0.01)),
    MaxPooling2D((2, 2)),
    
    Conv2D(18, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    MaxPooling2D((2, 2)),
    
    Conv2D(12, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    MaxPooling2D((2, 2)),
    
    Conv2D(8, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(1, activation='linear')
])


# In[4]:


model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse','mae'])


# # Model Training

# In[15]:


model.fit(x_train, y_train, epochs = 3, batch_size = 16)


# # Save Model

# In[16]:


predictions = model.predict(x_test)


# In[33]:


predictions


# In[31]:


# Initialize lists to store actual and predicted ratings
actual_ratings = []
predicted_ratings = []

# Iterate through each image in the test array
for i in range(len(x_train)):
    # Display the image
    plt.imshow(Image.fromarray(x_train[i]))
    plt.title(f"Actual Rating: {y_train[i]}, Predicted Rating: {predictions[i]}")
    plt.axis('off')
    plt.show()
    
    # Append actual and predicted ratings to lists
    actual_ratings.append(y_train[i])
    predicted_ratings.append(predictions[i])

# Convert lists to numpy arrays for calculation
actual_ratings = np.array(actual_ratings)
predicted_ratings = np.array(predicted_ratings)

# Calculate Mean Squared Error (MSE) and Mean Absolute Error (MAE)
mse = mean_squared_error(actual_ratings, predicted_ratings)
mae = mean_absolute_error(actual_ratings, predicted_ratings)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")


# In[26]:


from keras.models import load_model
model.save("face_rating_prediction_model_1.28.h5")


# In[27]:


model = load_model('face_rating_prediction_model_1.28.h5')


# In[29]:


model


# In[ ]:




