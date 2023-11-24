#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical


# In[2]:


# Function to load and preprocess images
def load_and_preprocess_data(folder_path, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".TIF"):
            image_path = os.path.join(folder_path, filename)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = resize_image(img, target_size=(100, 100))
            images.append(img)
            labels.append(label)
    return images, labels

# Function to resize image
def resize_image(img, target_size):
    img = cv2.resize(img, target_size)
    return img

# Load and preprocess data for each class
all_images = []
all_labels = []

# Specify the root folder where your class folders are located
root_folder = "class_folder"

# Iterate through numerical class names
for class_label in [0, 2, 4, 6, 9]:
    class_folder = os.path.join(root_folder, str(class_label))
    images, labels = load_and_preprocess_data(class_folder, class_label)
    
    all_images.extend(images)
    all_labels.extend(labels)

# Convert lists to NumPy arrays
all_images = np.array(all_images)
all_labels = np.array(all_labels)


# In[3]:


unique_labels = np.unique(all_labels)
print("Unique Labels:", unique_labels)


# In[4]:


# Example: Map [0, 2, 4, 6, 9] to [0, 1, 2, 3, 4]
label_mapping = {0: 0, 2: 1, 4: 2, 6: 3, 9: 4}
all_labels = np.array([label_mapping[label] for label in all_labels])


# In[5]:


num_classes = len(np.unique(all_labels))
all_labels = to_categorical(all_labels, num_classes=num_classes)


# In[6]:


# Normalize pixel values to the range [0, 1]
all_images = all_images / 255.0


# In[7]:


print("Number of images:", len(all_images))
print("Number of labels:", len(all_labels))


# In[8]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)


# In[21]:


from tensorflow.keras.optimizers import Adam

# Define learning rate and dropout rate
learning_rate = 0.001
dropout_rate = 0.55


# Define the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(dropout_rate))  # Adding dropout layer
model.add(layers.Dense(5, activation='softmax'))

# Define the optimizer with the specified learning rate
optimizer = Adam(learning_rate=learning_rate)

# Compile the model with the custom optimizer
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")


# In[22]:


from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# In[23]:


# Make predictions on the test set
y_pred_probabilities = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probabilities, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Compute metrics
precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
accuracy = accuracy_score(y_true_classes, y_pred_classes)
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

# Print metrics
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")


# In[24]:


# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=[0, 2, 4, 6, 9], yticklabels=[0, 2, 4, 6, 9])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




