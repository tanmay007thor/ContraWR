# %%
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# %%

# Function to load data from multiple .pkl files
def load_data_from_directory(directory):
    X_data, y_data = [], []
    for file in os.listdir(directory):
        if file.endswith(".pkl"):
            file_path = os.path.join(directory, file)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                X_data.append(data['X'])
                y_data.append(data['y'])
    return np.array(X_data), np.array(y_data)

# %%


# Load data from train and test directories
train_dir = './preprocess/SLEEP_data/cassette_processed/train'
test_dir = './preprocess/SLEEP_data/cassette_processed/test'

X_train, y_train = load_data_from_directory(train_dir)
X_test, y_test = load_data_from_directory(test_dir)

# If the test set is empty, create it from a portion of the training data
if len(X_test) == 0 or len(y_test) == 0:
    print("Test set is empty. Splitting some data from the training set to use as the test set.")
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Normalize the data
X_train = (X_train - np.mean(X_train)) / np.std(X_train)
X_test = (X_test - np.mean(X_test)) / np.std(X_test)

# Create a mapping for class modification
class_mapping = {
    'e': None,      # Class 'e' will be removed
    '1': 'N1',      # Map class '1' to 'N1'
    '2': 'N2',      # Map class '2' to 'N2'
    'W': 'Wake',    # Map class 'W' to 'Wake'
    'R': 'REM',     # Map class 'R' to 'REM'
    '3': 'N3',      # Merge class '3' into 'N3'
    '4': 'N3',      # Merge class '4' into 'N3'
}

# Modify the labels according to the mapping
def modify_labels(y):
    return np.array([class_mapping.get(label, label) for label in y])

y_train_modified = modify_labels(y_train)
y_test_modified = modify_labels(y_test)

 
train_mask = y_train_modified != None  # Mask for removing class 'e'
test_mask = y_test_modified != None     # Mask for removing class 'e'

 
X_train = X_train[train_mask]
y_train_modified = y_train_modified[train_mask]

X_test = X_test[test_mask]
y_test = y_test[test_mask]

 
all_labels_modified = np.concatenate((y_train_modified, y_test_modified))

 
label_encoder = LabelEncoder()
label_encoder.fit(all_labels_modified)  # Fit on the combined modified labels

# Encode the labels
y_train_encoded = label_encoder.transform(y_train_modified)
y_test_encoded = label_encoder.transform(y_test_modified)

 
X_train = X_train.reshape(X_train.shape[0], -1)  # Flatten the input for Neural Network
X_test = X_test.reshape(X_test.shape[0], -1)

 
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train_encoded)

 


# %%
model = keras.Sequential([
    layers.Input(shape=(X_train_resampled.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(len(label_encoder.classes_), activation='softmax') 
])

 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

 
history = model.fit(X_train_resampled, y_train_resampled, epochs=50, batch_size=32, validation_split=0.2)

# %%
# Make predictions
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Print shapes for debugging
print(f'Shape of y_test_encoded: {y_test_encoded.shape}, Shape of y_pred: {y_pred.shape}')
print(f'Shape of X_test: {X_test.shape}, Shape of y_test_modified: {y_test_modified.shape}')

# Calculate accuracy only if shapes match
if y_test_encoded.shape[0] == y_pred.shape[0]:
    accuracy = accuracy_score(y_test_encoded, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
else:
    print("Mismatch in lengths: Cannot calculate accuracy.")

# Plot accuracy and loss over epochs
plt.figure(figsize=(14, 6))

# Line width for plots
line_width = 2.5

# Font sizes for titles, labels, legends
title_fontsize = 18
label_fontsize = 14
legend_fontsize = 12

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=line_width, color='b')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=line_width, color='r')
plt.title('Model Accuracy', fontsize=title_fontsize)
plt.ylabel('Accuracy', fontsize=label_fontsize)
plt.xlabel('Epoch', fontsize=label_fontsize)
plt.legend(loc='upper left', fontsize=legend_fontsize)
plt.grid(True, linestyle='--', linewidth=0.7)
plt.tick_params(axis='both', which='major', labelsize=label_fontsize)

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', linewidth=line_width, color='b')
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=line_width, color='r')
plt.title('Model Loss', fontsize=title_fontsize)
plt.ylabel('Loss', fontsize=label_fontsize)
plt.xlabel('Epoch', fontsize=label_fontsize)
plt.legend(loc='upper left', fontsize=legend_fontsize)
plt.grid(True, linestyle='--', linewidth=0.7)
plt.tick_params(axis='both', which='major', labelsize=label_fontsize)

# Adjust layout
plt.tight_layout()
plt.show()


