# Chest-X-Ray-Pneumonia-Detection-using-Deep-Learning
This project demonstrates the application of Transfer Learning with a Convolutional Neural Network (CNN) to classify chest X-ray images, distinguishing between Normal cases and those showing signs of Pneumonia.
Project Goal
The primary goal is to build an efficient deep learning model (MobileNetV2) that can assist in medical image diagnosis by quickly and accurately classifying X-ray images, thereby showcasing the potential of AI in healthcare support.

📦 Dataset
Source: Chest X-Ray Images (Pneumonia) - Kaggle Dataset

Total Images: 5,863

Classes:

NORMAL (Minority Class)

PNEUMONIA (Majority Class - Imbalanced)

The dataset is structured into three main directories: train, test, and val.

🚀 Setup and Installation
1. Prerequisites
You must have Python 3.8+ installed. This project uses a virtual environment for dependency management.

2. Project Setup
Clone or Download this project repository.

Create a Virtual Environment (recommended):

Bash

python -m venv venv
Activate the Environment:

Windows (PowerShell): .\venv\Scripts\Activate.ps1

macOS/Linux: source venv/bin/activate

3. Install Dependencies
Install the necessary libraries using pip:

Bash

pip install tensorflow matplotlib scikit-learn
💾 Data Configuration
Crucial Step: Before running the script, ensure you have downloaded and extracted the Kaggle dataset.

In the pneumonia_detector.py file, verify that the BASE_DIR variable points correctly to the folder containing the train, val, and test directories.

The current setup uses the following path:

Python

BASE_DIR = 'E:/archive (2)/chest_xray'
⚙️ Model Architecture and Training
Transfer Learning Strategy
The project utilizes Transfer Learning to leverage the feature extraction capabilities of a large, pre-trained model.

Base Model: MobileNetV2, pre-trained on the ImageNet dataset.

Freezing: The convolutional layers of MobileNetV2 are frozen (not trained) to preserve learned features.

Custom Head: A new classification head is added, consisting of GlobalAveragePooling2D, a Dense layer (128 units, ReLU), Dropout (0.5), and a final Dense output layer (1 unit, Sigmoid).
Imbalance HandlingDue to the significant imbalance between the PNEUMONIA and NORMAL classes, Class Weighting is used during training. This ensures that the model is heavily penalized for misclassifying the minority NORMAL samples, leading to better Recall (sensitivity).ExecutionRun the main script from your activated virtual environment:Bashpython pneumonia_detector.py
📊 Results and EvaluationThe model will be evaluated on the unseen test dataset. Since the data is imbalanced, metrics like Recall and the F1-score are more important than simple Accuracy.Expected Output Metrics (Example)MetricValueInterpretationTest Accuracy~85-95%Overall correct classifications.Test Recall~90-98%Sensitivity: How well the model identifies true Pneumonia cases. (Crucial metric)Test Precision~80-95%Of all predicted Pneumonia cases, how many were correct.F1-Score~85-95%Balanced measure of Precision and Recall.Confusion Matrix (Example Layout)The confusion matrix provides a detailed breakdown of model performance:Predicted NORMALPredicted PNEUMONIAActual NORMALTrue Negatives (TN)False Positives (FP)Actual PNEUMONIAFalse Negatives (FN)True Positives (TP)Note: Low False Negatives (FN) is highly desired in this context, as FN means a sick patient was classified as healthy.💾 Model SavingThe trained model weights are saved in the project directory after training is complete:pneumonia_detection_mobilenet.h5
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # Added Callbacks
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

# Suppress warnings for cleaner output during training
warnings.filterwarnings("ignore")

# --- 1. Define Paths and Parameters ---
# **CRITICAL: Using the path you provided.**
BASE_DIR = 'E:/archive (2)/chest_xray' 
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')
TEST_DIR = os.path.join(BASE_DIR, 'test')

# Model Parameters
IMG_SIZE = 224 
BATCH_SIZE = 32
EPOCHS = 10 
LEARNING_RATE = 0.0001 

# --- 2. Data Preprocessing and Augmentation ---

# Data Augmentation for the TRAINING set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True, 
    fill_mode='nearest'
)

# Only rescale (normalize) for Validation and Test sets
test_val_datagen = ImageDataGenerator(rescale=1./255)

# Load data from directories
print("Loading Training Data...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

print("Loading Validation Data...")
val_generator = test_val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False 
)

print("Loading Test Data...")
test_generator = test_val_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False 
)

# Get Class Weights to handle Imbalance
normal_count = train_generator.classes.tolist().count(0) 
pneumonia_count = train_generator.classes.tolist().count(1) 
total_samples = normal_count + pneumonia_count

# Calculate class weights
weight_for_normal = (1 / normal_count) * (total_samples / 2.0)
weight_for_pneumonia = (1 / pneumonia_count) * (total_samples / 2.0)

class_weights = {0: weight_for_normal, 1: weight_for_pneumonia}
print(f"Calculated Class Weights (0=NORMAL, 1=PNEUMONIA): {class_weights}")

# --- 3. Model Building (Transfer Learning with MobileNetV2) ---

# Load MobileNetV2 pre-trained on ImageNet
base_model = MobileNetV2(
    weights='imagenet', 
    include_top=False, 
    input_shape=(IMG_SIZE, IMG_SIZE, 3) 
)

# Freeze the base model layers (initial phase)
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers (Head)
x = base_model.output
x = GlobalAveragePooling2D()(x) 
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x) 
predictions = Dense(1, activation='sigmoid')(x) 

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions) 

# --- 4. Callbacks and Compilation ---

# Save the best model based on validation loss
checkpoint_filepath = 'best_pneumonia_model.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss', 
    mode='min',         
    save_best_only=True,
    verbose=1           
)

# Stop training if validation loss plateaus (patience=3 epochs)
early_stopping_callback = EarlyStopping(
    monitor='val_loss', 
    patience=3,         
    restore_best_weights=True, 
    verbose=1
)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy', 
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')] 
)

model.summary()

# --- 5. Train Model (Initial Phase with Frozen Base) ---
print("\n--- Starting Model Training (Phase 1: Frozen Base) ---")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    class_weight=class_weights, 
    callbacks=[model_checkpoint_callback, early_stopping_callback],
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=val_generator.samples // BATCH_SIZE
)

# Load the best weights found during the initial phase
model.load_weights(checkpoint_filepath)
print(f"\nLoaded best weights from: {checkpoint_filepath}")


# --- 5B. Fine-Tuning the Model (Optional Performance Boost) ---
# Unfreeze the top layers of the base model for fine-tuning

print("\n--- Starting Fine-Tuning (Phase 2: Unfrozen Top Layers) ---")

base_model.trainable = True

# Freeze all layers before the fine_tune_at layer
fine_tune_at = 100 
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Re-compile the model with a very low learning rate
FINE_TUNE_LEARNING_RATE = LEARNING_RATE / 10 # 1e-5
model.compile(
    optimizer=Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
    loss='binary_crossentropy', 
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')] 
)

FINE_TUNE_EPOCHS = 5
TOTAL_EPOCHS = EPOCHS + FINE_TUNE_EPOCHS 

history_fine = model.fit(
    train_generator,
    epochs=TOTAL_EPOCHS,
    initial_epoch=history.epoch[-1], # Start from where the previous training left off
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[model_checkpoint_callback, early_stopping_callback],
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=val_generator.samples // BATCH_SIZE
)

# --- 6. Evaluation on Test Set ---
print("\n--- Evaluating Model on Test Set ---")
# Reload the very best weights from the fine-tuning run before final evaluation
model.load_weights(checkpoint_filepath)
results = model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE)

# Get predictions for Confusion Matrix and Classification Report
test_generator.reset()
Y_pred = model.predict(test_generator, steps=test_generator.samples // BATCH_SIZE + 1)
y_pred_classes = (Y_pred > 0.5).astype(int)
y_true = test_generator.classes[:len(y_pred_classes)] 

print(f"\nTest Accuracy: {results[1]*100:.2f}%")
print(f"Test Precision: {results[2]:.4f}")
print(f"Test Recall (Sensitivity): {results[3]:.4f}")

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_true, y_pred_classes)
print(cm)

print("\n--- Classification Report ---")
target_names = ['0: NORMAL', '1: PNEUMONIA']
print(classification_report(y_true, y_pred_classes, target_names=target_names))

# --- 7. Plotting Performance ---
# Combine history objects for plotting
history_combined = {}
for key in history.history.keys():
    history_combined[key] = history.history[key] + history_fine.history[key]

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_combined['accuracy'], label='Training Accuracy')
plt.plot(history_combined['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_combined['loss'], label='Training Loss')
plt.plot(history_combined['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.show()

print(f"\nFinal Best Model saved as: {checkpoint_filepath}")
