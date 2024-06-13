import os
import numpy as np
import glob
import shutil
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import itertools

# Download and extract dataset
_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
zip_file = tf.keras.utils.get_file(origin=_URL, fname="flower_photos.tgz", extract=True)
base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

# Define parameters
classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']
num_classes = len(classes)
epochs = 1
batch_size = 64
IMG_SHAPE = 224
learning_rate = 0.001
checkpoint_dir = "model_checkpoint.keras"

# Create model name and log directory
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
modelName = "DenseNet121_finetune_" + "E" + str(epochs) + "_LR" + str(learning_rate) + "_" + timestamp
log_dir = os.path.join("logs", "fit", modelName)

# Collect image paths and labels
image_paths = []
image_labels = []

for cl in classes:
    img_path = os.path.join(base_dir, cl)
    images = glob.glob(img_path + '/*.jpg')
    image_paths.extend(images)
    image_labels.extend([cl] * len(images))

# Stratified split
X_temp, X_test, y_temp, y_test = train_test_split(image_paths, image_labels, test_size=0.4, stratify=image_labels, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Data augmentation and generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.5,
    validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    batch_size=batch_size,
    classes=classes,
    subset='training',
    class_mode='sparse')

validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    batch_size=batch_size,
    classes=classes,
    subset='validation',
    class_mode='sparse')

# Load pre-trained DenseNet121 model
base_model = tf.keras.applications.DenseNet121(input_shape=(IMG_SHAPE, IMG_SHAPE, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Add custom top layers
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir, save_best_only=True)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[tensorboard_callback, checkpoint_callback, early_stopping_callback])

# Preprocess test images
def preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_SHAPE, IMG_SHAPE))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.0 #Normalicao da imagem
    return img

x_test_images = np.array([preprocess_image(img) for img in X_test])
y_test_labels = np.array([classes.index(label) for label in y_test])

# Make predictions
y_pred = np.argmax(model.predict(x_test_images,verbose=0), axis=-1)

# Compute confusion matrix
cm = confusion_matrix(y_test_labels, y_pred)
print("Confusion Matrix")
print(cm)

# Classification report
print("Classification Report")
print(classification_report(y_test_labels, y_pred, target_names=classes))

# Accuracy
accuracy = accuracy_score(y_test_labels, y_pred)
print(f"Accuracy: {accuracy}")



