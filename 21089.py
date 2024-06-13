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
dataset_split_percentage = [0.6, 0.2, 0.2] # percentages for training, validation, and test sets
epochs = 50
batch_size = 32
IMG_SHAPE = 224
learning_rate = 0.001
checkpoint_dir = "checkpoints"

# Create model name and log directory
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
modelName = "DenseNet121_finetune_" + "E" + str(epochs) + "_LR" + str(learning_rate) + "_" + timestamp
log_dir = os.path.join("logs", "fit", modelName)

# Function to stratify split the dataset
def stratified_split(images, labels, train_size, val_size):
    x_train, x_temp, y_train, y_temp = train_test_split(images, labels, stratify=labels, train_size=train_size)
    val_ratio = val_size / (1 - train_size)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, stratify=y_temp, test_size=val_ratio)
    return x_train, x_val, x_test, y_train, y_val, y_test

# Collect image paths and labels
image_paths = []
image_labels = []

for cl in classes:
    img_path = os.path.join(base_dir, cl)
    images = glob.glob(img_path + '/*.jpg')
    image_paths.extend(images)
    image_labels.extend([cl] * len(images))

# Stratified split
x_train, x_val, x_test, y_train, y_val, y_test = stratified_split(image_paths, image_labels, dataset_split_percentage[0], dataset_split_percentage[1])

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
    img = img / 255.0
    return img

x_test_images = np.array([preprocess_image(img) for img in x_test])
y_test_labels = np.array([classes.index(label) for label in y_test])

# Make predictions
y_pred = np.argmax(model.predict(x_test_images), axis=1)

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

# Plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

plt.figure()
plot_confusion_matrix(cm, classes=classes, title='Confusion Matrix')
plt.show()
