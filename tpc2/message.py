import os
import numpy as np
import glob
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Download and extract the dataset
_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
zip_file = tf.keras.utils.get_file(origin=_URL, fname="flower_photos.tgz", extract=True)
base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']
dataset_split_percentage = 0.6 # 60% train, 20% val, 20% test
epochs = 20     # 40
batch_size = 32
IMG_SHAPE = (299, 299) # InceptionV3 expects 299x299 images

# Stratified split of the dataset
for cl in classes:
    img_path = os.path.join(base_dir, cl)
    images = glob.glob(img_path + '/*.jpg')
    print("{}: {} Images".format(cl, len(images)))
    num_train = int(len(images) * dataset_split_percentage)
    num_val = int((len(images) - num_train) / 2)
    train, val, test = images[:num_train], images[num_train:num_train + num_val], images[num_train + num_val:]

    # Move images to respective directories
    for split, split_name in zip([train, val, test], ['train', 'val', 'test']):
        split_dir = os.path.join(base_dir, split_name, cl)
        os.makedirs(split_dir, exist_ok=True)
        for img in split:
            destination = os.path.join(split_dir, os.path.basename(img))
            if not os.path.exists(destination):
                shutil.move(img, destination)

# Data Generators with augmentation for training and rescaling for validation/testing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,  # Custom data augmentation: Random flip (horizontal)
    zoom_range=0.2,        # Custom data augmentation: Random zoom
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(base_dir, 'train'),
    target_size=IMG_SHAPE,
    batch_size=batch_size,
    class_mode='sparse'
)
val_generator = val_test_datagen.flow_from_directory(
    os.path.join(base_dir, 'val'),
    target_size=IMG_SHAPE,
    batch_size=batch_size,
    class_mode='sparse'
)
test_generator = val_test_datagen.flow_from_directory(
    os.path.join(base_dir, 'test'),
    target_size=IMG_SHAPE,
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=False
)

# Load pre-trained InceptionV3 model and fine-tune
base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False # Freeze the base model

# Add custom top layers for our specific classification task
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

# Compile the model with an optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # Explicitly choosing: Optimizer and Learning rate

## optimizer=tf.keras.optimizers.Adamax(learning_rate=0.002)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy']) # Explicitly choosing: Loss function

# Define callbacks for training
log_dir = os.path.join("../logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir) # Tensorboard callback
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('../best_model.h5', save_best_only=True) # Checkpoint callback
early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True) # Early stopping callback

# Train the model
history = model.fit(
    train_generator,
    epochs=epochs, # Explicitly choosing: Number of epochs
    validation_data=val_generator,
    callbacks=[tensorboard_callback, checkpoint_callback, early_stopping_callback], # Using callbacks
    batch_size=batch_size # Explicitly choosing: Batch size
)

# Evaluate the model on the test set
model.load_weights('best_model.h5')
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc}")

# Make predictions on the test set
y_true = test_generator.classes
y_pred = np.argmax(model.predict(test_generator), axis=-1)

# Calculate evaluation metrics
conf_matrix = confusion_matrix(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
accuracy = accuracy_score(y_true, y_pred)

print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plot Precision, Recall, F-score, Accuracy
metrics = {'Precision': precision, 'Recall': recall, 'F1 Score': f1, 'Accuracy': accuracy}
plt.figure(figsize=(8, 5))
plt.bar(metrics.keys(), metrics.values())
plt.ylim(0, 1)
plt.title('Evaluation Metrics')
plt.show()

# Plot training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
