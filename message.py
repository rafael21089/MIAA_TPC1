import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import time
import warnings

# Load labels
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = open(labels_path).read().splitlines()
imagenet_labels = imagenet_labels[1:]

# Load calibration set
calibration_list_of_files = "ILSVRC2012_CalibrationSet.txt"
calibration_images_folder = "./images/ILSVRC2012_img_cal"
calibration_groundtruth = "ILSVRC2012_calibration_ground_truth.txt"

cal_list_of_files = open(calibration_list_of_files, 'r')
cal_files = [file.rstrip('\n') for file in cal_list_of_files]

cal_gt_file = open(calibration_groundtruth, 'r')
cal_gt = [int(y.rstrip('\n')) for y in cal_gt_file]

# Models to evaluate
models = {
    'VGG19': tf.keras.applications.vgg19.VGG19(weights='imagenet'),
    'ResNet50': tf.keras.applications.resnet50.ResNet50(weights='imagenet'),
    'InceptionV3': tf.keras.applications.inception_v3.InceptionV3(weights='imagenet'),
    'DenseNet121': tf.keras.applications.densenet.DenseNet121(weights='imagenet'),
    'MobileNetV2': tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet')
}

preprocess_input = {
    'VGG19': tf.keras.applications.vgg19.preprocess_input,
    'ResNet50': tf.keras.applications.resnet50.preprocess_input,
    'InceptionV3': tf.keras.applications.inception_v3.preprocess_input,
    'DenseNet121': tf.keras.applications.densenet.preprocess_input,
    'MobileNetV2': tf.keras.applications.mobilenet_v2.preprocess_input
}

target_size = {
    'VGG19': (224, 224),
    'ResNet50': (224, 224),
    'InceptionV3': (299, 299),
    'DenseNet121': (224, 224),
    'MobileNetV2': (224, 224)
}

results = {}

for model_name, model in models.items():
    print(f"Evaluating {model_name}...")
    start_time = time.time()
    cal_pred = []

    for i, file in enumerate(cal_files):
        image_path = os.path.join(calibration_images_folder, file)

        image = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size[model_name])
        x = tf.keras.preprocessing.image.img_to_array(image)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input[model_name](x)

        result = model.predict(x, verbose=0)
        predicted_class = np.argmax(result[0], axis=-1)
        cal_pred.append(predicted_class)

    end_time = time.time()
    inference_time = end_time - start_time
    fps = len(cal_files) / inference_time

    precision = precision_score(cal_gt, cal_pred, average='macro', zero_division=0)
    recall = recall_score(cal_gt, cal_pred, average='macro', zero_division=0)
    f1 = f1_score(cal_gt, cal_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(cal_gt, cal_pred)
    cm_absolute = confusion_matrix(cal_gt, cal_pred)
    cm_normalized = confusion_matrix(cal_gt, cal_pred, normalize='true')

    results[model_name] = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'confusion_matrix_absolute': cm_absolute,
        'confusion_matrix_normalized': cm_normalized,
        'fps': fps
    }

    print(f"{model_name} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, Accuracy: {accuracy:.4f}, FPS: {fps:.2f}")

# Display results
for model_name, metrics in results.items():
    print(f"\nMetrics for {model_name}:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Inference FPS: {metrics['fps']:.2f}")
    print("Confusion Matrix (Absolute):")
    print(metrics['confusion_matrix_absolute'])
    print("Confusion Matrix (Normalized):")
    print(metrics['confusion_matrix_normalized'])
