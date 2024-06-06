import os
import time
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

from matplotlib import pyplot as plt

calibration_list_of_files = "ILSVRC2012_CalibrationSet.txt"
calibration_images_folder = "./images/ILSVRC2012_img_cal"
calibration_groundtruth = "ILSVRC2012_calibration_ground_truth.txt"

labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = open(labels_path).read().splitlines()
imagenet_labels = imagenet_labels[1:]


cal_list_of_files = open(calibration_list_of_files, 'r')
cal_files = list()
for file in cal_list_of_files:
    cal_files.append(file.rstrip('\n'))

cal_gt_file = open(calibration_groundtruth, 'r')
cal_gt = []
for y in cal_gt_file:
    cal_gt.append(int(y.rstrip('\n')))

models = {
    'vgg19': tf.keras.applications.vgg19.VGG19(weights='imagenet'),
    'densenet': tf.keras.applications.densenet.DenseNet121(weights='imagenet'),
    'resnet50': tf.keras.applications.resnet50.ResNet50(weights='imagenet'),
    'mobilenet_v2': tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet'),
    'efficientnet': tf.keras.applications.EfficientNetB0(weights='imagenet')
}

preprocess_input = {
    'vgg19': tf.keras.applications.vgg19.preprocess_input,
    'densenet': tf.keras.applications.densenet.preprocess_input,
    'resnet50': tf.keras.applications.resnet50.preprocess_input,
    'mobilenet_v2': tf.keras.applications.mobilenet_v2.preprocess_input,
    'efficientnet': tf.keras.applications.efficientnet.preprocess_input

}

IMAGE_RES = 224

img_size = {
    'vgg19': (IMAGE_RES, IMAGE_RES),
    'resnet50': (IMAGE_RES, IMAGE_RES),
    'efficientnet': (IMAGE_RES, IMAGE_RES),
    'densenet': (IMAGE_RES, IMAGE_RES),
    'mobilenet_v2': (IMAGE_RES, IMAGE_RES)
}

cal_pred = []
results = {}

for model_name, model in models.items():
    print(f"Evaluating {model_name}...")
    start_time = time.time()

    # Reset cal_pred for each model evaluation
    cal_pred = []

    for i, file in enumerate(cal_files):
        image_path = os.path.join(calibration_images_folder, file)

        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMAGE_RES, IMAGE_RES))
        x = tf.keras.preprocessing.image.img_to_array(image)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input[model_name](x)  # Use preprocessing function based on the model

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
