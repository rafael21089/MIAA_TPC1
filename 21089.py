import os
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# Carregar para calibracao
calibration_list_of_files = "ILSVRC2012_CalibrationSet.txt"
calibration_images_folder = "./images/ILSVRC2012_img_cal"
calibration_groundtruth = "ILSVRC2012_calibration_ground_truth.txt"

# Carregar Labels
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

#Modelos Utilizados: vgg19 , densenet , resnet , mobilenet e efficientnet
models = {
    'vgg19': tf.keras.applications.vgg19.VGG19(weights='imagenet'),
    'densenet': tf.keras.applications.densenet.DenseNet121(weights='imagenet'),
    'resnet': tf.keras.applications.resnet.ResNet152(weights='imagenet'),
    'mobilenet': tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet'),
    'efficientnet': tf.keras.applications.efficientnet.EfficientNetB0(weights='imagenet')
}

# Input para Preprocess
preprocess_input = {
    'vgg19': tf.keras.applications.vgg19.preprocess_input,
    'densenet': tf.keras.applications.densenet.preprocess_input,
    'resnet': tf.keras.applications.resnet.preprocess_input,
    'mobilenet': tf.keras.applications.mobilenet_v2.preprocess_input,
    'efficientnet': tf.keras.applications.efficientnet.preprocess_input
}

#Tamanho 224x224
IMAGE_RES = 224

# Tamanho das imagens
img_size = {
    'vgg19': (IMAGE_RES, IMAGE_RES),
    'resnet': (IMAGE_RES, IMAGE_RES),
    'efficientnet': (IMAGE_RES, IMAGE_RES),
    'densenet': (IMAGE_RES, IMAGE_RES),
    'mobilenet': (IMAGE_RES, IMAGE_RES)
}

cal_pred = []
results = {}

for model_name, model in models.items():

    # Medida do tempo inicial para calculo de FPS
    start_time = time.time()
    # Reset de cal_pred para cada avaliacao do modelo
    cal_pred = []

    print("\n--------------------------------------\n")
    print(f"Modelo-> {model_name}...\n")

    for i, file in enumerate(cal_files):
        image_path = os.path.join(calibration_images_folder, file)

        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMAGE_RES, IMAGE_RES))
        x = tf.keras.preprocessing.image.img_to_array(image)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input[model_name](x)

        result = model.predict(x, verbose=0)
        predicted_class = np.argmax(result[0], axis=-1)
        cal_pred.append(predicted_class)

    # Calcule o FPS (FPS é calculado com o número de imagens total dividido pelo tempo total gasto.)
    total_time = time.time() - start_time
    fps = len(cal_files) / total_time


    #Calculos da presicion , recall , f1 score , accuracy score e confusion matrix (Absolute e Normalizada)
    precision = precision_score(cal_gt, cal_pred, average='macro', zero_division=0)
    recall = recall_score(cal_gt, cal_pred, average='macro', zero_division=0)
    fscore = f1_score(cal_gt, cal_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(cal_gt, cal_pred)
    confusion_matrix_absolute = confusion_matrix(cal_gt, cal_pred)
    confusion_matrix_normalized = confusion_matrix(cal_gt, cal_pred, normalize='true')

    # Guardar resultados por modelo
    results[model_name] = {
        'precision': precision,
        'recall': recall,
        'fscore': fscore,
        'accuracy': accuracy,
        'confusion_matrix_absolute': confusion_matrix_absolute,
        'confusion_matrix_normalized': confusion_matrix_normalized,
        'fps': fps
    }

    # Print Resultados
    print(f"{model_name} -> Precision: {precision:.4f} || Recall: {recall:.4f} || F-Score: {fscore:.4f} || Accuracy: {accuracy:.4f} || FPS: {fps:.2f} \n")

    # Print confusion matrices
    print(f"Confusion Matrix (Absoluta) para {model_name}:\n{confusion_matrix_absolute}\n")
    print(f"Confusion Matrix (Normalizada) para {model_name}:\n{confusion_matrix_normalized}")

# Grafico que demonstra os resultados
model_names = list(results.keys())
precisions = [results[model]['precision'] for model in model_names]
recalls = [results[model]['recall'] for model in model_names]
fscores = [results[model]['fscore'] for model in model_names]
accuracies = [results[model]['accuracy'] for model in model_names]
fpss = [results[model]['fps'] for model in model_names]

fig, ax = plt.subplots(3, 2, figsize=(15, 15))

ax[0, 0].bar(model_names, precisions, color='b')
ax[0, 0].set_title('Precision')
ax[0, 0].set_ylabel('Score')

ax[0, 1].bar(model_names, recalls, color='g')
ax[0, 1].set_title('Recall')
ax[0, 1].set_ylabel('Score')

ax[1, 0].bar(model_names, fscores, color='r')
ax[1, 0].set_title('F1-Score')
ax[1, 0].set_ylabel('Score')

ax[1, 1].bar(model_names, accuracies, color='c')
ax[1, 1].set_title('Accuracy')
ax[1, 1].set_ylabel('Score')

ax[2, 0].bar(model_names, fpss, color='m')
ax[2, 0].set_title('Inference Frame Rate (FPS)')
ax[2, 0].set_ylabel('FPS')

ax[2, 1].axis('off')

plt.tight_layout()
plt.show()