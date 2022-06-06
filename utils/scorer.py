from CNNs import UNetResNet_5lvl_Expanded, UNetResNet_5lvl_AvgPool
from CNNs.Models import UNetResNet_5lvl
import numpy as np
import os
import tqdm
from tqdm import tqdm
from keras.losses import MeanSquaredError
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras.layers import Input
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard,CSVLogger,LambdaCallback
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from tensordash.tensordash import Tensordash
import json


print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_gpu_available(cuda_only=True))

date = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
pathtoDataSet = "D:/Datasets"
dataset = "dataset_9_Combined_to_pred_256x256_10-11-2021_11-24-53"
type1 = "resc_wrpd"
type2 = "resc"


path_to_inputs = os.path.join(pathtoDataSet,dataset,type1).replace("\\", "/")
path_to_outputs = os.path.join(pathtoDataSet,dataset,type2).replace("\\", "/")
print(f"Input path: {path_to_inputs}")
print(f"Output path: {path_to_outputs}" )


inputs = next(os.walk(path_to_inputs))[2]
outputs = next(os.walk(path_to_outputs))[2]

inputs.sort()
outputs.sort()

print(inputs)
print(len(inputs))
print(outputs)
print(len(outputs))



X = np.zeros((len(inputs), 256, 256, 1), dtype=np.float32)
y = np.zeros((len(outputs), 256, 256, 1), dtype=np.float32)
print(X.shape, y.shape)


for i in tqdm(range(len(inputs)), desc="IMPORTING IMAGES: "):

    wrapped = img_to_array(load_img(os.path.join(path_to_inputs, inputs[i]).replace('\\','/'), color_mode="grayscale"))
    wrapped = resize(wrapped, (256,256,1), mode = 'constant', preserve_range = True)
    # Load unwrapped images (outputs)
    unwrapped = img_to_array(load_img(os.path.join(path_to_outputs, outputs[i]).replace('\\','/'), color_mode="grayscale"))
    unwrapped = resize(unwrapped, (256,256,1), mode = 'constant', preserve_range = True)

    X[i] = wrapped
    y[i] = unwrapped


print(X.dtype)
print(y.dtype)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=30)
print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

model = keras.models.load_model("C:/Users/Michał/PycharmProjects/BadaniaMchtr/CNNs/Results/UNetResNet5lvl_resc_wrpd_10-11-2021_23-52-21/model/UNetResNet5lvl_resc_wrpd_10-11-2021_23-52-21.h5")

score, acc = model.evaluate(X_train, y_train,
                            batch_size=2)
print('Train score:', score)
print('Train accuracy:', acc)


score, acc = model.evaluate(X_valid, y_valid,
                            batch_size=2)
print('Test score:', score)
print('Test accuracy:', acc)