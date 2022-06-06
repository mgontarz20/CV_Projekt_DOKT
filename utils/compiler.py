from CNNs import UNetResNet_6lvl_softmax
from keras.layers import Input
from tensorflow.keras.optimizers import Adam
from keras.losses import MeanSquaredError, SparseCategoricalCrossentropy
from tensorflow.keras.utils import plot_model
import os
from keras.models import Model
import tensorflow as tf
import visualkeras
from PIL import ImageFont
from tensorflow.python.keras.layers import BatchNormalization,Activation, Conv2D, add

name = "WrpdDenoise"

input_img = Input((256, 256, 1), name='img')

model = UNetResNet_6lvl_softmax.get_unet(input_img, num_classes = 9, n_filters = 8, kernel_size=7, activation='relu', kernel_regularizer=None)

def SSIMLoss(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 255.0))

def SIMM_MSE(y_true, y_pred):
  return MeanSquaredError - SSIMLoss(y_true, y_pred)

model.compile(optimizer=Adam(lr = 0.001), loss=SparseCategoricalCrossentropy, metrics = ['mse'])
to_drop = list(range(0,200,1))
to_stay = [1,10,11,14,15,24,25,28,29,38,39,42,43,52,53,56,57,66,67,70,71,80,81, 84, 85, 94, 95, 98,99,100,109,110,113,114,115,124,124,128,129,130,139,140,143,144,145,154,155,158,159,160,169,170,173,174,175,184,185,188,189,190,199]

to_drop = [e for e in to_drop if e not in to_stay]
model.summary()
font = ImageFont.truetype("arial.ttf", 32)

# visualkeras.layered_view(model, legend=True,font=font, spacing =10, index_ignore=to_drop, to_file='3DModel.png')
# visualkeras.layered_view(model, legend=True,font=font,draw_volume=False, spacing =10, index_ignore=to_drop, to_file='2DModel.png')
from collections import defaultdict

color_map = defaultdict(dict)
color_map[Conv2D]['fill'] = 'orange'
color_map[BatchNormalization]['fill'] = 'gray'
color_map[Activation]['fill'] = 'pink'

def model2(input_tensor, num_classes = 9, n_filters = 8, kernel_size=7, activation='relu', kernel_regularizer=None):
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer='he_normal', padding='same', kernel_regularizer=kernel_regularizer)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer='he_normal', padding='same', kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    x = add([input_tensor, x])

    model = Model(inputs = [input_img], outputs = [x])
    return model

model2 = model2(input_img, num_classes = 9, n_filters = 8, kernel_size=7, activation='relu', kernel_regularizer=None)

model2.compile(optimizer=Adam(lr = 0.001), loss=SparseCategoricalCrossentropy, metrics = ['mse'])
visualkeras.layered_view(model2, legend=True,font=font, color_map=color_map).show()
visualkeras.layered_view(model2, legend=True,font=font, draw_volume=False).show()

os.makedirs(f"tests/{name}", exist_ok=True)

plot_model(model, to_file=f"tests/{name}/{name}model_'LR'.jpg",show_layer_names= True, rankdir = 'LR', expand_nested=True, show_shapes=True)
plot_model(model, to_file=f"tests/{name}/{name}model_'TB'.jpg",show_layer_names= True, rankdir = 'TB', expand_nested=False, show_shapes=True)