import visualkeras
from keras.models import load_model
import tensorflow as tf
import keras
from collections import defaultdict
from keras.layers import Conv2D, BatchNormalization, add, Concatenate, Activation, MaxPooling2D, InputLayer
if __name__ == '__main__':
    def SSIMMetric(self, y_true, y_pred):
        return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


    model = load_model(r'C:\Users\mgont\PycharmProjects\CV_Projekt_DOKT\good_model\UNet_5_96x96_2022-06-03_15-22_fringes_MSE\UNet_5_96x96_2022-06-03_15-22_fringes_MSE.h5', custom_objects={'SSIMMetric':SSIMMetric})
    color_map = defaultdict(dict)
    color_map[Conv2D]['fill'] = 'orange'
    color_map[BatchNormalization]['fill'] = 'gray'
    color_map[add]['fill'] = 'pink'
    color_map[MaxPooling2D]['fill'] = 'red'
    color_map[Concatenate]['fill'] = 'green'
    color_map[Activation]['fill'] = 'teal'
    color_map[InputLayer]['fill'] = 'blue'
    to_ignore = [BatchNormalization,  Activation, InputLayer]
    visualkeras.layered_view(model,legend = True, color_map= color_map, scale_xy=20, draw_volume=False, type_ignore=to_ignore, to_file='model_plot.png').show()