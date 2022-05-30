import keras
from keras.models import load_model
import tensorflow as tf
import numpy as np
import os
import shutil
import imageio

def SSIMMetric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 255.0))

def get_model_paths(cnn_paths):

    good_dirs = []
    models_dirs = next(os.walk(cnn_paths))[1]
    for dir in models_dirs:
        files = next(os.walk(os.path.join(cnn_paths, dir)))[2]
        for file in files:
            if file.endswith(".h5"):
                good_dirs.append(dir)
    return good_dirs

def load_imgs(img_dirs):
    mixed = next(os.walk(os.path.join(os.getcwd(), 'data_for_test','patches', 'mixed')))[2]
    imgs = np.empty((len(mixed), 64,64,1))

    for idx, filename in enumerate(mixed):
        imgs[idx] = (np.resize(imageio.v2.imread(os.path.join(img_dirs, filename)), (64,64,1))/255.0).astype('float32')

    return imgs, mixed

def predict(cnn_paths, dirs, imgs, save_path, filenames):
    models = []
    for dir in dirs:
        files = next(os.walk(os.path.join(cnn_paths, dir)))[2]
        for file in files:
            print(file)
            if file.endswith('h5'):

                model = load_model(os.path.join(cnn_paths, dir, file), custom_objects={'SSIMMetric': SSIMMetric}, compile=False)
                os.makedirs(os.path.join(save_path, file.split('.')[0]),exist_ok=True)
                for img, filename in zip(imgs, filenames):
                    try:
                        pred = model.predict(img).astype('float32')[:,:,:,0]
                        print(pred.shape)
                        imageio.imwrite(os.path.join(save_path, file.split('.')[0], f"{filename.split('.')[0]}.tiff"), pred)
                    except FileNotFoundError as err:
                        print(err)
                        pass
        del model
def main():
    root_path = os.path.dirname(os.getcwd())
    cnn_paths = os.path.join(root_path, 'results')
    img_path = os.path.join(root_path,'testings','data_for_test','patches', 'mixed')
    save_path = os.path.join(os.path.join(root_path,'testings', 'data_for_test', 'preds'))
    os.makedirs(save_path,exist_ok=True)
    good_dirs = get_model_paths(cnn_paths)
    imgs, mixed = load_imgs(img_path)
    predict(cnn_paths, good_dirs, imgs, save_path, mixed)






if __name__ == '__main__':
    main()