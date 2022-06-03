import keras
from keras.models import load_model
import tensorflow as tf
import numpy as np
import os
import shutil
import imageio
import sys

def SSIMMetric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 255.0))

def custom_mse_SSIM_Loss(self, y_true, y_pred):
    return (1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0)))


def get_model_paths(cnn_paths):

    good_dirs = []
    models_dirs = next(os.walk(cnn_paths))[1]
    for dir in models_dirs:
        files = next(os.walk(os.path.join(cnn_paths, dir)))[2]
        for file in files:
            if file.endswith(".h5"):

                good_dirs.append(dir)
    return good_dirs

def load_imgs(img_dirs,size):
    mixed = next(os.walk(os.path.join(os.getcwd(), 'data_for_test',f'patches_{size}', 'mixed')))[2]
    imgs = np.empty((len(mixed), size,size,1))

    for idx, filename in enumerate(mixed):
        imgs[idx] = (np.resize((imageio.v2.imread(os.path.join(os.getcwd(), 'data_for_test',f'patches_{size}', 'mixed', filename))/255.0).astype('float32'), (size,size,1)))

    return imgs, mixed

def predict(cnn_paths, dirs, imgs, save_path, filenames, size):
    models = []
    preds_path = f'c:/Users/mgont/PycharmProjects/CV_Projekt_DOKT/testings/data_for_test/preds_{size}'
    #preds_path = f'/home/mgontarz/PycharmProjects/CV_Projekt_DOKT/testings/data_for_test/preds_{size}'
    for dir in dirs:
        files = next(os.walk(os.path.join(cnn_paths, dir)))[2]
        if int(dir.split('_')[2].split('x')[0])== size:
            for file in files:
                if file.endswith('h5'):
                    if file.split('.')[0] not in next(os.walk(preds_path))[1]:

                        model = load_model(os.path.join(cnn_paths, dir, file), custom_objects={'SSIMMetric': SSIMMetric, "custom_mse_SSIM_Loss":custom_mse_SSIM_Loss}, compile=False)
                        os.makedirs(os.path.join(save_path, file.split('.')[0]),exist_ok=True)
                        for img, filename in zip(imgs, filenames):
                            try:
                                pred = model.predict(img).astype('float32')[:,:,:,0]
                                print(pred.shape)
                                imageio.imwrite(os.path.join(save_path, file.split('.')[0], f"{filename.split('.')[0]}.tiff"), pred*255.0)
                            except FileNotFoundError as err:
                                print(err)
                                pass
                        del model
                    else: print(f'{file.split(".")[0]} predicted already.')


def main():
    root_path = os.path.dirname(os.getcwd())
    cnn_paths = os.path.join(root_path, 'results')
    img_path = os.path.join(root_path,'testings','data_for_test','patches', 'mixed')
    img_sizes = set([int(file.split("_")[2].split('x')[0]) for file in next(os.walk(cnn_paths))[1]])
    print(img_sizes)
    for size in img_sizes:
        save_path = os.path.join(os.path.join(root_path,'testings', 'data_for_test', f'preds_{size}'))
        os.makedirs(save_path,exist_ok=True)
        good_dirs = get_model_paths(cnn_paths)
        imgs, mixed = load_imgs(img_path, size)
        predict(cnn_paths, good_dirs, imgs, save_path, mixed, size)






if __name__ == '__main__':
    main()