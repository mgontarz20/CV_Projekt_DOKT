import os
import keras.layers
import UNetResNet_5lvl
from DNN import DNN
from CNN1_FPAUDL import CNN1
from datetime import datetime
from keras.losses import MeanSquaredError
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint

def SSIMMetric(self, y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def get_params(arch:str, loss_name:str):
    """Model parameter definition."""
    input_size = 96
    num_filters = 32
    kernel_size = 3
    activation = 'relu'
    kernel_regularizer = 'l1'
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    output_type = 'fringes'

    layers = 15
    if arch.upper() == 'DNN':
        model_name = f"DNN_CV_{input_size}x{input_size}_faces_{timestamp}_{output_type}_sc_{loss_name}_{layers}"
    elif arch.upper() == 'CNN1':
        model_name = f"CNN1_CV_{input_size}x{input_size}_faces_{timestamp}_{output_type}_{loss_name}"
    else:
        model_name = f"UNet_CV_{input_size}x{input_size}_faces_{timestamp}_{output_type}_{loss_name}"
    return input_size, num_filters,kernel_size,activation,kernel_regularizer,model_name, output_type, layers


def getTrainingParams():

    """Definition of training parameters."""
    initial_lr = 0.0001
    #loss = custom_mse_SSIM_Loss(y_true, y_pred)

    def SSIMMetric(y_true, y_pred):
        return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
    metrics = ['accuracy', SSIMMetric]
    test_split = 0.2
    random_state = 1561
    loss_name = 'MSE'
    batch_size = 8
    epoch_limit = 600
    norm = 'norm'

    return initial_lr, loss_name, metrics, test_split, random_state, batch_size, epoch_limit, norm


def load_data(img_dir, test_size, random_state, output_type, input_size, norm):
    """Data loading"""
    X = np.load(os.path.join(img_dir, f'mixed_patches_{input_size}_{norm}.npz'))['arr_0']
    y = np.load(os.path.join(img_dir, f'{output_type}_patches_{input_size}_{norm}.npz'))['arr_0']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
    print(X_train.dtype, X_valid.dtype, y_train.dtype, y_valid.dtype)
    del X
    del y

    return X_train, X_valid, y_train, y_valid


def main():

    """The main control panel used for model training. Here the data is loaded, split into
    test/train splits and then the model is trained on it. The model is trained on GPU"""

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    cnn_dir = os.getcwd()
    img_dir = os.path.join(cnn_dir, 'data_prep', 'data_test')
    dataset_folder = 'dataset_1_test'
    dataset_dir = os.path.join(cnn_dir, dataset_folder)
    print('[INFO] Getting params')
    arch = 'DNN'
    initial_lr, loss_name, metrics, test_split, random_state,  batch_size, epoch_limit, norm = getTrainingParams()
    input_size, num_filters, kernel_size, activation, kernel_regularizer, model_name, output_type, layers = get_params(arch, loss_name)

    print('[INFO] Loading data')
    X_train, X_valid, y_train, y_valid = load_data(img_dir, test_split, random_state, output_type, input_size, norm)


    print('[INFO] Data loaded, Defining and initializing model')
    #CALLBACK DEFINITION
    callbacks = [
        EarlyStopping(patience=40, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=20, min_lr=0.00000001, verbose=1),
        ModelCheckpoint(f'{cnn_dir}/results/{model_name}/{model_name}.h5', verbose=1,
                        save_best_only=True),
        CSVLogger(f"{cnn_dir}/results/{model_name}/{model_name}.csv"),
    ]
    #FRAGMENT USED TO TRAIN INITIAL DNN MODEL
    # DNN_40x40 = DNN(input_size, num_filters,kernel_size,activation,kernel_regularizer,model_name, cnn_dir, layers)
    #
    # model = DNN_40x40.getModel()
    # DNN_40x40.compile(initial_lr,metrics, loss_name,summarize=True, tofile=True)
    # DNN_40x40.train(X_train, y_train, X_valid, y_valid, batch_size, epoch_limit, verbose=1)
    # DNN_40x40.save_log(test_split, random_state, dataset_dir, trained=True)
    #
    # DNN_40x40.plot_results(trained=True)

    #FRAGMENT USED TO TRAIN THE NEXT MODEL
    # CNN1Faudl = CNN1(input_size, num_filters,kernel_size,activation,kernel_regularizer,model_name, cnn_dir, layers)
    #
    # model = CNN1Faudl.getModel()
    # CNN1Faudl.compile(initial_lr, metrics, loss_name, summarize=True, tofile=True)
    # CNN1Faudl.train(X_train, y_train, X_valid, y_valid, batch_size, epoch_limit, verbose=1)
    # CNN1Faudl.save_log(test_split, random_state, dataset_dir, trained=False)
    #
    # CNN1Faudl.plot_results(trained=False)



    #TRAINING USING THE FINAL U-Net MODEL
    input_img = keras.layers.Input(shape=(96,96,1))
    model = UNetResNet_5lvl.get_unet(input_img, n_filters = num_filters, kernel_size = kernel_size, activation = activation)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-07),
                  loss=MeanSquaredError(), metrics=metrics)
    model.summary()

    results = model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch_limit, callbacks=callbacks,
                        validation_data=(X_valid, y_valid), verbose=1)


if __name__ == '__main__':
    main()