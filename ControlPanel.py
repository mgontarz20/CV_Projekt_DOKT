import os
from DNN import DNN
from CNN1_FPAUDL import CNN1
from datetime import datetime
from keras.losses import MeanSquaredError
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


def SSIMMetric(self, y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def get_params(arch:str):
    input_size = 64
    num_filters = 60
    kernel_size = 3
    activation = 'relu'
    kernel_regularizer = 'l2'
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    output_type = 'bg'
    if arch.upper() == 'DNN':
        model_name = f"DNN_CV_{input_size}x{input_size}_{timestamp}_{output_type}"
    elif arch.upper() == 'CNN1':
        model_name = f"CNN1_CV_{input_size}x{input_size}_{timestamp}_{output_type}"
    else:
        model_name = f"Other_CV_{input_size}x{input_size}_{timestamp}_{output_type}"
    return input_size, num_filters,kernel_size,activation,kernel_regularizer,model_name, output_type


def getTrainingParams():
    initial_lr = 0.0001
    #loss = custom_mse_SSIM_Loss(y_true, y_pred)

    def SSIMMetric(y_true, y_pred):
        return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
    metrics = ['accuracy', SSIMMetric]
    test_split = 0.2
    random_state = 3213
    loss_name = 'MSE'
    batch_size = 32
    epoch_limit = 600

    return initial_lr, loss_name, metrics, test_split, random_state, batch_size, epoch_limit


def load_data(img_dir, test_size, random_state, output_type):

    X = np.load(os.path.join(img_dir, 'mixed_patches.npz'))['arr_0']
    y = np.load(os.path.join(img_dir, f'{output_type}_patches.npz'))['arr_0']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
    print(X_train.dtype, X_valid.dtype, y_train.dtype, y_valid.dtype)
    del X
    del y

    return X_train, X_valid, y_train, y_valid


def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    cnn_dir = os.getcwd()
    img_dir = os.path.join(cnn_dir, 'data_prep', 'data_test')
    dataset_folder = 'dataset_1_test'
    dataset_dir = os.path.join(cnn_dir, dataset_folder)
    print('[INFO] Getting params')
    arch = 'CNN1'
    input_size, num_filters, kernel_size, activation, kernel_regularizer, model_name, output_type = get_params(arch)
    initial_lr, loss_name, metrics, test_split, random_state,  batch_size, epoch_limit = getTrainingParams()
    print('[INFO] Loading data')
    X_train, X_valid, y_train, y_valid = load_data(img_dir, test_split, random_state, output_type)


    print('[INFO] Data loaded, Defining and initializing model')
    #DNN_40x40 = DNN(input_size, num_filters,kernel_size,activation,kernel_regularizer,model_name, cnn_dir)

    # model = DNN_40x40.getModel()
    # DNN_40x40.compile(initial_lr,metrics, loss_name,summarize=True, tofile=True)
    # DNN_40x40.train(X_train, y_train, X_valid, y_valid, batch_size, epoch_limit, verbose=1)
    # DNN_40x40.save_log(test_split, random_state, dataset_dir, trained=True)
    #
    # DNN_40x40.plot_results(trained=True)
    #

    CNN1Faudl = CNN1(input_size, num_filters,kernel_size,activation,kernel_regularizer,model_name, cnn_dir)

    model = CNN1Faudl.getModel()
    CNN1Faudl.compile(initial_lr, metrics, loss_name, summarize=True, tofile=True)
    CNN1Faudl.train(X_train, y_train, X_valid, y_valid, batch_size, epoch_limit, verbose=1)
    CNN1Faudl.save_log(test_split, random_state, dataset_dir, trained=False)

    CNN1Faudl.plot_results(trained=False)




if __name__ == '__main__':
    main()