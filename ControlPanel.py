import os
from DNN import DNN
from datetime import datetime
from keras.losses import MeanSquaredError
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard,CSVLogger
import tensorflow as tf

def get_params():
    input_size = 40
    num_filters = 64
    kernel_size = 3
    activation = 'relu'
    kernel_regularizer = None
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    model_name = f"DNN_CV_{input_size}x{input_size}_{timestamp}"

    return input_size, num_filters,kernel_size,activation,kernel_regularizer,model_name

def getTrainingParams():
    initial_lr = 0.0001
    loss = MeanSquaredError()
    metrics = ['accuracy']
    test_split = 0.2
    random_state = 13412
    loss = 'MSE'
    batch_size = 64
    epoch_limit = 300

    return initial_lr,loss,metrics, test_split, random_state,loss, batch_size, epoch_limit


def load_data(img_dir, test_size, random_state):

    X = np.load(os.path.join(img_dir, 'mixed_patches.npz'))['arr_0']
    y = np.load(os.path.join(img_dir, 'fringes_patches.npz'))['arr_0']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
    del X
    del y

    return X_train, X_valid, y_train, y_valid


def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    cnn_dir = os.getcwd()
    img_dir = os.path.join(cnn_dir, 'data_prep', 'data_test')
    dataset_folder = 'dataset_1_test'
    dataset_dir = os.path.join(cnn_dir, dataset_folder)
    input_size, num_filters, kernel_size, activation, kernel_regularizer, model_name = get_params()
    initial_lr, loss, metrics, test_split, random_state, loss, batch_size, epoch_limit = getTrainingParams()
    X_train, X_valid, y_train, y_valid = load_data(img_dir, test_split, random_state)

    callbacks = [
        #EarlyStopping(patience=stop_patience, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=50, min_lr=0.0000001, verbose=1),
        ModelCheckpoint(f'{cnn_dir}/results/{model_name}/{model_name}.h5', verbose=1, save_best_only=True),
        # Tensordash(ModelName=f"{name}", email='mgontarz15@gmail.com', password='dupadupa'),
        # TensorBoard(log_dir=f"{root}{name}/logs", write_graph=True, write_images= True, update_freq=5),
        CSVLogger(f"{cnn_dir}/results/{model_name}/{model_name}.csv"),
        # json_logging_callback,
    ]

    DNN_40x40 = DNN(input_size, num_filters,kernel_size,activation,kernel_regularizer,model_name, cnn_dir)

    model = DNN_40x40.getModel()
    DNN_40x40.compile(initial_lr,loss,metrics,summarize=True, tofile=True)
    DNN_40x40.train(X_train, y_train, X_valid, y_valid, batch_size, epoch_limit, callbacks, verbose=1)
    DNN_40x40.plot_results(trained=True)
    DNN_40x40.save_log(test_split, random_state, dataset_dir, trained=True)




if __name__ == '__main__':
    main()