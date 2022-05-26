from keras import layers
from keras import Sequential
import tensorflow as tf
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import json
from keras.losses import MeanSquaredError
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard,CSVLogger


class DNN:
    def __init__(self, input_size, num_filters, kernel_size, activation, kernel_regularizer, model_name, cnn_dir):
        self.cnn_dir = cnn_dir
        self.model_name = model_name
        os.makedirs(os.path.join(self.cnn_dir, 'results', self.model_name), exist_ok=True)

        self.model_name = model_name
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.kernel_regularizer = kernel_regularizer
        self.input_size = input_size
        self.cnn_dir = cnn_dir


    def getModel(self):
        self.model = Sequential()

        self.model.add(layers.Input((self.input_size, self.input_size,1)))

        self.model.add(layers.Conv2D(filters= self.num_filters, kernel_size=(self.kernel_size, self.kernel_size), padding = 'same', kernel_regularizer=self.kernel_regularizer, strides = 1))
        self.model.add(layers.ReLU())

        for i in range(15):
            self.model.add(layers.Conv2D(filters=self.num_filters, kernel_size=(self.kernel_size, self.kernel_size), padding='same',
                                         kernel_regularizer=self.kernel_regularizer, strides=1))
            self.model.add(layers.BatchNormalization())
            self.model.add(layers.ReLU())

        self.model.add(layers.Conv2D(filters= 1, kernel_size=(self.kernel_size, self.kernel_size), padding = 'same', kernel_regularizer=self.kernel_regularizer, strides = 1))

        return self.model

    def compile(self, initial_lr, loss, metrics, summarize:bool, tofile:bool):
        self.initial_lr = initial_lr
        if loss =='MSE':
            self.loss = MeanSquaredError()
        self.metrics = metrics

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = self.initial_lr, epsilon = 0.0001, beta_1 = 0.95), loss = self.loss, metrics = self.metrics)

        if summarize:
            self.model.summary()
            if tofile:
                with open(f'{self.cnn_dir}/results/{self.model_name}/summary_{self.input_size}x{self.input_size}.txt', 'w+') as f:
                    self.model.summary(print_fn=lambda x: f.write(x + '\n'))

    def train(self, X_train, y_train, X_test, y_test, batch_size, epoch_limit, verbose):
        start = datetime.datetime.now()
        self.batch_size = batch_size
        self.epoch_limit = epoch_limit
        self.callbacks = [
            ReduceLROnPlateau(factor=0.1, patience=20, min_lr=0.00000001, verbose=1),
            ModelCheckpoint(f'{self.cnn_dir}/results/{self.model_name}/{self.model_name}.h5', verbose=1, save_best_only=True),
            CSVLogger(f"{self.cnn_dir}/results/{self.model_name}/{self.model_name}.csv"),
        ]

        self.history = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs = self.epoch_limit, callbacks = self.callbacks, validation_data = (X_test, y_test), verbose = verbose)

        #print(self.history.history)
        stop = datetime.datetime.now()
        self.elapsed_time = str(stop-start)


    def plot_results(self, trained:bool, SSIM_Metric:bool = True):


        plot_model(self.model, to_file=f"{self.cnn_dir}/results/{self.model_name}/{self.model_name}model_'LR'.jpg", show_layer_names=True, rankdir='LR',
                   expand_nested=True, show_shapes=True)
        plot_model(self.model, to_file=f"{self.cnn_dir}/results/{self.model_name}/{self.model_name}model_'TB'.jpg", show_layer_names=True, rankdir='TB',
                   expand_nested=False, show_shapes=True)
        if trained:
            plt.figure(figsize=(10, 10))
            plt.title("Learning curve")
            plt.plot(self.history.history["loss"], label="loss")
            plt.plot(self.history.history["val_loss"], label="val_loss")
            plt.plot(np.argmin(self.history.history["val_loss"]), np.min(self.history.history["val_loss"]), marker="x", color="r",
                     label="best model")
            plt.xlabel("Epochs")
            plt.ylabel("log_loss")
            plt.legend()
            plt.savefig(f"{self.cnn_dir}/results/{self.model_name}/loss_{self.model_name}.jpg")
            plt.clf()
            ii = 0
            for metric in self.metrics:
                ii+=1
                if not isinstance(metric, str):
                    metric = f'SSIMMetric'
                plt.title(f"{metric} curve")
                plt.plot(self.history.history[metric], label=metric)
                plt.plot(self.history.history[f"val_{metric}"], label=f"Val_{metric}")
                plt.plot(np.argmax(self.history.history[f"val_{metric}"]), np.max(self.history.history[f"val_{metric}"]), marker="x",
                         color="r", label="best model")
                plt.xlabel("Epochs")
                plt.ylabel("Accuracy")
                plt.legend()
                plt.savefig(f"{self.cnn_dir}/results/{self.model_name}/{metric}_{self.model_name}.jpg")

    def save_log(self, test_split, random_state, dataset_dir, trained:bool):

        log_dict = {
            'dir':self.cnn_dir,
            'name':self.model_name,
            'ksize':self.kernel_size,
            'reg':self.kernel_regularizer,
            'activation':self.activation,
            'dataset_dir':dataset_dir,
            'loss':'MSE',
            'init_lr':self.initial_lr,
            'metrics':'acc_SSIM'
        }

        if trained:
            train_dict = {
                'batch_size':self.batch_size,
                'epoch_limit':self.epoch_limit,
                'train_time':self.elapsed_time,
                'best_val_loss':float(np.min(self.history.history["val_loss"])),
                'for_loss':int(np.argmin(self.history.history["val_loss"])),
                'test_split':test_split,
                'random_state':random_state
            }
            log_dict.update(train_dict)


        json.dump(log_dict, open(f"{self.cnn_dir}/results/{self.model_name}/log.json", 'w'), indent=4)

