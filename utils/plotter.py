from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt
import numpy as np

def plot_model_data(root, model, name, results):

    plot_model(model, to_file=f"{root}{name}/{name}model_'LR'.jpg",show_layer_names= True, rankdir = 'LR', expand_nested=True, show_shapes=True)
    plot_model(model, to_file=f"{root}{name}/{name}model_'TB'.jpg",show_layer_names= True, rankdir = 'TB', expand_nested=False, show_shapes=True)

    plt.figure(figsize=(10, 10))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()
    plt.savefig(f"{root}{name}/loss_{name}.jpg")
    plt.clf()
    plt.title("Learning curve")
    plt.plot(results.history["accuracy"], label="Accuracy")
    plt.plot(results.history["val_accuracy"], label="Val_Accuracy")
    plt.plot( np.argmax(results.history["val_accuracy"]), np.max(results.history["val_accuracy"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{root}{name}/acc_{name}.jpg")