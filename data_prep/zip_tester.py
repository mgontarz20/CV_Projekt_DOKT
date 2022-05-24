import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    fringes = np.load(os.path.join(os.getcwd(), 'data_test', 'fringes_patches.npz'))['arr_0']
    mixed = np.load(os.path.join(os.getcwd(), 'data_test', 'mixed_patches.npz'))['arr_0']

    i = 10050

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(fringes[i])
    ax[1].imshow(mixed[i])

    plt.show()