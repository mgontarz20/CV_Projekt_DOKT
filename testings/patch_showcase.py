import matplotlib.pyplot as plt
import numpy as np
import os
from random import sample
import imageio

def choose_filenames(pred_dir):
    model_dirs = next(os.walk(pred_dir))[1]
    chosen_paths = []
    for dir in model_dirs:
        if 'showcase' in dir:
            continue
        files = next(os.walk(os.path.join(pred_dir, dir)))[2]
        chosen = sample(files, 5)
        for file in chosen:
            chosen_paths.append(os.path.join(pred_dir, dir, file))

    return chosen_paths


def load_imgs(pred_dir, patch_dir, file_path):
    mixed_name = f"{file_path.split('/')[-1].split('.')[0]}.png"
    fringe_name = f"{file_path.split('/')[-1].split('_')[0]}_{file_path.split('/')[-1].split('_')[1]}_{file_path.split('/')[-1].split('_')[-1].split('.')[0]}.png"
    pred = imageio.v2.imread(file_path)
    patch_mixed = imageio.v2.imread(os.path.join(patch_dir, 'mixed', mixed_name))
    patch_fringes = imageio.v2.imread(os.path.join(patch_dir, 'fringes', fringe_name))

    return pred, patch_mixed, patch_fringes
def main():
    pred_dir = "/home/mgontarz/PycharmProjects/CV_Projekt_DOKT/testings/data_for_test/preds"
    patch_dir = '/home/mgontarz/PycharmProjects/CV_Projekt_DOKT/testings/data_for_test/patches'
    chosen_paths = choose_filenames(pred_dir)




    for path in chosen_paths:
        fig, ax = plt.subplots(1, 3)
        os.makedirs(f'{"/".join(path.split("/")[:-2])}/showcase/{path.split("/")[-2]}',exist_ok=True)
        pred, patch_mixed, patch_fringes = load_imgs(pred_dir, patch_dir, path)
        ax[0].imshow(patch_mixed)
        ax[1].imshow(pred)
        ax[2].imshow(patch_fringes)
        ax[0].set_title('Patch zmieszany\n(prążki + tło)')
        ax[1].set_title('Predykcja sieci')
        ax[2].set_title('Same prążki\njak to ma wyglądać.')
        plt.savefig(f'{"/".join(path.split("/")[:-2])}/showcase/{path.split("/")[-2]}/{path.split("/")[-1].split(".")[0]}.png',dpi=200)
    #plt.show()

if __name__ == '__main__':
    main()
