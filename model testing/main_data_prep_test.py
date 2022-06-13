import numpy as np
import imageio
import os
import json
import random
from tqdm import tqdm
import cv2 as cv

def img_splitter(iter, img_mixed, img_fringes, img_bg, mixed_name, patch_dict, fringe_patch_dir, mixed_patch_dir, bg_patch_dir, window_size):
    """Function responsible for splitting the image into patches and saving the patches coordinates in the original image."""

    rows, cols = img_mixed.shape

    if img_mixed.shape != img_fringes.shape:
        raise ValueError("Fringe and mixed image dimension mismatch!")
    j = i = idx_X = idx_Y = 0
    patch_dict[mixed_name] = {}
    img_dict = {}
    while (rows - j) >= window_size :
        while (cols-i) >= window_size:
            img_mixed_patch = img_mixed[j:j+window_size, i:i+window_size]
            img_fringes_patch = img_fringes[j:j+window_size, i:i+window_size]
            img_bg_patch = img_bg[j:j+window_size, i:i+window_size]

            patch_mixed_name = f"{window_size}_{j}_{i}_{mixed_name}"
            patch_bg_name = f"{window_size}_{j}_{i}_{'_'.join(mixed_name.split('_')[:-1])}.png"
            patch_fringe_name = f"{window_size}_{j}_{i}_{patch_mixed_name.split('_')[-1]}"

            patch_dict[mixed_name][patch_mixed_name] = f"{j}_{i}"
            i += (window_size-10)

            img_dict[patch_mixed_name] = (patch_fringe_name, patch_bg_name)
            if img_fringes_patch.shape == (window_size,window_size):
                imageio.imwrite(os.path.join(fringe_patch_dir, f"{patch_fringe_name}"), img_fringes_patch)
                imageio.imwrite(os.path.join(bg_patch_dir, f"{patch_bg_name}"), img_bg_patch)

                imageio.imwrite(os.path.join(mixed_patch_dir, f"{patch_mixed_name}"), img_mixed_patch)
        i = 0
        j += (window_size-10)

    return patch_dict, img_dict



def main():
    """This script serves as a control panel for data preparation for testing. The data prepared with this script has not been seen by the model during training.
            Here a random portion of an initial dataset created by Oskar is loaded, and then
            a background image, a mixed image and the fringe pattern are cut into the same patches of size 96x96
            with overlap. All the data saved for training is saved in a temp folder called 'data_test', which is not included in the repo."""

    #Directory generation
    img_dir = os.path.join(os.getcwd(), 'data_for_test')
    mixed_dir = os.path.join(img_dir, 'mixed')
    fringes_dir = os.path.join(img_dir, 'fringes')
    #models_dir = '/home/mgontarz/PycharmProjects/CV_Projekt_DOKT/results'
    models_dir = r"c:/Users/mgont/PycharmProjects/CV_Projekt_DOKT/results"
    bg_dir = os.path.join(img_dir, 'bg')


    patch_dict = {}
    img_dict = {}

    #During the project we experimented with various patch sizes, this generated data for every size tested.
    img_sizes = set([int(file.split("_")[2].split('x')[0]) for file in next(os.walk(models_dir))[1]])

    #Random sample chosen from original dataset
    #mixed = random.sample(next(os.walk(mixed_dir))[2], 50)
    mixed = next(os.walk(mixed_dir))[2]
    fringes = [file.split('_')[-1] for file in mixed]
    bg = [f"{'_'.join(file.split('_')[:-1])}.png" for file in mixed]



    iter = 0
    for mixed_name, fringes_name, bg_name in tqdm(zip(mixed, fringes, bg), desc="PREPARING PATCHES: "):
        iter += 1

        for size in img_sizes:
            if size != 96:
                img_mixed = imageio.v2.imread(os.path.join(mixed_dir, mixed_name))
                img_fringes = cv.resize(imageio.v2.imread(os.path.join(fringes_dir, fringes_name)),
                                        dsize=(img_mixed.shape[1], img_mixed.shape[0]), interpolation=cv.INTER_AREA)
                img_bg = imageio.v2.imread(os.path.join(bg_dir, bg_name))
            else:
                img_mixed = np.pad(imageio.v2.imread(os.path.join(mixed_dir, mixed_name)),
                                   pad_width=[(16, 16), (16, 16)], mode='edge')
                img_fringes = np.pad(imageio.v2.imread(os.path.join(fringes_dir, fringes_name)),
                                     pad_width=[(16, 16), (16, 16)], mode='edge')
                img_bg = np.pad(imageio.v2.imread(os.path.join(bg_dir, bg_name)), pad_width=[(16, 16), (16, 16)],
                                mode='edge')

            fringe_patch_dir = os.path.join(img_dir, f'patches_{size}', 'fringes')
            mixed_patch_dir = os.path.join(img_dir, f'patches_{size}', 'mixed')
            bg_patch_dir = os.path.join(img_dir, f'patches_{size}', 'bg')

            os.makedirs(f'{fringe_patch_dir}', exist_ok=True)
            os.makedirs(f'{mixed_patch_dir}', exist_ok=True)

            os.makedirs(f'{bg_patch_dir}', exist_ok=True)
            patch_dict_single, img_dict_single = img_splitter(iter, img_mixed, img_fringes, img_bg, mixed_name, patch_dict,
                                                          fringe_patch_dir, mixed_patch_dir, bg_patch_dir, size)

            patch_dict.update(patch_dict_single)
            img_dict.update(img_dict_single)

        del img_mixed, img_fringes, patch_dict_single
    #Json files saving, as a reminder of image coordinates

    json.dump(patch_dict, open(f"{img_dir}/image_coords.json", 'w'), indent=4)
    json.dump(img_dict, open(f"{img_dir}/image_pairs.json", 'w'), indent=4)


if __name__ == '__main__':
    main()
