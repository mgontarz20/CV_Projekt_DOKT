import numpy as np
import os
import json
import imageio
from tqdm import tqdm
import sys

def main(size):

    """This function within this script serves as a way of efficiently storing image data in numpy arrays,
    which are saved on the disk. This type of data is then more efficiently stored and loaded for training."""


    #Directory generation
    img_dir = os.path.join(os.getcwd(), 'data_test')
    mixed_dir = os.path.join(img_dir, 'mixed')
    fringes_dir = os.path.join(img_dir, 'fringes')
    fringe_patch_dir = os.path.join(img_dir, 'patches', 'fringes')
    mixed_patch_dir = os.path.join(img_dir, 'patches', 'mixed')
    bg_patch_dir = os.path.join(img_dir, 'patches', 'bg')
    f = open(os.path.join(img_dir, f'image_pairs_{size}.json'))
    pair_dict = json.load(f)

    #NPArray initializadion
    mixed = np.zeros((len(pair_dict.items()), size, size, 1), dtype='float32')
    fringes = np.zeros((len(pair_dict.items()), size, size, 1), dtype='float32')
    bgs = np.zeros((len(pair_dict.items()), size, size, 1), dtype='float32')

    #Image names reading
    bg_list = [f"{value[1]}" for idx,value in enumerate(pair_dict.values())]
    fringe_list = [f"{value[0]}" for idx,value in enumerate(pair_dict.values())]
    mixed_list = [f"{key}" for idx,key in enumerate(pair_dict.keys())]

    #Iteratively loading images and saving them into arrays, which are then saved to .npz
    for i in tqdm(range(len(fringes)), desc="IMPORTING FRINGES: "):
        fringe_patch_img = imageio.v2.imread(os.path.join(fringe_patch_dir, fringe_list[i])).astype('float32')
        fringe_patch_img = np.resize(fringe_patch_img, (size,size,1))
        fringes[i] = fringe_patch_img/255.0
    np.savez_compressed(os.path.join(img_dir, f'fringes_patches_{size}_norm.npz'), fringes)
    del fringes
    for j in tqdm(range(len(mixed)), desc="IMPORTING MIXED: "):
        mixed_patch_img = imageio.v2.imread(os.path.join(mixed_patch_dir, mixed_list[j])).astype('float32')
        mixed_patch_img = np.resize(mixed_patch_img, (size,size,1))
        mixed[j] = mixed_patch_img/255.0
    np.savez_compressed(os.path.join(img_dir, f'mixed_patches_{size}_norm.npz'), mixed)
    del mixed
    for k in tqdm(range(len(bgs)), desc="IMPORTING BGs: "):
        bg_patch_img = imageio.v2.imread(os.path.join(bg_patch_dir, bg_list[k])).astype('float32')
        bg_patch_img = np.resize(bg_patch_img, (size,size,1))
        bgs[k] = bg_patch_img/255.0
    np.savez_compressed(os.path.join(img_dir, f'bg_patches_{size}_norm.npz'), bgs)
    del bgs



    

if __name__ == '__main__':
    size = sys.argv[1]
    main(int(size))