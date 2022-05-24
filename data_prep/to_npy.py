import numpy as np
import os
import json
import imageio
from tqdm import tqdm

def main():
    img_dir = os.path.join(os.getcwd(), 'data_test')
    mixed_dir = os.path.join(img_dir, 'mixed')
    fringes_dir = os.path.join(img_dir, 'fringes')
    fringe_patch_dir = os.path.join(img_dir, 'patches', 'fringes')
    mixed_patch_dir = os.path.join(img_dir, 'patches', 'mixed')
    f = open(os.path.join(img_dir, 'image_pairs.json'))
    pair_dict = json.load(f)
    mixed = np.zeros((len(pair_dict.keys()), 40, 40, 1), dtype=np.uint8)
    fringes = np.zeros((len(pair_dict.values()), 40, 40, 1), dtype=np.uint8)
    fringe_list = list(pair_dict.values())
    mixed_list = list(pair_dict.keys())

    for i in tqdm(range(len(fringes)), desc="IMPORTING WRAPPED: "):
        fringe_patch_img = imageio.v2.imread(os.path.join(fringe_patch_dir, fringe_list[i]))
        fringe_patch_img = np.resize(fringe_patch_img, (40,40,1))
        fringes[i] = fringe_patch_img

    for j in tqdm(range(len(mixed)), desc="IMPORTING WRAPPED: "):
        mixed_patch_img = imageio.v2.imread(os.path.join(mixed_patch_dir, mixed_list[j]))
        mixed_patch_img = np.resize(mixed_patch_img, (40,40,1))
        mixed[j] = mixed_patch_img

    print(fringes.shape)
    print(mixed.shape)
    np.savez_compressed(os.path.join(img_dir, 'fringes_patches.npz'), fringes)
    np.savez_compressed(os.path.join(img_dir, 'mixed_patches.npz'), mixed)



    

if __name__ == '__main__':
    main()