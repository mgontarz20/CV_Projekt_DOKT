import numpy as np
import imageio
import os
import json
import random
from tqdm import tqdm
import to_npy
import sys
def img_splitter(iter, img_mixed, img_fringes, img_bg, mixed_name, patch_dict, fringe_patch_dir, mixed_patch_dir, bg_patch_dir, window_size):

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

            patch_mixed_name = f"{j}_{i}_{mixed_name}"
            patch_bg_name = f"{j}_{i}_{'_'.join(mixed_name.split('_')[:-1])}.png"
            patch_fringe_name = f"{j}_{i}_{patch_mixed_name.split('_')[-1]}"

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
    img_dir = os.path.join(os.getcwd(), 'data_test')
    mixed_dir = os.path.join(img_dir, 'mixed')
    fringes_dir = os.path.join(img_dir, 'fringes')
    bg_dir = os.path.join(img_dir, 'bg')
    fringe_patch_dir = os.path.join(img_dir, 'patches', 'fringes')
    mixed_patch_dir = os.path.join(img_dir, 'patches', 'mixed')
    bg_patch_dir = os.path.join(img_dir, 'patches', 'bg')
    os.makedirs(fringe_patch_dir, exist_ok=True)
    os.makedirs(mixed_patch_dir, exist_ok=True)
    os.makedirs(bg_patch_dir, exist_ok=True)
    patch_dict = {}
    img_dict = {}
    #mixed = random.sample(next(os.walk(mixed_dir))[2],50)
    mixed = next(os.walk(mixed_dir))[2]
    fringes = [file.split('_')[-1] for file in mixed]
    bg = [f"{'_'.join(file.split('_')[:-1])}.jpg" for file in mixed]
    # print(mixed)
    # print(fringes)
    # print(len(mixed))
    iter = 0
    for mixed_name, fringes_name, bg_name in tqdm(zip(mixed, fringes, bg), desc="PREPARING PATCHES: "):
        iter +=1
        img_mixed = imageio.v2.imread(os.path.join(mixed_dir,mixed_name))
        img_fringes = imageio.v2.imread(os.path.join(fringes_dir,fringes_name))
        img_bg = imageio.v2.imread(os.path.join(bg_dir,bg_name))
        patch_dict_single, img_dict_single = img_splitter(iter, img_mixed, img_fringes, img_bg, mixed_name, patch_dict, fringe_patch_dir, mixed_patch_dir,bg_patch_dir, 512)

        patch_dict.update(patch_dict_single)
        img_dict.update(img_dict_single)

        del img_mixed, img_fringes, patch_dict_single

    json.dump(patch_dict, open(f"{img_dir}/image_coords.json", 'w'), indent=4)
    json.dump(img_dict, open(f"{img_dir}/image_pairs.json", 'w'), indent=4)




if __name__ == '__main__':
    size = sys.argv[1]
    print(size)
    main()
    to_npy.main(int(size))
