import numpy as np
import imageio
import os
import json

def img_splitter(img_mixed, img_fringes, mixed_name, patch_dict, fringe_patch_dir, mixed_patch_dir):

    rows, cols = img_mixed.shape

    if img_mixed.shape != img_fringes.shape:
        raise ValueError("Fringe and mixed image dimension mismatch!")
    j = i = idx_X = idx_Y = 0
    patch_dict[mixed_name] = {}
    img_dict = {}
    while (rows - j) >= 40 :
        while (cols-i) >= 40:
            img_mixed_patch = img_mixed[j:j+40, i:i+40]
            img_fringes_patch = img_fringes[j:j+40, i:i+40]

            patch_mixed_name = f"{j}_{i}_{mixed_name}"
            patch_fringe_name = f"{j}_{i}_{patch_mixed_name.split('_')[-1]}"

            patch_dict[mixed_name][patch_mixed_name] = f"{j}_{i}"
            i += 30

            img_dict[patch_mixed_name] = patch_fringe_name
            if img_fringes_patch.shape == (40,40):
                imageio.imwrite(os.path.join(fringe_patch_dir, patch_fringe_name), img_fringes_patch)
                imageio.imwrite(os.path.join(mixed_patch_dir, patch_mixed_name), img_mixed_patch)
        i = 0
        j += 30

    return patch_dict, img_dict



def main():
    img_dir = os.path.join(os.getcwd(), 'data_test')
    mixed_dir = os.path.join(img_dir, 'mixed')
    fringes_dir = os.path.join(img_dir, 'fringes')
    fringe_patch_dir = os.path.join(img_dir, 'patches', 'fringes')
    mixed_patch_dir = os.path.join(img_dir, 'patches', 'mixed')
    os.makedirs(fringe_patch_dir, exist_ok=True)
    os.makedirs(mixed_patch_dir, exist_ok=True)
    patch_dict = {}
    img_dict = {}
    mixed = next(os.walk(mixed_dir))[2]

    fringes = [file.split('_')[-1] for file in mixed]
    # print(mixed)
    # print(fringes)
    # print(len(mixed))

    for mixed_name, fringes_name in zip(mixed, fringes):
        img_mixed = imageio.v2.imread(os.path.join(mixed_dir,mixed_name))
        img_fringes = imageio.v2.imread(os.path.join(fringes_dir,fringes_name))
        patch_dict_single, img_dict_single = img_splitter(img_mixed, img_fringes, mixed_name, patch_dict, fringe_patch_dir, mixed_patch_dir)

        patch_dict.update(patch_dict_single)
        img_dict.update(img_dict_single)

        del img_mixed, img_fringes, patch_dict_single

    json.dump(patch_dict, open(f"{img_dir}/image_coords.json", 'w'), indent=4)
    json.dump(img_dict, open(f"{img_dir}/image_pairs.json", 'w'), indent=4)




if __name__ == '__main__':
    main()
