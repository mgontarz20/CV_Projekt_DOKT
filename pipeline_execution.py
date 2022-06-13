import numpy as np
import os
import imageio
from keras.models import load_model

def img_splitter(img_mixed, mixed_name, mixed_patch_dir, window_size):

    """Splits tested image as in the dataset generation."""
    rows, cols = img_mixed.shape

    j = i = 0
    while (rows - j) >= window_size :
        while (cols-i) >= window_size:
            img_mixed_patch = img_mixed[j:j+window_size, i:i+window_size]


            patch_mixed_name = f"{j}_{i}_{mixed_name}"


            i += (window_size-32)


            imageio.imwrite(os.path.join(mixed_patch_dir, f"{patch_mixed_name.split('.')[0]}.tiff"), img_mixed_patch/255.0)

        i = 0
        j += (window_size-32)


def stitcher(patch_out_dir, results_dir, input_name):

    """Stitching images back to 512x512 from patches."""
    patch_outs = next(os.walk(patch_out_dir))[2]
    to_stitch_on = np.zeros((512,512), dtype = 'float32')
    for patch_out in patch_outs:
        h = w = 64
        x = int(patch_out.split("_")[0])
        y = int(patch_out.split("_")[1])
        patch_out_img = imageio.v2.imread(os.path.join(patch_out_dir, patch_out)).astype('float32')
        to_stitch_on[x:x+h, y:y+w] =  patch_out_img[16:80, 16:80]

    imageio.imwrite(os.path.join(results_dir, f'{input_name}.tiff'),to_stitch_on)
    imageio.imwrite(os.path.join(results_dir, input_name),to_stitch_on.astype(np.uint8))

def main():
    """Function used to execute the pipeline. An fringe image with background needs to be placed
    into the execute directory, and the script cuts it, predicts on patches and stitches it back to generate result."""


    maindir = os.getcwd()
    img_dir = os.path.join(maindir,'execute')
    patch_in_dir = os.path.join(img_dir, 'temp','patch_input')
    patch_out_dir = os.path.join(img_dir, 'temp','patch_out')
    result_dir = os.path.join(img_dir, 'result')
    os.makedirs(patch_out_dir, exist_ok=True)
    os.makedirs(patch_in_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    input_name = next(os.walk(img_dir))[2][0]

    input_img = np.pad(imageio.v2.imread(os.path.join(img_dir, input_name)), pad_width=[(16,16), (16,16)], mode='edge').astype('float32')

    img_splitter(input_img, input_name, patch_in_dir, 96)

    model = load_model(os.path.join(maindir, 'good_model', 'UNet_5_96x96_2022-06-03_15-22_fringes_MSE', 'UNet_5_96x96_2022-06-03_15-22_fringes_MSE.h5'), compile = False)

    patch_inputs = next(os.walk(patch_in_dir))[2]
    for patch_name in patch_inputs:
        patch = imageio.v2.imread(os.path.join(patch_in_dir, patch_name))
        pred = model.predict(np.array([patch]), verbose = 0)[0]

        imageio.imwrite(os.path.join(patch_out_dir, patch_name), pred*255.0)

    stitcher(patch_out_dir, result_dir, input_name)

if __name__ == '__main__':
    main()

