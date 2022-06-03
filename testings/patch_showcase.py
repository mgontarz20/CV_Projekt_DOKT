import matplotlib.pyplot as plt
import numpy as np
import os
from random import sample
import imageio
from mpl_toolkits.axes_grid import make_axes_locatable

def choose_filenames(pred_dir):
    model_dirs = next(os.walk(pred_dir))[1]
    chosen_paths = []
    for dir in model_dirs:
        if 'showcase' in dir:
            continue
        files = next(os.walk(os.path.join(pred_dir, dir)))[2]
        try:
            chosen = sample(files, 10)
        except ValueError:
            chosen = files

        for file in chosen:
            chosen_paths.append(os.path.join(pred_dir, dir, file))

    return chosen_paths


def load_imgs(pred_dir, patch_dir, file_path, size, type):
    mixed_name = f"{file_path.split('/')[-1].split('.')[0]}.png"

    fringe_name = f"{size}_{file_path.split('/')[-1].split('_')[1]}_{file_path.split('/')[-1].split('_')[2]}_{file_path.split('/')[-1].split('_')[-1].split('.')[0]}.png"
    pred = imageio.v2.imread(file_path)
    patch_mixed = imageio.v2.imread(os.path.join(patch_dir, 'mixed', mixed_name).replace(r'\\', '/'))
    patch_fringes = imageio.v2.imread(os.path.join(patch_dir, type, fringe_name).replace(r'\\', '/'))

    return pred, patch_mixed, patch_fringes
def main():
    #data_dir = "/home/mgontarz/PycharmProjects/CV_Projekt_DOKT/testings/data_for_test"
    data_dir = 'c:/Users/mgont/PycharmProjects/CV_Projekt_DOKT/testings/data_for_test'
    folders = next(os.walk(data_dir))[1]


    #models_dir = '/home/mgontarz/PycharmProjects/CV_Projekt_DOKT/results'
    models_dir = 'c:/Users/mgont/PycharmProjects/CV_Projekt_DOKT/results'
    img_sizes = set([int(file.split("_")[2].split('x')[0]) for file in next(os.walk(models_dir))[1]])
    print(img_sizes)

    for size in img_sizes:
        pred_dir = os.path.join(data_dir, f'preds_{size}')
        patch_dir = os.path.join(data_dir, f'patches_{size}')
        os.makedirs(os.path.join(pred_dir, f'showcase_{size}'), exist_ok=True)


        chosen_paths = choose_filenames(pred_dir)
        for path in chosen_paths:
            path = path.replace('\\', '/')
            if f'{path.split("/")[-2]}' not in next(os.walk(f'{"/".join(path.split("/")[:-2])}/showcase_{size}/'))[1] or len(next(os.walk(f'{"/".join(path.split("/")[:-2])}/showcase_{size}/{path.split("/")[-2]}'))[2]) <= 10:
                fig, ax = plt.subplots(1, 3, figsize=(25, 15))
                os.makedirs(f'{"/".join(path.split("/")[:-2])}/showcase_{size}/{path.split("/")[-2]}',exist_ok=True)
                if len(path.split("_")) == 6:
                    type = path.split("_")[-1]
                else:
                    type = 'fringes'
                pred, patch_mixed, patch_fringes = load_imgs(pred_dir, patch_dir, path, size, type)
                img0 = ax[0].imshow(patch_mixed)
                img1 = ax[1].imshow((pred))
                img2 = ax[2].imshow(patch_fringes)
                ax[0].set_title('Patch zmieszany\n(prążki + tło)')
                ax[1].set_title('Predykcja sieci')
                ax[2].set_title('Same prążki\njak to ma wyglądać.')
                divider1 = make_axes_locatable(ax[0])
                cax1 = divider1.append_axes("right", size="5%", pad=0.05)
                cbar1 = fig.colorbar(img0, ax=ax[0], cax=cax1, extend='both')
                divider2 = make_axes_locatable(ax[1])
                cax2 = divider2.append_axes("right", size="5%", pad=0.05)
                cbar2 = fig.colorbar(img1, ax=ax[1], cax=cax2, extend='both')
                divider3 = make_axes_locatable(ax[2])
                cax3 = divider3.append_axes("right", size="5%", pad=0.05)
                cbar3 = fig.colorbar(img2, ax=ax[2], cax=cax3, extend='both')
                #print(f'{"/".join(path.split("/")[:-2])}/showcase/{path.split("/")[-2]}/{path.split("/")[-1].split(".")[0]}.png')
                plt.savefig(f'{"/".join(path.split("/")[:-2])}/showcase_{size}/{path.split("/")[-2]}/{path.split("/")[-1].split(".")[0]}.png',dpi=200)
                plt.close()
            else: print(f'Already done showcase for {path.split("/")[-2]}')
        #plt.show()

if __name__ == '__main__':
    main()
