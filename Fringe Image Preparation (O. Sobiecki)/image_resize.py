import cv2
import os
from tkinter.filedialog import askdirectory



if __name__ == '__main__':
    path = askdirectory(title='Select Folder')  # shows dialog box and return the path, used for
    head, tail = os.path.split(path)
    new_path = os.path.join(head, tail + '_res')    # ścieżka nowego folderu na skalowane zdjęcia
    files = os.listdir(path)    # lista zdjęć w folderze

    if ~os.path.isdir(new_path):    # nowy folder na resizowane zdjęcia
        os.mkdir(new_path)
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            img = cv2.imread(path + '/' + file)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)   # konwersja kanałów
            img_g_res = cv2.resize(img_gray, (512, 512))    # zmiana rozmiaru zdjęcia

            cv2.imwrite(os.path.join(new_path, file), img_g_res)
