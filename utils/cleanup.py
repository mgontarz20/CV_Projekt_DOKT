import os
import shutil
def clearEmptyDirectories(path):
    folders = next(os.walk(path))[1]
    print("Cleaning up results directory\n")
    i = 0
    for folder in folders:
        try:
            files = next(os.walk(os.path.join(path, folder,'model')))[2]
            if not files:
                print(f"REMOVING PATH: {os.path.join(path, folder)}")
                shutil.rmtree(os.path.join(path, folder))
                i += 1
        except StopIteration:
            files = next(os.walk(os.path.join(path, folder)))[2]
            if not files:
                print(f"REMOVING PATH: {os.path.join(path, folder)}")
                shutil.rmtree(os.path.join(path, folder))
                i += 1

    if not i:
        print("No folders removed!")
    else:
        print(f"Removed {i} folders!")


