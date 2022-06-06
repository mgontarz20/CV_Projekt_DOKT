def write_cfg(dir, name, op,strip_newline = False, **kwargs):


    f = open(f"{dir}/{name}.txt", op)
    if op == 'w':
        f.write(f"dir:{dir}\nname: {name}\n")
    if strip_newline:
        for key, value in kwargs.items():
            f.write(f"{key}: {value}")
    else:
        for key, value in kwargs.items():
            f.write(f"{key}: {value}\n")

    f.close()