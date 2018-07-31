import numpy as np

def read_npz_files(paths):
    files = []
    for path in paths:
        files.append(np.load(path))
    return files

def files_to_dicts(files):
    dicts = []
    for file in files:
        dict = {}
        for i, x in enumerate(file["x_original"]):
            dict[x.tobytes()] = i
        dicts.append(dict)
    return dicts

def get_intersection(data):
    keys = data[0].keys()
    for i in range(1, len(data)):
        keys &= data[i].keys()
    return keys

def get_common_labels(files, data, common, class_names):
    res = []
    for key in common:
        res.append(class_names[files[0]["labels"][data[0][key]]])
    return res

def get_object(files, data, key):
    dicts = []
    for file, curr_data in zip(files, data):
        dicts.append({k: file[k][curr_data[key]] for k in file.keys()})
    return dicts