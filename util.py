import os
import pickle

def save_model(file_path: str, data: dict):
    with open(file_path, mode='wb') as file:
        pickle.dump(data, file)

def load_model(file_path: str) -> dict:
    ret = None
    if not os.path.exists(file_path):
        return ret
    if not os.path.isfile(file_path):
        return ret

    with open(file_path, mode='rb') as file:
        ret = pickle.load(file)

    return ret