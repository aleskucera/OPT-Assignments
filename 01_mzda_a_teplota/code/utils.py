import yaml
import matplotlib.pyplot as plt
import numpy as np


def load_data(path: str) -> dict:
    with open(path, 'r') as f:
        data = np.array(list(map(lambda x: list(map(float, x.split(' '))), f.readlines())))
        data = np.array(data, dtype=np.float32)
    return data


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        plt.rc(key, **value)
    return config


if __name__ == "__main__":
    load_data("/home/ales/OneDrive/School/5. semestr/OPT/OPT-Assignments/01_mzda_a_teplota/data/mzdy.txt")
