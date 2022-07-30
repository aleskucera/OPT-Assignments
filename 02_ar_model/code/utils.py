import yaml
import matplotlib.pyplot as plt


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        plt.rc(key, **value)
    return config
