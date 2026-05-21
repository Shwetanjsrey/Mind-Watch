from pathlib import Path
import os


def create_directories(*dirs):
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)


def get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"