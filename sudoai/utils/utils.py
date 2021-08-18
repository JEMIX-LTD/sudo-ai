import os
from pathlib import Path


def datapath(path: str):
    """get the absolute path in data folder

    Args:
          path (str): path from data folder

    Returns:
          str: absolute path in data folder
    """
    _ROOT = Path(__file__).parent.parent
    return os.path.join(_ROOT, 'data', path)
