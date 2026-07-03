"""Load recorded Lidar samples from text files."""

import numpy as np


def read_lidar_data(path):
    """
    Load Lidar frames from a text file.

    Frames are separated by a line containing ``---``. Leading ``---`` is allowed.

    Args:
        path: Path to the text file (comma-separated x, y, z per line).

    Returns:
        List of ``numpy.ndarray``, each of shape (N, 3).
    """
    with open(path, encoding="utf-8") as file:
        raw = file.read()

    array_strings = raw.strip().lstrip("---").split("\n---\n")
    return [np.loadtxt(block.splitlines(), delimiter=",") for block in array_strings]
