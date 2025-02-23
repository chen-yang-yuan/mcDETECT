import numpy as np
from scipy.spatial import cKDTree

def make_tree(d1 = None, d2 = None, d3 = None):
    active_dimensions = [dimension for dimension in [d1, d2, d3] if dimension is not None]
    if len(active_dimensions) == 1:
        points = np.c_[active_dimensions[0].ravel()]
    elif len(active_dimensions) == 2:
        points = np.c_[active_dimensions[0].ravel(), active_dimensions[1].ravel()]
    elif len(active_dimensions) == 3:
        points = np.c_[active_dimensions[0].ravel(), active_dimensions[1].ravel(), active_dimensions[2].ravel()]
    return cKDTree(points)