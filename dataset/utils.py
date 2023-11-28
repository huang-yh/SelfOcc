import numpy as np


def get_rm(angle, axis, deg=False):
    if deg:
        angle = np.deg2rad(angle)
    rm = np.eye(3)
    if axis == 'x':
        rm[1, 1] = np.cos(angle)
        rm[2, 2] = np.cos(angle)
        rm[1, 2] = - np.sin(angle)
        rm[2, 1] = np.sin(angle)
    elif axis == 'y':
        rm[0, 0] = np.cos(angle)
        rm[2, 2] = np.cos(angle)
        rm[0, 2] = np.sin(angle)
        rm[2, 0] = - np.sin(angle)
    elif axis == 'z':
        rm[0, 0] = np.cos(angle)
        rm[1, 1] = np.cos(angle)
        rm[0, 1] = - np.sin(angle)
        rm[1, 0] = np.sin(angle)
    return rm
