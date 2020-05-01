from math import pi

import numpy as np


def rmse(estimation, groundtruth):

    estimation = np.asarray(estimation)
    groundtruth = np.asarray(groundtruth)

    if estimation.shape != groundtruth.shape:
        raise ValueError('Not correct dimensions of estimation and groundtruth')

    err = (estimation - groundtruth) ** 2

    err = np.mean(err, axis=0)

    err = np.sqrt(err)

    return err


def normalize_angle(angle: float):
    return (angle + pi) % (2 * pi) - pi
