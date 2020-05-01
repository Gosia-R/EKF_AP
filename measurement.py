from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Sensor(IntEnum):
    LIDAR = 0
    RADAR = 1
    INVALID = 2


@dataclass
class Measurement:
    sensor: Sensor
    data: np.ndarray
    ts_us: float
