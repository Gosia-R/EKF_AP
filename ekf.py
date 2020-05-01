from math import cos, sin, atan2
from typing import Optional

import numpy as np

from measurement import Measurement, Sensor
from tools import normalize_angle


class EKF:
    def __init__(self):

        self.is_initialized = False
        self.ts_prev = 0.0

        # Placeholders
        self._x: Optional[np.ndarray] = np.array([0., 0., 0., 0.])  # State vector [position_x, position_y, velocity_x, velocity_y]
        self._P: Optional[np.ndarray] = None  # Covariance matrix placeholder
        self._F: Optional[np.ndarray] = None  # Model transition matrix
        self._Q: Optional[np.ndarray] = None  # Covariance matrix of process noise

        self._radar_noise_range_std = 0.3
        self._radar_noise_azimuth_std = 0.03
        self._radar_noise_range_rate_std = 0.3

        self._lidar_noise_pos_x_std = 0.15
        self._lidar_noise_pos_y_std = 0.15

        self.noise_acc_x_std = 3  # Model noise acceleration in x axis
        self.noise_acc_y_std = 3  # Model noise acceleration in y axis

        # Noise matrices per sensor type
        self._R_radar = np.array(
            [
                [self._radar_noise_range_std ** 2, 0, 0],
                [0, self._radar_noise_azimuth_std ** 2, 0],
                [0, 0, self._radar_noise_range_rate_std ** 2]
            ]
        )

        self._R_lidar = np.array(
            [
                [self._lidar_noise_pos_x_std ** 2, 0],
                [0, self._lidar_noise_pos_x_std ** 2]
            ]
        )

    def process_measurement(self, measurement: Measurement):
        """
        Process given measurement and returns current estimation of car state
        :param measurement: Measurement structure with sensor type, data and measurement timestamp in microseconds
        """

        if not self.is_initialized:
            # Run initialization procedure
            self.initialize(measurement)

        else:
            # Run predict and update procedure
            dt = (measurement.ts_us - self.ts_prev) / 1000000.0
            self.ts_prev = measurement.ts_us

            # Run prediction if elapsed time
            if dt > 0.01:
                self.predict(dt)

            self.update(measurement)

    def initialize(self, measurement):

        self.ts_prev = measurement.ts_us

        """
        PLACE YOUR CODE HERE
        TODO: Implement initialization procedure of Kalman Filter
        - initialize state vector self._x
        - initialize covariance matrix self._P - look for good estimates of variance of given state variables
        """
        if measurement.sensor == Sensor.RADAR:
            self._x = np.array([measurement.data[0]*np.cos(measurement.data[1]),measurement.data[0]*np.sin(measurement.data[1]), 0, 0])
        else:
            self._x = np.array([measurement.data[0], measurement.data[1], 0, 0])
        self._P = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1000, 0],
                           [0, 0, 0, 1000]])

        self.is_initialized = True

    def predict(self, dt):
        """
        Predicts state and covariance in delta time
        :param dt: time elapsed from previous measurement update in seconds
        """

        """
        PLACE YOUR CODE HERE
        TODO: Implement prediction of state and covariance due to elapsed time
        - 
        - Define self._F state-transition model on dt parameter
        - Define self._Q covariance matrix of process noise (how not measured acceleration impacts covariance of state
        - With use of self._F predict next state and save it to self._x
        - With use of self._F and self._Q update state covariance matrix
        """
        self._F = np.array ([[1, 0, dt, 0],
                             [0, 1, 0, dt],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
        self._Q = np.array ([[dt*dt*dt*dt/4*self.noise_acc_x_std, 0, dt*dt*dt/2*self.noise_acc_x_std, 0],
               [0, dt*dt*dt*dt/4*self.noise_acc_y_std, 0, dt*dt*dt/2*self.noise_acc_y_std],
               [dt*dt*dt/2*self.noise_acc_x_std, 0, dt*dt*self.noise_acc_x_std, 0],
               [0, dt*dt*dt/2*self.noise_acc_y_std, 0,dt*dt*self.noise_acc_y_std]])
        self._x = self._x @ self._F
        self._P = self._F @ self._P @ np.transpose(self._F) + self._Q

        pass

    def update(self, measurement):

        if measurement.sensor is Sensor.RADAR:
            self.update_radar(measurement)

        elif measurement.sensor is Sensor.LIDAR:
            self.update_lidar(measurement)

    def update_radar(self, measurement):
        """
        Radar measurement update procedure
        :param measurement: Radar measurement
        """
        # Rewrite measurement vector
        range = measurement.data[0]
        azimuth = measurement.data[1]
        range_rate = measurement.data[2]

        """
        PLACE YOUR CODE HERE
        TODO: Implement radar update (correction) step
        - Calculate predicted measurement based on current state self._x
        - Calculate error between real measurement and this predicted from state
        - Be careful with angle as it rolls around (error between -179 deg and 179 deg is only 2 deg - 
          use normalize_angle function from tools package)
        - Use measurement noise matrix self._R_radar
        - As observation model is non-linear, H matrix (observation model) will have to be linearization (Jacobian 
          matrix) of it in given state. To ease it, use _calc_jacobian_for_radar function where this is calculated for 
          you
        - Follow Extended Kalman filter update procedure - update self._x and self._P
        """

        self.R_ = self._R_radar
        h = self._calc_jacobian_for_radar()
        px = self._x[0]
        py = self._x[1]
        vx = self._x[2]
        vy = self._x[3]

        #hj = np.array([np.sqrt(vx*vx+vy*vy), np.arctan(py/px), (vx+)])
        rho = np.sqrt(px * px + py * py)
        phi = 0
        rho_dot = 0

        if np.fabs(rho) > 0.0001:
            phi = atan2(py, px)
            rho_dot = ((px * vx + py * vy) / rho)
        z= np.array([range, azimuth, range_rate])
        z_pred = np.array([rho, phi, rho_dot])
        y = z - z_pred
        while y[1] < -np.pi:
            y[1] += 2 * np.pi
        while y[1] > np.pi:
            y[1] -= 2 * np.pi
        ht = h.transpose()
        PHt = self._P @ ht
        S = h @ PHt + self._R_radar
        Si = np.linalg.inv(S)
        K = PHt @ Si
        self._x = self._x + (K @ y)
        I = np.eye(4, 4)
        self._P = (I - K @ h) @ self._P

    pass

    def update_lidar(self, measurement):
        """
        Lidar measurement update procedure
        :param measurement: Lidar measurement
        :return:
        """
        # Set measurement
        pos_x = measurement.data[0]
        pos_y = measurement.data[1]

        """
        PLACE YOUR CODE HERE
        TODO: Implement lidar update (correction) step
        - Get predicted measurement straight forward from state self._x
        - Calculate error between real measurement and this predicted from state
        - Use measurement noise matrix self._R_lidar
        - As lidar observation model is linear, this step is simply Kalman filter update
        - Follow Extended Kalman filter update procedure - update self._x and self._P
        """
        self.R_ = self._R_lidar
        h = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        z_pred = h @ self._x
        y = measurement.data - z_pred

        while y[1] < -np.pi:
            y[1] += 2 * np.pi
        while y[1] > np.pi:
            y[1] -= 2 * np.pi
        ht = h.transpose()
        PHt = self._P @ ht
        S = h @ PHt + self._R_lidar
        Si = np.linalg.inv(S)
        K = PHt @ Si
        self._x = self._x + (K @ y)
        I = np.eye(4, 4)
        self._P = (I - K @ h) @ self._P

        pass

    def get_estimation(self):
        """
        Returns current estimation of vehicle state as a numpy array
        :return: Returns array of values from state [ position_x, position_y, velocity_x, velocity_y ]
        """
        return [self._x[0], self._x[1], self._x[2], self._x[3]]

    def _calc_jacobian_for_radar(self):
        """
        Calculates Jacobian matrix (for observation model) for radar measurement
        :return: Jacobian matrix
        """
        # Initialize Jacobian
        Hj = np.zeros((3, 4))

        # Rewrite state
        px = self._x[0]
        py = self._x[1]
        vx = self._x[2]
        vy = self._x[3]

        # Check if there will be division by zero in Jacobian
        d1 = px * px + py * py
        d2 = np.sqrt(d1)
        d3 = d1 * d2

        if abs(d1) > 0.0001:
            Hj[0, 0] = px / d2
            Hj[0, 1] = py / d2
            Hj[0, 2] = 0
            Hj[0, 3] = 0

            Hj[1, 0] = -py / d1
            Hj[1, 1] = px / d1
            Hj[1, 2] = 0
            Hj[1, 3] = 0

            Hj[2, 0] = py * (vx * py - vy * px) / d3
            Hj[2, 1] = px * (vy * px - vx * py) / d3
            Hj[2, 2] = px / d2
            Hj[2, 3] = py / d2

        else:
            raise ValueError('Division by zero in Jacobian calculation')

        return Hj






