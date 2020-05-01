import numpy as np
from aiohttp import web
import socketio

from ekf import EKF
from measurement import Measurement, Sensor
from tools import rmse


class KalmanFilterProxy:
    def __init__(self,  host='localhost', port=4567):
        # Connection details
        self._host = host
        self._port = port

        self.ekf = EKF()

        # Server callbacks definition
        self._sio = socketio.AsyncServer()
        self._sio.on('connect', self._on_sim_connect)
        self._sio.on('disconnect', self._on_sim_disconnect)
        self._sio.on('telemetry', self._on_telemetry)

        # History stacking
        self.estimations_history = []
        self.groundtruth_history = []

    def run(self):
        # Application creation
        app = web.Application()
        self._sio.attach(app)

        # Run app itself
        web.run_app(app, host=self._host, port=self._port)

    @staticmethod
    def _on_sim_connect(sid, environ):
        print('Connected: ', sid)

    @staticmethod
    def _on_sim_disconnect(sid):
        print('Disconnected: ', sid)

    async def _on_telemetry(self, sid, data):
        # Visualize data received data
        print('Telemetry received: ', data)

        # Communication
        return_msg = dict()
        if data is not None:

            meas_list = str.split(data['sensor_measurement'])

            # Parse data accordingly to sensor type
            if meas_list[0] == "L":
                measurement = Measurement(
                    sensor=Sensor.LIDAR,
                    data=np.array([float(x) for x in meas_list[1:3]]),
                    ts_us=int(meas_list[3])
                )
                groundtruth = [float(x) for x in meas_list[4:8]]

            elif meas_list[0] == "R":

                measurement = Measurement(
                    sensor=Sensor.RADAR,
                    data=np.array([float(x) for x in meas_list[1:4]]),
                    ts_us=int(meas_list[4])
                )
                groundtruth = [float(x) for x in meas_list[5:9]]

            else:
                raise ValueError('Not correct sensor type')

            self.ekf.process_measurement(measurement)
            estimation = self.ekf.get_estimation()

            # Stack history of estimation vs groundtruth
            self.estimations_history.append(estimation)
            self.groundtruth_history.append(groundtruth)

            # Calculate RMSE
            estimation_rmse = rmse(self.estimations_history, self.groundtruth_history)

            # Create message
            return_msg['estimate_x'] = float(estimation[0])
            return_msg['estimate_y'] = float(estimation[1])
            return_msg['rmse_x'] = estimation_rmse[0]
            return_msg['rmse_y'] = estimation_rmse[1]
            return_msg['rmse_vx'] = estimation_rmse[2]
            return_msg['rmse_vy'] = estimation_rmse[3]

            # Message sending
            print("Sending data: ", return_msg)
            await self._sio.emit('estimate_marker', return_msg)

        else:
            # If data is not valid, send dummy response
            await self._sio.emit('manual', {})
