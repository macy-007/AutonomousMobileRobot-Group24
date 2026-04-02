# wind_flag = False
# Implement a controller

import numpy as np

# Imports the class from the InnerLoopController.py file and initialses it.
from InnerLoopController import InnerLoopController
inner_controller = InnerLoopController()

def controller(state, target_pos, dt, wind_enabled=False):
    # state format: [position_x (m), position_y (m), position_z (m), roll (radians), pitch (radians), yaw (radians)]
    # target_pos format: (x (m), y (m), z (m), yaw (radians))
    # dt: time step (s)
    # wind_enabled: boolean flag to indicate if wind disturbance should be considered in the control algorithm

    current_pos = np.array(state[0:3])
    current_yaw = state[5]
    # # return velocity command format: (velocity_x_setpoint (m/s), velocity_y_setpoint (m/s), velocity_z_setpoint (m/s), yaw_rate_setpoint (radians/s))
    output = (0, 0, 0, 0)

    return output