# wind_flag = False
# Implement a controller

import numpy as np
import csv
import os
import time

# Imports the class from the InnerLoopController.py file and initialses it.
from InnerLoopController import InnerLoopController
inner_controller = InnerLoopController()

timestamp = time.strftime("%Y%m%d_%H%M%S")
fileName = f"flight_data_{timestamp}.csv"

# If the file doesn't exist yet (e.g., first run), write the headers
if not os.path.exists(fileName):
    with open(fileName, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'current_x', 'current_y', 'current_z', 'current_yaw', 
            'target_x', 'target_y', 'target_z', 'target_yaw'
        ])

def controller(state, target_pos, dt, wind_enabled=False):
    # state format: [position_x (m), position_y (m), position_z (m), roll (radians), pitch (radians), yaw (radians)]
    # target_pos format: (x (m), y (m), z (m), yaw (radians))
    # dt: time step (s)
    # wind_enabled: boolean flag to indicate if wind disturbance should be considered in the control algorithm

    current_pos = np.array(state[0:3])
    current_yaw = state[5]

    # Append the current states and targets to the CSV file every time step
    with open(fileName, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            current_pos[0], current_pos[1], current_pos[2], current_yaw,
            target_pos[0], target_pos[1], target_pos[2], target_pos[3]
        ])

    # # return velocity command format: (velocity_x_setpoint (m/s), velocity_y_setpoint (m/s), velocity_z_setpoint (m/s), yaw_rate_setpoint (radians/s))
    output = (0, 0, 0, 0)

    return output