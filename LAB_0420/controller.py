import numpy as np
import time
import os
import csv
from OuterLoopController import OuterLoopController
from InnerLoopController import InnerLoopController


TELLO_VICON_NAME = "Tello_Marker_3"
TELLO_ID = "106"

POSITION_ERROR = 0.5
YAW_ERROR = 0.5
MAX_SPEED = 30

# ==========================================
# 1. INITIALIZATION
# ==========================================
# Initialize the controllers outside the main function so they retain memory
outer_loop = OuterLoopController()
inner_loop = InnerLoopController()

# Data buffer to prevent hard drive I/O lag during high-frequency flight control
flight_data_buffer = []
prev_timestamp = None

# Create a unique CSV file for this flight
timestamp_str = time.strftime("%Y%m%d_%H%M%S")
fileName = f"real_tello_flight_data_{timestamp_str}.csv"

# [MODIFIED]: Expanded to 21 columns to capture all internal PID states for video analysis!
if not os.path.exists(fileName):
    with open(fileName, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'time_ms', 'dt', 
            'cur_x', 'cur_y', 'cur_z', 'cur_yaw', 
            'tgt_x', 'tgt_y', 'tgt_z', 'tgt_yaw',
            'v_des_body_x', 'v_des_body_y',
            'cmd_x', 'cmd_y', 'cmd_z', 'cmd_yaw',
            'out_I_x', 'out_I_y',  # Outer Loop Integrals (Proof of wind resistance)
            'inn_I_x', 'inn_I_y',  # Inner Loop Integrals 
            'total_error'          # Total Euclidean Distance
        ])

# [MODIFIED]: Conversion multiplier. 
# Tello API takes -100 to 100 (percentage). Inner loop outputs m/s (e.g., -1.0 to 1.0).
# Increase this gain if the drone flies too slowly towards the target.
MS_TO_PERCENTAGE_GAIN = 60.0 

# ==========================================
# 2. MAIN CONTROLLER FUNCTION
# ==========================================
def controller(state, target_pos, timestamp, wind_enabled=False):
    global prev_timestamp, flight_data_buffer
    
    # --- Step 1: Unpack the Vicon state ---
    current_pos = np.array(state[0:3])
    current_yaw = state[5]
    
    target_position = np.array(target_pos[0:3])
    target_yaw = target_pos[3]

    # --- Step 2: Safely calculate physics dt ---
    if prev_timestamp is None:
        dt = 0.01
    else:
        dt = (timestamp - prev_timestamp) / 1000.0 # Convert ms to seconds
    
    # Cap dt to prevent integral explosion if Vicon network lags
    if dt <= 0.0 or dt > 0.5:
        dt = 0.01 
        
    prev_timestamp = timestamp

    # --- Step 3: Run Outer Loop (Cascade Layer 1) ---
    v_des_body, yaw_rate_cmd = outer_loop.compute_outer_loop(
        current_pos, target_position, current_yaw, target_yaw, dt
    )

    # --- Step 4: Run Inner Loop (Cascade Layer 2) ---
    final_v = inner_loop.compute_inner_loop(v_des_body, current_pos, dt, current_yaw)

    # Safety net for the first frame
    if final_v is None:
        final_v = [0.0, 0.0, 0.0]

    # --- Step 5: Convert m/s to percentage and Clamp ---
    cmd_x_percent = final_v[0] * MS_TO_PERCENTAGE_GAIN
    cmd_y_percent = final_v[1] * MS_TO_PERCENTAGE_GAIN
    cmd_z_percent = final_v[2] * MS_TO_PERCENTAGE_GAIN
    cmd_yaw_percent = yaw_rate_cmd * MS_TO_PERCENTAGE_GAIN

    # Clamp the final output strictly to the -100 to +100 Tello safety bounds
    cmd_x = np.clip(cmd_x_percent, -100.0, 100.0)
    cmd_y = np.clip(cmd_y_percent, -100.0, 100.0)
    cmd_z = np.clip(cmd_z_percent, -100.0, 100.0)
    cmd_yaw = np.clip(cmd_yaw_percent, -100.0, 100.0)

    # --- Step 6: Live Error Tracking & Data Logging ---
    total_error = np.linalg.norm(target_position - current_pos)
    
    # Print to terminal for live monitoring
    print(f"Error: {total_error:.3f}m | Thrust X: {cmd_x:.1f}% Y: {cmd_y:.1f}%")

    # Append EXACTLY 21 items to match the super header
    flight_data_buffer.append([
        timestamp, dt,
        current_pos[0], current_pos[1], current_pos[2], current_yaw,
        target_position[0], target_position[1], target_position[2], target_yaw,
        v_des_body[0], v_des_body[1],
        cmd_x, cmd_y, cmd_z, cmd_yaw,
        outer_loop.integral_pos[0], outer_loop.integral_pos[1], 
        inner_loop.integral_vel[0], inner_loop.integral_vel[1], 
        total_error
    ])

    # Batch write to disk every 100 frames (~2-3 seconds)
    if len(flight_data_buffer) >= 100:
        save_data_to_csv()

    # Return final commands to the drone
    return (cmd_x, cmd_y, cmd_z, cmd_yaw)

# ==========================================
# 3. DATA SAVING HELPER
# ==========================================
def save_data_to_csv():
    global flight_data_buffer
    if len(flight_data_buffer) == 0:
        return
        
    # Open the file and append the buffer block
    with open(fileName, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(flight_data_buffer)
        
    # Clear the RAM buffer after saving to prevent memory leaks
    flight_data_buffer.clear()