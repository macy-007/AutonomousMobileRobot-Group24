import numpy as np
import time
import os
import csv
from OuterLoopController import OuterLoopController
from InnerLoopController import InnerLoopController

# Initialize the controllers outside the main function so they retain their memory 
outer_loop = OuterLoopController()
inner_loop = InnerLoopController()

# [MODIFIED]: Created a memory buffer for flight data to prevent disk I/O lag during flight
flight_data_buffer = []
prev_timestamp = None
timestamp_str = time.strftime("%Y%m%d_%H%M%S")
fileName = f"real_tello_flight_data_{timestamp_str}.csv"

# [MODIFIED]: Changed function signature to match the Vicon Tello real-world lab interface
def controller(state, target_pos, timestamp):
    global prev_timestamp, flight_data_buffer
    
    # 1. Unpack the Vicon state
    # state format: [x, y, z, roll, pitch, yaw]
    current_pos = np.array(state[0:3])
    current_yaw = state[5]
    
    # target_pos format: (x, y, z, yaw)
    target_position = np.array(target_pos[0:3])
    target_yaw = target_pos[3]

    # [MODIFIED]: Safely calculate 'dt' in seconds from the Vicon millisecond timestamp
    if prev_timestamp is None:
        dt = 0.01
    else:
        dt = (timestamp - prev_timestamp) / 1000.0 # Convert ms to seconds
    
    # [MODIFIED]: Safety cap for dt to prevent integral windup if the Vicon network lags
    if dt <= 0.0 or dt > 0.5:
        dt = 0.01 
        
    prev_timestamp = timestamp

    # [MODIFIED]: Append data to RAM buffer instead of writing to disk every frame
    flight_data_buffer.append([
        timestamp, current_pos[0], current_pos[1], current_pos[2], current_yaw,
        target_pos[0], target_pos[1], target_pos[2], target_pos[3]
    ])

    # [MODIFIED]: Batch write to disk every 200 frames to save I/O time without losing data
    if len(flight_data_buffer) >= 200:
        save_data_to_csv()

    # 2. Run the Outer Loop (Macy's Code)
    v_des_body, yaw_rate_cmd = outer_loop.compute_outer_loop(
        current_pos, target_position, current_yaw, target_yaw, dt
    )

    # 3. Run the Inner Loop (Ewan's Code)
    final_v = inner_loop.compute_inner_loop(v_des_body, current_pos, dt, current_yaw)

    # --- THE SAFETY NET ---
    if final_v is None:
        final_v = [0.0, 0.0, 0.0]

    # [MODIFIED]: Clamp the final output to -100/100 to match DJI Tello SDK safety limits
    cmd_x = np.clip(final_v[0], -100.0, 100.0)
    cmd_y = np.clip(final_v[1], -100.0, 100.0)
    cmd_z = np.clip(final_v[2], -100.0, 100.0)
    cmd_yaw = np.clip(yaw_rate_cmd, -100.0, 100.0)

    # --- LIVE ERROR TRACKING ---
    total_error = np.linalg.norm(target_position - current_pos)
    print(f"Live Error: {total_error:.4f} m | Cmd: X:{cmd_x:.1f} Y:{cmd_y:.1f}")

    # [MODIFIED]: Return the safely clamped actuation commands
    return (cmd_x, cmd_y, cmd_z, cmd_yaw)

# [MODIFIED]: Helper function to write buffered data to the CSV file
def save_data_to_csv():
    global flight_data_buffer
    if len(flight_data_buffer) == 0:
        return
        
    file_exists = os.path.exists(fileName)
    
    with open(fileName, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['timestamp_ms', 'cur_x', 'cur_y', 'cur_z', 'cur_yaw', 'tgt_x', 'tgt_y', 'tgt_z', 'tgt_yaw'])
        
        # Write all buffered rows at once
        writer.writerows(flight_data_buffer)
        
    # Clear the buffer after writing to start collecting new data
    flight_data_buffer.clear()