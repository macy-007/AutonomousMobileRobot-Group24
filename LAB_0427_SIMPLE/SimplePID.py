import numpy as np
import math
import csv
import os
import time
import atexit
import matplotlib.pyplot as plt

# ==========================================
# 1. CONTROLLER CLASSES & GLOBAL SETUP
# ==========================================

# --- OPTIMIZED SIMPLE PID GAINS ---
# Aim: Reduce overshoot and kill oscillations by adding KD and lowering KI.

SIMULATOR_POS_GAINS = {
    # [KP]: Lowered for XY to make the approach gentler.
    'kp': [0.6, 0.6, 1.2],   
    
    # [KI]: Drastically lowered. In a single-loop, KI is the main cause of overshoot.
    # Set this to a very small value just to clear steady-state error.
    'ki': [0.01, 0.01, 0.05],  
    
    # [KD]: Added KD! This is your "brake". Without KD, the drone has no way to 
    # slow down proactively as it nears the target.
    'kd': [0.15, 0.15, 0.2]    
}

SIMULATOR_YAW_GAINS = {
    'kp': 1.0,  # Lowered slightly to prevent snappy oscillations
    'ki': 0.0,
    'kd': 0.1   # Added some damping
}

gains_pos = SIMULATOR_POS_GAINS
gains_yaw = SIMULATOR_YAW_GAINS

class SimplePIDController:
    """
    A single-loop PID controller mapping position error directly to velocity commands.
    """
    def __init__(self):
        # PID value initialisation for Position
        self.kp_pos = np.array(gains_pos['kp'])
        self.ki_pos = np.array(gains_pos['ki'])
        self.kd_pos = np.array(gains_pos['kd'])

        # Error Tallys for Position
        self.integral_pos = np.zeros(3)
        self.prev_error_pos = np.zeros(3)
        self.max_integral_pos = 1.0 # Anti-windup limit

        # PID value initialisation for Yaw
        self.kp_yaw = gains_yaw['kp']
        self.ki_yaw = gains_yaw['ki']
        self.kd_yaw = gains_yaw['kd']

        # Error Tallys for Yaw
        self.integral_yaw = 0.0
        self.prev_error_yaw = 0.0

    def normalise_angle(self, angle):
        """ Keeps the angle within the -pi to pi range to find the shortest rotation path """
        return math.atan2(math.sin(angle), math.cos(angle))
    
    def global_to_body_frame(self, v_global_x, v_global_y, current_yaw):
        """ Transforms global velocity commands into the drone's local body frame """
        rotation_matrix = np.array([
            [np.cos(current_yaw), np.sin(current_yaw)],
            [-np.sin(current_yaw), np.cos(current_yaw)]
        ])

        v_global = np.array([v_global_x, v_global_y])
        v_body = rotation_matrix @ v_global
        return v_body[0], v_body[1]
    
    def compute_PID(self, current_pos, target_pos, current_yaw, target_yaw, dt):
        if dt <= 0.0:
            dt = 0.01

        # Reset integrals if the target changes significantly (prevents fly-aways)
        if getattr(self, 'prev_target_pos', None) is None:
            self.prev_target_pos = target_pos

        if np.linalg.norm(target_pos - self.prev_target_pos) > 0.1:
            self.integral_pos = np.zeros(3)
            self.integral_yaw = 0.0
            self.prev_error_pos = target_pos - current_pos 
            self.prev_error_yaw = self.normalise_angle(target_yaw - current_yaw)
            
        self.prev_target_pos = target_pos
            
        # --- Yaw Control ---
        error_yaw = self.normalise_angle(target_yaw - current_yaw)
        self.integral_yaw += error_yaw * dt
        yaw_derivative = self.normalise_angle(error_yaw - self.prev_error_yaw) / dt
        yaw_rate_cmd = (self.kp_yaw * error_yaw) + (self.ki_yaw * self.integral_yaw) + (self.kd_yaw * yaw_derivative)

        # --- Position Control ---
        error_pos = target_pos - current_pos
        self.integral_pos += error_pos * dt
        self.integral_pos = np.clip(self.integral_pos, -self.max_integral_pos, self.max_integral_pos)
        pos_derivative = (error_pos - self.prev_error_pos) / dt 

        v_des_global = (self.kp_pos * error_pos) + (self.ki_pos * self.integral_pos) + (self.kd_pos * pos_derivative)

        # Transform horizontal velocities to body frame
        v_body_x, v_body_y = self.global_to_body_frame(v_des_global[0], v_des_global[1], current_yaw)
        v_des_body = np.array([v_body_x, v_body_y, v_des_global[2]])

        # Update state memory
        self.prev_error_pos = error_pos
        self.prev_error_yaw = error_yaw
        
        return v_des_body, yaw_rate_cmd

# Instantiate the controller in global scope
pid_controller = SimplePIDController()

# Data Logging Globals
flight_data_buffer = []
global_time_elapsed = 0.0
timestamp_str = time.strftime("%Y%m%d_%H%M%S")
fileName = f"flight_data_{timestamp_str}.csv"

if not os.path.exists(fileName):
    with open(fileName, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'time_s', 'dt', 'cur_x', 'cur_y', 'cur_z', 'cur_yaw', 
            'tgt_x', 'tgt_y', 'tgt_z', 'tgt_yaw',
            'cmd_v_x', 'cmd_v_y', 'cmd_v_z', 'cmd_yaw_rate', 'total_error'
        ])

def save_data_to_csv():
    global flight_data_buffer
    if len(flight_data_buffer) == 0: return
    with open(fileName, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(flight_data_buffer)
    flight_data_buffer.clear()

def plot_flight_data():
    """Reads the CSV file and silently plots the flight performance when the program ends."""
    print("\n[INFO] Generating Flight Data Visualization...")
    save_data_to_csv()
    
    if not os.path.exists(fileName): return

    data = np.genfromtxt(fileName, delimiter=',', skip_header=1)
    if data.shape[0] < 2: return

    t = data[:, 0]
    cur_pos, tgt_pos = data[:, 2:5], data[:, 6:9]

    fig = plt.figure(figsize=(15, 8))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(cur_pos[:, 0], cur_pos[:, 1], cur_pos[:, 2], label='Actual Path', color='b')
    ax1.scatter(tgt_pos[:, 0], tgt_pos[:, 1], tgt_pos[:, 2], label='Target Points', color='r', marker='x')
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Flight Trajectory'); ax1.legend()

    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(t, cur_pos[:, 0], label='Current X', color='b')
    ax2.plot(t, tgt_pos[:, 0], label='Target X', color='r', linestyle='--')
    ax2.set_ylabel('X (m)'); ax2.legend(); ax2.grid(True)

    ax3 = fig.add_subplot(3, 2, 4)
    ax3.plot(t, cur_pos[:, 1], label='Current Y', color='g')
    ax3.plot(t, tgt_pos[:, 1], label='Target Y', color='r', linestyle='--')
    ax3.set_ylabel('Y (m)'); ax3.legend(); ax3.grid(True)

    ax4 = fig.add_subplot(3, 2, 6)
    ax4.plot(t, cur_pos[:, 2], label='Current Z', color='purple')
    ax4.plot(t, tgt_pos[:, 2], label='Target Z', color='r', linestyle='--')
    ax4.set_ylabel('Z (m)'); ax4.set_xlabel('Time (s)'); ax4.legend(); ax4.grid(True)

    plt.tight_layout()

    # Save silently to file instead of popping up
    img_name = fileName.replace(".csv", ".png")    
    plt.savefig(img_name, dpi=300, bbox_inches='tight')
    print(f"[INFO] Image saved to: {img_name}")    
    plt.close(fig)

atexit.register(plot_flight_data)

# ==========================================
# 2. REQUIRED SKELETON (MAIN FUNCTION)
# ==========================================

# wind_flag = False
# Implement a controller

def controller(state, target_pos, dt, wind_enabled=False):
    # state format: [position_x (m), position_y (m), position_z (m), roll (radians), pitch (radians), yaw (radians)]
    # target_pos format: (x (m), y (m), z (m), yaw (radians))
    # dt: time step (s)
    # wind_enabled: boolean flag to indicate if wind disturbance should be considered in the control algorithm
    # return velocity command format: (velocity_x_setpoint (m/s), velocity_y_setpoint (m/s), velocity_z_setpoint (m/s), yaw_rate_setpoint (radians/s))
    
    global global_time_elapsed, flight_data_buffer, pid_controller
    
    # 1. Unpack states
    current_pos = np.array(state[0:3])
    current_yaw = state[5]
    target_position = np.array(target_pos[0:3])
    target_yaw = target_pos[3]

    # 2. Safety check for dt
    if dt <= 0.0 or dt > 0.5:
        dt = 0.01 
    global_time_elapsed += dt

    # 3. Compute control outputs using the Simple PID
    v_des_body, yaw_rate_cmd = pid_controller.compute_PID(current_pos, target_position, current_yaw, target_yaw, dt)

    if v_des_body is None:
        v_des_body = [0.0, 0.0, 0.0]

    # 4. Limit Outputs for Safety (Crucial for Simple PID which can request extreme speeds)
    cmd_x = np.clip(v_des_body[0], -2.0, 2.0)
    cmd_y = np.clip(v_des_body[1], -2.0, 2.0)
    cmd_z = np.clip(v_des_body[2], -2.0, 2.0)
    cmd_yaw = np.clip(yaw_rate_cmd, -1.5, 1.5)

    # 5. Data Logging
    total_error = np.linalg.norm(target_position - current_pos)
    flight_data_buffer.append([
        global_time_elapsed, dt,
        current_pos[0], current_pos[1], current_pos[2], current_yaw,
        target_position[0], target_position[1], target_position[2], target_yaw,
        cmd_x, cmd_y, cmd_z, cmd_yaw, total_error
    ])

    if len(flight_data_buffer) >= 50:
        save_data_to_csv()

    # 6. Strict Skeleton Output Format
    output = (float(cmd_x), float(cmd_y), float(cmd_z), float(cmd_yaw))
    return output