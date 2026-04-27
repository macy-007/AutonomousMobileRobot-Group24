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

# --- SIMULATOR GAINS (Decoupled) ---

SIM_OUTER_POS_GAINS = {
    'kp': [0.5, 0.5, 2.0],   
    'ki': [0.02, 0.02, 0.1], 
    'kd': [0.0, 0.0, 0.0]    
}

SIM_INNER_VEL_GAINS = {
    'kp': [0.3, 0.3, 0.2],   
    'ki': [0.01, 0.01, 0.1],   
    'kd': [0.0, 0.0, 0.0]  
}

outer_gains = SIM_OUTER_POS_GAINS
inner_gains = SIM_INNER_VEL_GAINS

class InnerLoopController:
    def __init__(self):
        self.prev_pos = None
        # Loaded specific inner loop gains
        self.kp_vel = np.array(inner_gains['kp'])
        self.ki_vel = np.array(inner_gains['ki'])
        self.kd_vel = np.array(inner_gains['kd'])
        
        self.integral_vel = np.zeros(3)
        self.prev_error_vel = np.zeros(3)
        self.max_integral_vel = np.array([0.5, 0.5, 1.0])
        self.max_velocity = 2.0 # m/s (Absolute safety limit for inner loop)

    def global_to_body_frame(self, v_global_x, v_global_y, current_yaw):
        rotation_matrix = np.array([
            [np.cos(current_yaw), np.sin(current_yaw)],
            [-np.sin(current_yaw), np.cos(current_yaw)]
        ])
        v_global = np.array([v_global_x, v_global_y])
        v_body = rotation_matrix @ v_global
        return v_body[0], v_body[1]

    def compute_inner_loop(self, v_des_body, current_pos, dt, current_yaw):
        if dt <= 0.0: dt = 0.01
        if self.prev_pos is None:
            self.prev_pos = current_pos
            return np.array([0.0, 0.0, 0.0])
        
        current_vel = (current_pos - self.prev_pos) / dt
        v_body_x, v_body_y = self.global_to_body_frame(current_vel[0], current_vel[1], current_yaw)
        current_vel = np.array([v_body_x, v_body_y, current_vel[2]])
        
        error_vel = v_des_body - current_vel
        prop_term = self.kp_vel * error_vel

        self.integral_vel += error_vel * dt
        self.integral_vel = np.clip(self.integral_vel, -self.max_integral_vel, self.max_integral_vel)
        integral_term = self.ki_vel * self.integral_vel

        deriv_vel = (error_vel - self.prev_error_vel) / dt
        deriv_term = self.kd_vel * deriv_vel

        v_out = prop_term + integral_term + deriv_term + v_des_body
        v_out = np.clip(v_out, -self.max_velocity, self.max_velocity)

        self.prev_pos = current_pos
        self.prev_error_vel = error_vel
        return v_out

class OuterLoopController:
    def __init__(self):        
        # Loaded specific outer loop gains
        self.kp_pos = np.array(outer_gains['kp']) 
        self.ki_pos = np.array(outer_gains['ki']) 
        self.kd_pos = np.array(outer_gains['kd'])
        
        self.integral_pos = np.zeros(3)
        self.prev_error_pos = np.zeros(3)
        # [MODIFIED]: Relaxed outer integral limit to allow I-term to grow and push through steady-state error
        self.max_integral_pos = np.array([1.5, 1.5, 2.0]) 
        
        # [MODIFIED]: Softened Yaw parameters to prevent aggressive swaying
        self.kp_yaw = 0.8   # Lowered from 1.5
        self.ki_yaw = 0.05  # Added small integral for steady yaw error
        self.kd_yaw = 0.00  # Added slight damping
        
        self.integral_yaw = 0.0
        self.prev_error_yaw = 0.0

        self.prev_v_des_global = np.zeros(3)
        self.max_acceleration = 3.5  

    def normalize_angle(self, angle):
        # Prevents spin of death when wrapping around pi and -pi
        return math.atan2(math.sin(angle), math.cos(angle))

    def global_to_body_frame(self, v_global_x, v_global_y, current_yaw):
        rotation_matrix = np.array([
            [np.cos(current_yaw), np.sin(current_yaw)],
            [-np.sin(current_yaw), np.cos(current_yaw)]
        ])
        v_global = np.array([v_global_x, v_global_y])
        v_body = rotation_matrix @ v_global
        return v_body[0], v_body[1]

    def compute_outer_loop(self, current_pos, target_pos, current_yaw, target_yaw, dt):
        if dt <= 0.0: dt = 0.01 
        
        # Reset integrals if target changes significantly
        if getattr(self, 'prev_target_pos', None) is None:
            self.prev_target_pos = target_pos

        if np.linalg.norm(target_pos - self.prev_target_pos) > 0.1:
            self.integral_pos = np.zeros(3)
            self.integral_yaw = 0.0
            self.prev_error_pos = target_pos - current_pos 
            self.prev_error_yaw = self.normalize_angle(target_yaw - current_yaw)
            
        self.prev_target_pos = target_pos
            
        # 1. Yaw Control
        error_yaw = self.normalize_angle(target_yaw - current_yaw)
        self.integral_yaw += error_yaw * dt
        derivative_yaw = self.normalize_angle(error_yaw - self.prev_error_yaw) / dt
        yaw_rate_cmd = (self.kp_yaw * error_yaw) + (self.ki_yaw * self.integral_yaw) + (self.kd_yaw * derivative_yaw)
        yaw_rate_cmd = np.clip(yaw_rate_cmd, -1.5, 1.5)
        
        # 2. Position Control
        error_pos = target_pos - current_pos
        self.integral_pos += error_pos * dt
        self.integral_pos = np.clip(self.integral_pos, -self.max_integral_pos, self.max_integral_pos)
        derivative_pos = (error_pos - self.prev_error_pos) / dt
        
        v_des_global_raw = (self.kp_pos * error_pos) + (self.ki_pos * self.integral_pos) + (self.kd_pos * derivative_pos)
        
        # [MODIFIED]: Solves Z-axis lag by limiting horizontal speed so Z can catch up
        v_des_global_raw[0] = np.clip(v_des_global_raw[0], -0.8, 0.8) # X max
        v_des_global_raw[1] = np.clip(v_des_global_raw[1], -0.8, 0.8) # Y max
        v_des_global_raw[2] = np.clip(v_des_global_raw[2], -1.0, 1.0) # Z max

        # Acceleration slew limiter
        max_dv = self.max_acceleration * dt
        v_des_global = np.clip(v_des_global_raw, self.prev_v_des_global - max_dv, self.prev_v_des_global + max_dv)
        self.prev_v_des_global = v_des_global
        
        # 3. Global to Body Transformation
        v_body_x, v_body_y = self.global_to_body_frame(v_des_global[0], v_des_global[1], current_yaw)
        v_des_body = np.array([v_body_x, v_body_y, v_des_global[2]])

        self.prev_error_pos = error_pos
        self.prev_error_yaw = error_yaw
        return v_des_body, yaw_rate_cmd

# Instantiate controllers in global scope to retain memory between frames
outer_loop = OuterLoopController()
inner_loop = InnerLoopController()

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
    """Reads the CSV file and plots the flight performance when the program ends."""
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

    img_name = fileName.replace(".csv", ".png")    
    plt.savefig(img_name, dpi=300, bbox_inches='tight')
    print(f"[INFO] : {img_name}")    
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
    
    global global_time_elapsed, flight_data_buffer, outer_loop, inner_loop
    
    # 1. Unpack states
    current_pos = np.array(state[0:3])
    current_yaw = state[5]
    target_position = np.array(target_pos[0:3])
    target_yaw = target_pos[3]

    # 2. Safety check for dt
    if dt <= 0.0 or dt > 0.5:
        dt = 0.01 
    global_time_elapsed += dt

    # 3. Compute control outputs
    v_des_body, yaw_rate_cmd = outer_loop.compute_outer_loop(current_pos, target_position, current_yaw, target_yaw, dt)
    final_v = inner_loop.compute_inner_loop(v_des_body, current_pos, dt, current_yaw)

    if final_v is None:
        final_v = [0.0, 0.0, 0.0]

    # 4. Limit Outputs for Safety
    cmd_x = np.clip(final_v[0], -2.0, 2.0)
    cmd_y = np.clip(final_v[1], -2.0, 2.0)
    cmd_z = np.clip(final_v[2], -2.0, 2.0)
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