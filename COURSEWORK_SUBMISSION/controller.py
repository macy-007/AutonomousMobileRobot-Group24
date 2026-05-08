"""
Advanced Method Explanation: Cascade PID Controller
Our cascade controller is divided into two loops:
1. Outer Loop (Position Controller): Computes the desired global velocity based on position error.
2. Inner Loop (Velocity Controller): Computes the final control commands to track the desired velocity.

PID Controllers (Proportional Integral Derivative) utilise errors to find its goal:

Proportional: Response is proportional to the size of the current error.
Integral: Accumulates past error over a time-frame.
Derivative: Predicts future error by measuring the errors rate of change.
"""

import numpy as np
import math

# CONTROLLER GAINS 
# Tuned for gentle approach to prevent positional overshoot
SIM_OUTER_POS_GAINS = {
    'kp': [1.0, 1.0, 3.0],   
    'ki': [0.2, 0.2, 0.5],  
    'kd': [0.05, 0.05, 0.4]      
}
# Tuned stiffly to reject wind disturbances quickly
SIM_INNER_VEL_GAINS = {
    'kp': [0.3, 0.3, 0.2],   # Driver's reflexes (keep these the same)
    'ki': [0.01, 0.01, 0.1],
    'kd': [0.0, 0.0, 0.0]    
}

# CONTROLLER CLASSES
class InnerLoopController:
    """ Tracks desired velocities. Reacts fast to reject wind. """
    def __init__(self):
        self.prev_pos = None
        self.kp_vel = np.array(SIM_INNER_VEL_GAINS['kp'])
        self.ki_vel = np.array(SIM_INNER_VEL_GAINS['ki'])
        self.kd_vel = np.array(SIM_INNER_VEL_GAINS['kd'])
        self.integral_vel = np.zeros(3)
        self.prev_error_vel = np.zeros(3)

        # Anti-windup limits for safety
        self.max_integral_vel = np.array([0.5, 0.5, 1.0])
        self.max_velocity = 2.0 # m/s

    def global_to_body_frame(self, v_global_x, v_global_y, current_yaw):
        # 2D rotation matrix for horizontal velocity mapping.
        rotation_matrix = np.array([
            [np.cos(current_yaw), np.sin(current_yaw)],
            [-np.sin(current_yaw), np.cos(current_yaw)]
        ])
        v_global = np.array([v_global_x, v_global_y])
        v_body = rotation_matrix @ v_global
        return v_body[0], v_body[1]

    def compute_inner_loop(self, v_des_body, current_pos, dt, current_yaw):

        # Ensure at intialisation, division by zero doesn't occur.
        if dt <= 0.0: 
            dt = 0.01

        # Initialses initial position.
        if self.prev_pos is None:
            self.prev_pos = current_pos
            return np.array([0.0, 0.0, 0.0])
        
        # Estimate current velocity using backward difference.
        current_vel = (current_pos - self.prev_pos) / dt
        # Transform velocity to body frame for comparison.
        v_body_x, v_body_y = self.global_to_body_frame(current_vel[0], current_vel[1], current_yaw)
        current_vel = np.array([v_body_x, v_body_y, current_vel[2]])

        # PID calculation.
        error_vel = v_des_body - current_vel
        prop_term = self.kp_vel * error_vel
        self.integral_vel += error_vel * dt
        self.integral_vel = np.clip(self.integral_vel, -self.max_integral_vel, self.max_integral_vel) # Anti-windup
        integral_term = self.ki_vel * self.integral_vel
        deriv_vel = (error_vel - self.prev_error_vel) / dt
        deriv_term = self.kd_vel * deriv_vel

        # Output command (Feedforward + PID)
        v_out = prop_term + integral_term + deriv_term + v_des_body

        # Limit velocity output.
        v_out = np.clip(v_out, -self.max_velocity, self.max_velocity)
        self.prev_pos = current_pos
        self.prev_error_vel = error_vel
        return v_out

class OuterLoopController:
    """ Computes desired velocity based on distance to target. """
    def __init__(self):        
        self.kp_pos = np.array(SIM_OUTER_POS_GAINS['kp']) 
        self.ki_pos = np.array(SIM_OUTER_POS_GAINS['ki']) 
        self.kd_pos = np.array(SIM_OUTER_POS_GAINS['kd'])
        self.integral_pos = np.zeros(3)
        self.prev_error_pos = np.zeros(3)
        self.max_integral_pos = np.array([1.5, 1.5, 2.0]) 
        # Yaw parameters
        self.kp_yaw = 0.8   
        self.ki_yaw = 0.05  
        self.kd_yaw = 0.00  
        self.integral_yaw = 0.0
        self.prev_error_yaw = 0.0
        self.prev_target_pos = None
        self.prev_v_des_global = np.zeros(3)
        self.max_acceleration = 3.5  

    def normalize_angle(self, angle):
        # Keeps yaw error between -pi and pi to ensure shortest rotation path
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
        
        # Reset integrals if the mission planner gives a completely new target point
        if self.prev_target_pos is None:
            self.prev_target_pos = target_pos
        if np.linalg.norm(target_pos - self.prev_target_pos) > 0.1:
            self.integral_pos = np.zeros(3)
            self.integral_yaw = 0.0
            self.prev_error_pos = target_pos - current_pos 
            self.prev_error_yaw = self.normalize_angle(target_yaw - current_yaw)
        self.prev_target_pos = target_pos
        # --- 1. Yaw Control ---
        error_yaw = self.normalize_angle(target_yaw - current_yaw)
        self.integral_yaw += error_yaw * dt
        derivative_yaw = self.normalize_angle(error_yaw - self.prev_error_yaw) / dt
        yaw_rate_cmd = (self.kp_yaw * error_yaw) + (self.ki_yaw * self.integral_yaw) + (self.kd_yaw * derivative_yaw)
        yaw_rate_cmd = np.clip(yaw_rate_cmd, -1.5, 1.5)
        # --- 2. Position Control ---
        error_pos = target_pos - current_pos
        self.integral_pos += error_pos * dt
        self.integral_pos = np.clip(self.integral_pos, -self.max_integral_pos, self.max_integral_pos)
        derivative_pos = (error_pos - self.prev_error_pos) / dt
        v_des_global_raw = (self.kp_pos * error_pos) + (self.ki_pos * self.integral_pos) + (self.kd_pos * derivative_pos)
        # Limit horizontal speed slightly to prioritize Z-axis climbing if needed
        v_des_global_raw[0] = np.clip(v_des_global_raw[0], -0.8, 0.8) 
        v_des_global_raw[1] = np.clip(v_des_global_raw[1], -0.8, 0.8) 
        v_des_global_raw[2] = np.clip(v_des_global_raw[2], -1.0, 1.0) 
        # Acceleration slew limiter (prevents sudden jerky movements)
        max_dv = self.max_acceleration * dt
        v_des_global = np.clip(v_des_global_raw, self.prev_v_des_global - max_dv, self.prev_v_des_global + max_dv)
        self.prev_v_des_global = v_des_global
        # Transform the final desired velocity into the body frame for the inner loop
        v_body_x, v_body_y = self.global_to_body_frame(v_des_global[0], v_des_global[1], current_yaw)
        v_des_body = np.array([v_body_x, v_body_y, v_des_global[2]])
        self.prev_error_pos = error_pos
        self.prev_error_yaw = error_yaw
        return v_des_body, yaw_rate_cmd

# Instantiate controllers
outer_loop = OuterLoopController()
inner_loop = InnerLoopController()

def controller(state, target_pos, dt, wind_enabled=False):
    """
    Main entry point for the simulation.
    state: [pos_x, pos_y, pos_z, roll, pitch, yaw]
    target_pos: [tgt_x, tgt_y, tgt_z, tgt_yaw]
    """
    global outer_loop, inner_loop
    # 1. Unpack states
    current_pos = np.array(state[0:3])
    current_yaw = state[5]
    target_position = np.array(target_pos[0:3])
    target_yaw = target_pos[3]
    # 2. Safety check for time step
    if dt <= 0.0 or dt > 0.5:
        dt = 0.01 
    # 3. Compute cascade control outputs
    # The outer loop gives us the velocity needed to reach the target position
    v_des_body, yaw_rate_cmd = outer_loop.compute_outer_loop(current_pos, target_position, current_yaw, target_yaw, dt)
    # The inner loop gives us the final commands to track that velocity (handles wind natively)
    final_v = inner_loop.compute_inner_loop(v_des_body, current_pos, dt, current_yaw)
    if final_v is None:
        final_v = [0.0, 0.0, 0.0]
    # 4. Strict Output Clipping for Simulator Safety
    cmd_x = np.clip(final_v[0], -2.0, 2.0)
    cmd_y = np.clip(final_v[1], -2.0, 2.0)
    cmd_z = np.clip(final_v[2], -2.0, 2.0)
    cmd_yaw = np.clip(yaw_rate_cmd, -1.5, 1.5)
    output = (float(cmd_x), float(cmd_y), float(cmd_z), float(cmd_yaw))
    return output