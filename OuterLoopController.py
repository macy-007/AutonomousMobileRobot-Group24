import numpy as np
import math


# Use the SIMULATOR_GAINS for simulation and TELLO_REAL_GAINS for real-world testing.
SIMULATOR_GAINS = {
    'kp': [1.2, 1.2, 1.5],
    'ki': [0.1, 0.1, 0.2],
    'kd': [0.3, 0.3, 0.3]
}
gains = SIMULATOR_GAINS

# TELLO_REAL_GAINS = {
#     'kp': [0.5, 0.5, 0.8], 
#     'ki': [0.01, 0.01, 0.05],
#     'kd': [0.1, 0.1, 0.1]
# }
# gains = TELLO_REAL_GAINS 

class OuterLoopController:
    """
    Role 2: Outer Loop (Position & Yaw) Controller
    Responsible for generating desired body velocities and handling coordinate transformations.
    """
    def __init__(self):        
        # --- Position Loop PID Gains (Position -> Desired Velocity) ---
        # Note: Outer loop proportional gains are typically moderate to prevent 
        # commanding unattainable speeds to the inner velocity loop.
        self.kp_pos = np.array(gains['kp']) 
        self.ki_pos = np.array(gains['ki']) 
        self.kd_pos = np.array(gains['kd'])
        
        # State memory for position integration and derivation
        self.integral_pos = np.zeros(3)
        self.prev_error_pos = np.zeros(3)
        self.max_integral_pos = 1.0 # Integral anti-windup clamping limit
        
        # --- Yaw Loop PID Gains (Yaw -> Yaw Rate) ---
        self.kp_yaw = 1.5
        self.ki_yaw = 0.0
        self.kd_yaw = 0.2
        
        # State memory for yaw integration and derivation
        self.integral_yaw = 0.0
        self.prev_error_yaw = 0.0

    def normalize_angle(self, angle):
        """
        Normalizes the angle to the [-pi, pi] range.
        Ensures the UAV always takes the shortest rotational path to the target yaw.
        """
        return math.atan2(math.sin(angle), math.cos(angle))

    def global_to_body_frame(self, v_global_x, v_global_y, current_yaw):
        """
        Coordinate Transformation: Converts global X/Y velocity commands 
        into the Tello's local body-frame velocity commands using a 2D rotation matrix.
        """
        rotation_matrix = np.array([
            [np.cos(current_yaw), np.sin(current_yaw)],
            [-np.sin(current_yaw), np.cos(current_yaw)]
        ])
        
        v_global = np.array([v_global_x, v_global_y])
        v_body = rotation_matrix @ v_global
        
        return v_body[0], v_body[1]

    # [MODIFIED]: Added 'dt' as a required argument passed from the simulator
    def compute_outer_loop(self, current_pos, target_pos, current_yaw, target_yaw, dt):
        """
        Executes the outer loop control logic.
        Returns: 
            v_des_body (np.array): Desired velocity in the BODY frame [Vx, Vy, Vz].
            yaw_rate_cmd (float): Commanded yaw rate.
        """
        # Prevent division by zero on the very first control iteration just in case
        if dt <= 0.0:
            dt = 0.01 
            
        # ----------------------------------------------------
        # 1. Yaw Control (Calculates Yaw Rate Command)
        # ----------------------------------------------------
        error_yaw = self.normalize_angle(target_yaw - current_yaw)
        
        self.integral_yaw += error_yaw * dt
        derivative_yaw = (error_yaw - self.prev_error_yaw) / dt
        
        yaw_rate_cmd = (self.kp_yaw * error_yaw) + (self.ki_yaw * self.integral_yaw) + (self.kd_yaw * derivative_yaw)
        
        # ----------------------------------------------------
        # 2. Position Control (Calculates Desired Global Velocity)
        # ----------------------------------------------------
        error_pos = target_pos - current_pos
        
        # Accumulate integral and apply anti-windup clamping
        self.integral_pos += error_pos * dt
        self.integral_pos = np.clip(self.integral_pos, -self.max_integral_pos, self.max_integral_pos)
        
        derivative_pos = (error_pos - self.prev_error_pos) / dt
        
        # Calculate the desired velocity vector via PID formula (This is in GLOBAL frame)
        v_des_global = (self.kp_pos * error_pos) + (self.ki_pos * self.integral_pos) + (self.kd_pos * derivative_pos)

        # ----------------------------------------------------
        # 3. Coordinate Transformation (Global to Body)
        # ----------------------------------------------------
        # [MODIFIED]: Transform global X/Y to body X/Y to feed the Inner Loop correctly
        v_body_x, v_body_y = self.global_to_body_frame(v_des_global[0], v_des_global[1], current_yaw)
        
        # Reconstruct the 3D velocity array in the body frame 
        # (Z-axis is the same in both global and body frames for a standard quadcopter)
        v_des_body = np.array([v_body_x, v_body_y, v_des_global[2]])

        # ----------------------------------------------------
        # 4. State Update
        # ----------------------------------------------------
        self.prev_error_pos = error_pos
        self.prev_error_yaw = error_yaw
        # [MODIFIED]: Now returning v_des_body instead of v_des_global
        return v_des_body, yaw_rate_cmd