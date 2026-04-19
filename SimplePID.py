import numpy as np
import math

SIMULATOR_POS_GAINS = {
    'kp': [1.0, 1.0, 0.8],
    'ki': [0.1, 0.1, 0.5],
    'kd': [0.0, 0.0, 0.0]
}

SIMULATOR_YAW_GAINS = {
    'kp': 1.5,
    'ki': 0.0,
    'kd': 0.2
}

gains_pos = SIMULATOR_POS_GAINS
gains_yaw = SIMULATOR_YAW_GAINS

class SimplePIDController:
    def __init__(self):
                # PID value initialisation
        self.kp_pos = np.array(gains_pos['kp'])
        self.ki_pos = np.array(gains_pos['ki'])
        self.kd_pos = np.array(gains_pos['kd'])

        # Error Tallys
        self.integral_pos = np.zeros(3)
        self.prev_error_pos = np.zeros(3)
        self.max_integral_pos = 1.0 # Anti-windup

        self.kp_yaw = np.array(gains_yaw['kp'])
        self.ki_yaw = np.array(gains_yaw['ki'])
        self.kd_yaw = np.array(gains_yaw['kd'])

        self.integral_yaw = 0.0
        self.prev_error_yaw = 0.0

    def normalise_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))
    
    def global_to_body_frame(self, v_global_x, v_global_y, current_yaw):

            rotation_matrix = np.array([
                [np.cos(current_yaw), np.sin(current_yaw)],
                [-np.sin(current_yaw), np.cos(current_yaw)]
            ])

            v_global = np.array([v_global_x, v_global_y])
            v_body = rotation_matrix @ v_global
            return v_body[0], v_body[1]
    
    def compute_PID(self,current_pos, target_pos, current_yaw, target_yaw, dt):
         
        if dt <= 0.0:
            dt = 0.01
            
        error_yaw = self.normalise_angle(target_yaw - current_yaw)
        self.integral_yaw += error_yaw * dt
        yaw_derivative = (error_yaw - self.prev_error_yaw) / dt
        yaw_rate_cmd = (self.kp_yaw * error_yaw) + (self.ki_yaw* self.integral_yaw) + (self.kd_yaw * yaw_derivative)

        error_pos = target_pos - current_pos
        self.integral_pos += error_pos * dt
        self.integral_pos = np.clip(self.integral_pos, -self.max_integral_pos, self.max_integral_pos)
        pos_derivative = (error_pos - self.prev_error_pos) / dt 

        v_des_global = (self.kp_pos * error_pos) + (self.ki_pos * self.integral_pos) + (self.kd_pos * pos_derivative)

        v_body_x, v_body_y = self.global_to_body_frame(v_des_global[0], v_des_global[1], current_yaw)
    
        v_des_body = np.array([v_body_x, v_body_y, v_des_global[2]])

        self.prev_error_pos = error_pos
        self.prev_error_yaw = error_yaw
        
        return v_des_body, yaw_rate_cmd