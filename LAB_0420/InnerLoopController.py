import numpy as np

SIMULATOR_GAINS = {
    'kp': [0.3, 0.3, 0.4],   # Driver's reflexes (keep these the same)
    'ki': [0.05, 0.05, 0.1],
    'kd': [0.0, 0.0, 0.0]    
}

gains = SIMULATOR_GAINS

class InnerLoopController:

    """
    Role 1: Inner Loop
    Ensure the drone achieves the commanded body velocity.
    """

    def __init__(self):

        # Position
        self.prev_pos = None

        # PID value initialisation
        self.kp_vel = np.array(gains['kp'])
        self.ki_vel = np.array(gains['ki'])
        self.kd_vel = np.array(gains['kd'])

        # Error Tallys
        self.integral_vel = np.zeros(3)
        self.prev_error_vel = np.zeros(3)
        self.max_integral_vel = np.array([0.5, 0.5, 1.0])

        # Velocity Limit 
        self.max_velocity = 1.0 # m/s

    def global_to_body_frame(self, v_global_x, v_global_y, current_yaw):
            rotation_matrix = np.array([
                [np.cos(current_yaw), np.sin(current_yaw)],
                [-np.sin(current_yaw), np.cos(current_yaw)]
            ])
            v_global = np.array([v_global_x, v_global_y])
            v_body = rotation_matrix @ v_global
            return v_body[0], v_body[1]

    def compute_inner_loop(self,v_des_body, current_pos, dt, current_yaw):
        
        # Prevents division by zero on first frame.
        if dt <= 0.0:
            dt = 0.01

        # Frame 1 - Intiialses position
        if self.prev_pos is None:
            self.prev_pos = current_pos
            return np.array([0.0,0.0,0.0])
        
        current_vel = (current_pos - self.prev_pos) / dt

        v_body_x, v_body_y = self.global_to_body_frame(
            current_vel[0], current_vel[1], current_yaw
        )

        current_vel = np.array([v_body_x, v_body_y, current_vel[2]])

        # Error Calculation.
        error_vel = v_des_body - current_vel

        # Proportional Term Calculation.
        prop_term = self.kp_vel * error_vel

        # Integral Term Calculation
        self.integral_vel += error_vel * dt
        self.integral_vel = np.clip(self.integral_vel, -self.max_integral_vel, self.max_integral_vel)
        integral_term = self.ki_vel * self.integral_vel

        # Derivative Term Calculation
        deriv_vel = (error_vel - self.prev_error_vel) / dt
        deriv_term = self.kd_vel * deriv_vel


        # Calculate New Velcoity
        v_out = prop_term + integral_term + deriv_term + v_des_body
        v_out = np.clip(v_out, -self.max_velocity, self.max_velocity)

        # Update State Memory
        self.prev_pos = current_pos
        self.prev_error_vel = error_vel

        return v_out
    