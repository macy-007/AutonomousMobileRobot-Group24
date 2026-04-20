import numpy as np

SIMULATOR_GAINS = {
    'kp': [1.0, 1.0, 0.8],
    'ki': [0.1, 0.1, 0.5],
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
        self.max_integral_vel = 1.0 # Anti-windup

        # Velocity Limit 
        self.max_velocity = 1.0 # m/s

    def compute_inner_loop(self,v_des_body, current_pos, dt):
        
        # Prevents division by zero on first frame.
        if dt <= 0.0:
            dt = 0.01

        # Frame 1 - Intiialses position
        if self.prev_pos is None:
            self.prev_pos = current_pos
            return np.array([0.0,0.0,0.0])
        
        current_vel = (current_pos - self.prev_pos) / dt

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
    