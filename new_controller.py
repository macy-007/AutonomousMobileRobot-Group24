import numpy as np
from OuterLoopController import OuterLoop
from InnerLoopController import InnerLoopController

# Initialize the controllers outside the main function so they retain their memory 
# (This is critical for their Integral and Derivative math to work over time)
outer_loop = OuterLoop()
inner_loop = InnerLoopController()

def controller(state, target_pos, dt, wind_enabled=False):
    # 1. Unpack the simulator state
    # state format: [x, y, z, roll, pitch, yaw]
    current_pos = np.array(state[0:3])
    current_yaw = state[5]
    
    # target_pos format: (x, y, z, yaw)
    target_position = np.array(target_pos[0:3])
    target_yaw = target_pos[3]

    # 2. Run the Outer Loop (Macy's Code)
    # This outputs desired body velocity AND handles the axis transformation automatically!
    v_des_body, yaw_rate_cmd = outer_loop.compute_outer_loop(
        current_pos, target_position, current_yaw, target_yaw, dt
    )

    # 3. Run the Inner Loop (Ewan's Code)
    # This takes the desired body velocity and outputs the final motor commands
    final_v = inner_loop.compute_inner_loop(v_des_body, current_pos, dt)

    # --- THE SAFETY NET ---
    # If the inner loop accidentally returns None (like on the very first frame), default to zero velocity
    if final_v is None:
        final_v = [0.0, 0.0, 0.0]

    # --- LIVE ERROR TRACKING ---
    # np.linalg.norm calculates the total 3D distance between the two points
    total_error = np.linalg.norm(target_position - current_pos)
    print(f"Live Error: {total_error:.4f} meters")

    # 4. Return the final actuation commands to the simulator
    return (final_v[0], final_v[1], final_v[2], yaw_rate_cmd)
