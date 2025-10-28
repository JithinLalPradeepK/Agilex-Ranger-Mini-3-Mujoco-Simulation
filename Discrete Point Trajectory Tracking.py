import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import math
import matplotlib.pyplot as plt

xml_path = 'ranger_mini.xml' #xml file (assumes this is in the same folder as this file)
simend = 20 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

# --- NAVIGATION & CONTROL GLOBALS ---
# Define the full trajectory as a list of [x, y] coordinates
TRAJECTORY_POINTS = np.array([
    [0.0, 0.0],  # Point 1: Start
    [2.0, 0.0],  # Point 2
    [2.0, 2.0],  # Point 3
    [0.0, 2.0],  # Point 4
    [-2.0, 0.0]  # Point 5: Final Goal
])

# Global index tracking the current target waypoint
TRAJECTORY_INDEX = 1 
# The actual target goal will be updated dynamically from the TRAJECTORY_POINTS array

GOAL_TOLERANCE = 0.1 # Distance in meters to consider the goal reached
WHEEL_RADIUS = 0.09 # Approximate wheel radius in meters for speed conversion

# Navigation Gains
KP_LIN = 8    # Proportional gain for linear velocity
KP_ANG = 2.0    # Proportional gain for steering angle

# Thresholds (Tuned for speed)
GOAL_TOLERANCE = 0.05 # Distance to consider goal reached (m)
MAX_LIN_VEL = 10.0    # Set maximum speed (m/s)
MIN_CRUISE_SPEED = 0.5 # Minimum speed when close to but not at goal

# Steering Constraints
MAX_STEER_RAD = 2.1  # Max steering angle from XML joint limit (approx 120 degrees)
EPSILON = 0.05       # Deadband for stability near 90/270 degrees
MAX_FORWARD_ANGLE = math.pi / 2.0 # 90 degrees (pi/2 rad)

# Actuator names from the XML file
STEERING_ACTUATORS = {
    'fl': 'fl_steering', 'fr': 'fr_steering',
    'rl': 'rl_steering', 'rr': 'rr_steering'
}
WHEEL_ACTUATORS = {
    'fl': 'fl_wheel_motor', 'fr': 'fr_wheel_motor',
    'rl': 'rl_wheel_motor', 'rr': 'rr_wheel_motor'
}

# --- NEW OBLIQUE STEERING KINEMATICS ---
def calculate_oblique_wheel_commands(speed, steering_angle):
    """
    Calculates wheel commands for oblique (crab-like) motion.
    All wheels are steered to the same angle, and all move at the same speed.
    """
    wheel_velocities = {
        'fl': speed,
        'fr': speed,
        'rl': speed,
        'rr': speed
    }
    
    steering_angles = {
        'fl': steering_angle,
        'fr': steering_angle,
        'rl': steering_angle,
        'rr': steering_angle
    }
    
    return wheel_velocities, steering_angles

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

# --- Global Controller and Navigation Variables ---
pos_pid = None
vel_pid = None

class VelocityPIDController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.kp = 80.0
        self.ki = 0.5
        self.kd = 1.0
        self.prev_error = {'fl': 0.0, 'fr': 0.0, 'rl': 0.0, 'rr': 0.0}
        self.integral = {'fl': 0.0, 'fr': 0.0, 'rl': 0.0, 'rr': 0.0}
        self.wheel_joints = {'fl': 'fl_wheel', 'fr': 'fr_wheel', 'rl': 'rl_wheel', 'rr': 'rr_wheel'}
        self.wheel_actuators = {'fl': 'fl_wheel_motor', 'fr': 'fr_wheel_motor', 'rl': 'rl_wheel_motor', 'rr': 'rr_wheel_motor'}
        self.max_control = 10.0
        self.min_control = -10.0

    def compute_vel_pid(self, wheel, setpoint):
        joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, self.wheel_joints[wheel])
        current_vel = self.data.joint(joint_id).qvel[0]
        error = setpoint - current_vel
        self.integral[wheel] += error
        derivative = error - self.prev_error[wheel]
        output = (self.kp * error + self.ki * self.integral[wheel] + self.kd * derivative)
        output = np.clip(output, self.min_control, self.max_control)
        self.prev_error[wheel] = error
        return output

    def set_wheel_vel(self, velocities):
        for wheel in ['fl', 'fr', 'rl', 'rr']:
            if wheel in velocities:
                control = self.compute_vel_pid(wheel, velocities[wheel])
                actuator_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, self.wheel_actuators[wheel])
                self.data.ctrl[actuator_id] = control

class AnglePIDController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.kp = 10.0
        self.ki = 0.01
        self.kd = 1.0
        self.prev_error = {'fl': 0.0, 'fr': 0.0, 'rl': 0.0, 'rr': 0.0}
        self.integral = {'fl': 0.0, 'fr': 0.0, 'rl': 0.0, 'rr': 0.0}
        self.steering_joints = {'fl': 'fl_steering_joint', 'fr': 'fr_steering_joint', 'rl': 'rl_steering_joint', 'rr': 'rr_steering_joint'}
        self.steering_actuators = {'fl': 'fl_steering', 'fr': 'fr_steering', 'rl': 'rl_steering', 'rr': 'rr_steering'}
        self.max_control = math.pi / 2
        self.min_control = -math.pi / 2

    def compute_pos_pid(self, wheel, setpoint):
        joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, self.steering_joints[wheel])
        current_pos = self.data.joint(joint_id).qpos[0]
        error = setpoint - current_pos
        self.integral[wheel] += error
        derivative = error - self.prev_error[wheel]
        output = (self.kp * error + self.ki * self.integral[wheel] + self.kd * derivative)
        output = np.clip(output, self.min_control, self.max_control)
        self.prev_error[wheel] = error
        return output

    def set_steering_angle(self, angles):
        for wheel in ['fl', 'fr', 'rl', 'rr']:
            if wheel in angles:
                control = self.compute_pos_pid(wheel, angles[wheel])
                actuator_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, self.steering_actuators[wheel])
                self.data.ctrl[actuator_id] = control

def navigation_controller(model, data):
    """
    Calculates desired speed and steering angle to reach the current point
    in the TRAJECTORY_POINTS list, and advances to the next point upon arrival.
    """
    global TRAJECTORY_INDEX
    
    # 1. Check if the entire trajectory is complete
    if TRAJECTORY_INDEX >= len(TRAJECTORY_POINTS):
        print("Trajectory Completed!")
        return 0.0, 0.0 
    
    # Set the current target goal from the trajectory list
    TARGET_GOAL = TRAJECTORY_POINTS[TRAJECTORY_INDEX]

    # 2. Get Current Robot Pose
    robot_x, robot_y = data.qpos[0], data.qpos[1]
    base_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'base_link')
    R = data.body(base_id).xmat.reshape(3, 3)
    robot_yaw = math.atan2(R[1, 0], R[0, 0])
    
    dx = TARGET_GOAL[0] - robot_x
    dy = TARGET_GOAL[1] - robot_y
    distance_to_goal = np.sqrt(dx**2 + dy**2)

    # 3. Check for Waypoint Arrival
    if distance_to_goal < GOAL_TOLERANCE:
        print(f"Reached Waypoint {TRAJECTORY_INDEX + 1}. Advancing to next point.")
        TRAJECTORY_INDEX += 1 # Advance to the next waypoint
        if TRAJECTORY_INDEX >= len(TRAJECTORY_POINTS):
            return 0.0, 0.0 # Stop if the final point was reached
        # Update target goal for the new point immediately
        TARGET_GOAL = TRAJECTORY_POINTS[TRAJECTORY_INDEX]
        dx = TARGET_GOAL[0] - robot_x
        dy = TARGET_GOAL[1] - robot_y
        distance_to_goal = np.sqrt(dx**2 + dy**2)


    # 4. Determine Travel Strategy (Forward vs. Reverse)
    angle_to_goal = math.atan2(dy, dx)
    relative_angle_to_goal = angle_to_goal - robot_yaw
    relative_angle_to_goal = (relative_angle_to_goal + math.pi) % (2 * math.pi) - math.pi
    
    abs_steer_angle = abs(relative_angle_to_goal)
    
    # Default is Forward travel
    sign = 1.0 
    
    # Check if angle requires reverse travel (outside the forward +/- 90 degree cone)
    if abs_steer_angle > MAX_FORWARD_ANGLE + EPSILON:
        # Travel REVERSE
        sign = -1.0
        # Flip steering angle by 180 degrees
        steering_angle_cmd = relative_angle_to_goal - math.copysign(math.pi, relative_angle_to_goal)
    elif abs_steer_angle < MAX_FORWARD_ANGLE - EPSILON:
        # Travel FORWARD (Standard Oblique Steering)
        steering_angle_cmd = relative_angle_to_goal
    else:
        # Stability Zone (Near 90/270 degrees). Maintain previous travel direction.
        # Since this path starts forward, we bias forward.
        steering_angle_cmd = math.copysign(MAX_FORWARD_ANGLE - 0.001, relative_angle_to_goal)
        sign = 1.0


    # 5. Final Command Calculation
    
    # Speed command (P-control)
    abs_speed_cmd = KP_LIN * distance_to_goal
    
    # Apply minimum cruise speed logic
    if distance_to_goal > GOAL_TOLERANCE:
         if abs_speed_cmd < MIN_CRUISE_SPEED:
             abs_speed_cmd = MIN_CRUISE_SPEED
             
    # Apply global speed limit and sign
    abs_speed_cmd = np.clip(abs_speed_cmd, 0, MAX_LIN_VEL)
    speed_cmd = sign * abs_speed_cmd

    # Limit steering angle command to physical limits
    steering_angle_cmd = np.clip(steering_angle_cmd, -MAX_STEER_RAD, MAX_STEER_RAD)

    return speed_cmd, steering_angle_cmd

def init_controller(model,data):
    global pos_pid, vel_pid
    pos_pid = AnglePIDController(model, data)
    vel_pid = VelocityPIDController(model, data)
    
  
def controller(model, data):
    """Main controller function called in the simulation loop."""
    # 1. Navigation: Determine high-level commands
    speed_cmd, steering_angle_cmd = navigation_controller(model, data)
    
    # 2. Kinematics: Convert to individual wheel targets
    wheel_velocities, steering_angles = calculate_oblique_wheel_commands(speed_cmd, steering_angle_cmd)
    
    # 3. Control: Use PID controllers to apply the targets
    if pos_pid:
        pos_pid.set_steering_angle(steering_angles)
    if vel_pid:
        vel_pid.set_wheel_vel(wheel_velocities)

def plot_final_path(path_x, path_y, trajectory_points):
    """Generates and saves the final path plot."""
    
    plt.figure(figsize=(10, 8))
    
    # 1. Plot the PLANNED TRAJECTORY (The sequence of waypoints)
    # The [:, 0] gets all X-coords, [:, 1] gets all Y-coords
    plt.plot(trajectory_points[:, 0], trajectory_points[:, 1], 
             label='Planned Trajectory', color='gray', linestyle='--', linewidth=1.5, zorder=1)
    
    # 2. Plot the ACTUAL PATH (The robot's recorded movement)
    plt.plot(path_x, path_y, 
             label='Actual Robot Path', color='blue', linewidth=2.5, zorder=2)
    
    # 3. Mark Start, End, and Waypoints
    
    # Mark Start (Point 1)
    plt.scatter(trajectory_points[0, 0], trajectory_points[0, 1], 
                color='green', s=150, zorder=5, edgecolors='black', label='Start')
    
    # Mark Waypoints (Intermediate points)
    if len(trajectory_points) > 2:
        plt.scatter(trajectory_points[1:-1, 0], trajectory_points[1:-1, 1], 
                    color='orange', s=80, zorder=4, edgecolors='black', label='Intermediate Waypoints')
    
    # Mark Final Goal
    plt.scatter(trajectory_points[-1, 0], trajectory_points[-1, 1], 
                color='red', s=150, zorder=5, marker='X', edgecolors='black', label='Final Goal')
    
    # 4. Add Labels and Formatting
    plt.title('Ranger Mini Trajectory Tracking (Oblique Steering)', fontsize=14)
    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Y Position (m)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Ensure a square plot aspect
    plt.axis('equal')
    
    # Save and show
    plt.savefig('robot_trajectory_final.png')
    print("Final trajectory graph saved as 'robot_trajectory_final.png'")
    plt.show()

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
cam.azimuth = 90
cam.elevation = -45
cam.distance = 4
cam.lookat = np.array([0.0, 0.0, 0])

#initialize the controller
init_controller(model,data)

#set the controller
#mj.set_mjcb_control(controller)

path_x = []
path_y = []


while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0/60.0):
        
        path_x.append(data.qpos[0])
        path_y.append(data.qpos[1])
        
        controller(model, data)
        mj.mj_step(model, data)

    if (data.time>=simend):
        break;

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    #print camera configuration (help to initialize the view)
    if (print_camera_config==1):
        print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
        print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()

#Plot Path
plot_final_path(path_x, path_y, TRAJECTORY_POINTS)