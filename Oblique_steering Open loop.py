import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import math
import matplotlib.pyplot as plt

xml_path = 'ranger_mini.xml' #xml file (assumes this is in the same folder as this file)
simend = 5 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

# --- NAVIGATION & CONTROL GLOBALS ---
TARGET_GOAL = np.array([2, 1]) # The (x, y) point the robot will navigate to
GOAL_TOLERANCE = 0.1 # Distance in meters to consider the goal reached
WHEEL_RADIUS = 0.09 # Approximate wheel radius in meters for speed conversion

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

dx = TARGET_GOAL[0] 
dy = TARGET_GOAL[1] 
angle_to_goal = math.atan2(dy, dx)
speed_cmd = 10
# Normalize the angle to be within [-pi, pi]
steering_angle_cmd = (angle_to_goal + math.pi) % (2 * math.pi) - math.pi
print("angle_to_goal: ",angle_to_goal*57.297, " steering_angle_cmd: ", steering_angle_cmd*57.297)
wheel_velocities, steering_angles = calculate_oblique_wheel_commands(speed_cmd, steering_angle_cmd)

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
        self.kp = 40.0
        self.ki = 0.01
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

def init_controller(model,data):
    global pos_pid, vel_pid
    pos_pid = AnglePIDController(model, data)
    vel_pid = VelocityPIDController(model, data)
    print(f"PID controllers initialized. Navigating to goal: {TARGET_GOAL}")
  
def controller(model, data):

        # 3. Control: Use PID controllers to apply the targets
    if pos_pid:
        pos_pid.set_steering_angle(steering_angles)
    if vel_pid:
        vel_pid.set_wheel_vel(wheel_velocities)



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
cam.distance = 8
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
plt.figure(figsize=(8, 8))
plt.plot(path_x, path_y, label='Robot Path')
plt.scatter([0], [0], color='green', s=100, zorder=5, label='Start (0,0)')
plt.scatter([TARGET_GOAL[0]], [TARGET_GOAL[1]], color='red', s=100, zorder=5, label=f'Goal ({TARGET_GOAL[0]},{TARGET_GOAL[1]})')
plt.title('Path Followed by the Robot')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
plt.savefig('robot_path.png')
