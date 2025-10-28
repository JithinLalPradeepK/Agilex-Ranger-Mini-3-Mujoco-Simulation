import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import math

xml_path = 'ranger_mini.xml' #xml file (assumes this is in the same folder as this file)
simend = 10 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

# Global variables to hold controller instances
pos_pid = None
vel_pid = None
target_lin_x = 0.0   # Target forward velocity (m/s)
target_ang_z = 10.0   # Target angular velocity (rad/s)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

class VelocityPIDController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # PID parameters for velocity control (tuned values)
        self.kp = 40.0  # Proportional gain
        self.ki = 0.01   # Integral gain
        self.kd = 1    # Derivative gain
        
        # Initialize error terms for each wheel
        self.prev_error = {'fl': 0.0, 'fr': 0.0, 'rl': 0.0, 'rr': 0.0}
        self.integral = {'fl': 0.0, 'fr': 0.0, 'rl': 0.0, 'rr': 0.0}
        
        # Wheel joint and actuator names
        self.wheel_joints = {
            'fl': 'fl_wheel',
            'fr': 'fr_wheel',
            'rl': 'rl_wheel',
            'rr': 'rr_wheel'
        }
        
        self.wheel_actuators = {
            'fl': 'fl_wheel_motor',
            'fr': 'fr_wheel_motor',
            'rl': 'rl_wheel_motor',
            'rr': 'rr_wheel_motor'
        }
        
        # Control limits
        self.max_control = 10.0
        self.min_control = -10.0

    def compute_vel_pid(self, wheel, setpoint):
        """Compute velocity PID control for a single wheel."""
        joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, self.wheel_joints[wheel])
        current_vel = self.data.joint(joint_id).qvel[0]  # Using velocity instead of position
        
        error = setpoint - current_vel
        self.integral[wheel] += error
        derivative = error - self.prev_error[wheel]
        
        # Compute PID output
        output = (self.kp * error + 
                 self.ki * self.integral[wheel] + 
                 self.kd * derivative)
        
        # Apply output limits
        output = np.clip(output, self.min_control, self.max_control)
        self.prev_error[wheel] = error
        
        return output
    
    def set_wheel_vel(self, velocities):
        """Set target velocities for all wheels."""
        for wheel in ['fl', 'fr', 'rl', 'rr']:
            if wheel in velocities:
                control = self.compute_vel_pid(wheel, velocities[wheel])
                actuator_id = mj.mj_name2id(
                    self.model, 
                    mj.mjtObj.mjOBJ_ACTUATOR, 
                    self.wheel_actuators[wheel]
                )
                self.data.ctrl[actuator_id] = control

class AnglePIDController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # PID parameters for steering (tuned values)
        self.kp = 10.0  # Proportional gain
        self.ki = 0.01   # Integral gain
        self.kd = 1.0   # Derivative gain
        
        # Initialize error terms
        self.prev_error = {'fl': 0.0, 'fr': 0.0, 'rl': 0.0, 'rr': 0.0}
        self.integral = {'fl': 0.0, 'fr': 0.0, 'rl': 0.0, 'rr': 0.0}
        
        # Steering joint and actuator names
        self.steering_joints = {
            'fl': 'fl_steering_joint',
            'fr': 'fr_steering_joint',
            'rl': 'rl_steering_joint',
            'rr': 'rr_steering_joint'
        }
        
        self.steering_actuators = {
            'fl': 'fl_steering',
            'fr': 'fr_steering',
            'rl': 'rl_steering',
            'rr': 'rr_steering'
        }
        
        # Control limits (in radians)
        self.max_control = math.pi/2   # 90 degrees
        self.min_control = -math.pi/2

    def compute_pos_pid(self, wheel, setpoint):
        """Compute position PID control for steering."""
        joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, self.steering_joints[wheel])
        current_pos = self.data.joint(joint_id).qpos[0]
        
        error = setpoint - current_pos
        self.integral[wheel] += error
        derivative = error - self.prev_error[wheel]
        
        # Compute PID output
        output = (self.kp * error + 
                 self.ki * self.integral[wheel] + 
                 self.kd * derivative)
        
        # Apply output limits
        output = np.clip(output, self.min_control, self.max_control)
        self.prev_error[wheel] = error
        
        return output
    
    def set_steering_angle(self, angles):
        """Set steering angles for all wheels."""
        for wheel in ['fl', 'fr', 'rl', 'rr']:
            if wheel in angles:
                control = self.compute_pos_pid(wheel, angles[wheel])
                actuator_id = mj.mj_name2id(
                    self.model, 
                    mj.mjtObj.mjOBJ_ACTUATOR, 
                    self.steering_actuators[wheel]
                )
                self.data.ctrl[actuator_id] = control


def init_controller(model,data):
    global pos_pid
    global vel_pid
    # Initialize the controllers
    pos_pid = AnglePIDController(model, data)
    vel_pid = VelocityPIDController(model, data)

    # NOTE: In a real application, target_lin_x and target_ang_z would come from 
    # a command source (like ROS cmd_vel), but here they are hardcoded.
    print(f"Controller initialized. Target motion: Linear X = {target_lin_x} m/s, Angular Z = {target_ang_z} rad/s")    


# Helper function (based on FourWheelSteeringController from the other file)
def calculate_wheel_commands(lin_x, ang_z):
    # Robot dimensions (use values from ranger_mini_PID_controller.py)
    track = 0.4
    wheel_base = 0.5
    
    wheel_velocities = {}
    steering_angles = {}

    # Pure rotation (simplified logic)
    if (abs(lin_x) < 0.01 and abs(ang_z) > 0.01):
        # Steer to turn in place
        steering_angle = np.arctan2(wheel_base, track)
        
        steering_angles = {
            'fl': steering_angle,  'fr': -steering_angle,
            'rl': -steering_angle, 'rr': steering_angle
        }
        
        # Calculate wheel speed for pure rotation
        wheel_speed = np.sqrt((track * 0.5)**2 + (wheel_base * 0.5)**2) * abs(ang_z)
        
        # Apply speed direction based on ang_z
        sign = 1 if ang_z > 0 else -1
        wheel_velocities = {
            'fl': wheel_speed * sign, 'fr': -wheel_speed * sign,
            'rl': wheel_speed * sign, 'rr': -wheel_speed * sign
        }

    # Pure translation (straight line)
    elif (abs(lin_x) > 0.01 and abs(ang_z) < 0.01):
        # Steer straight (angle 0)
        angle = 0.0
        
        steering_angles = {
            'fl': angle, 'fr': angle,
            'rl': angle, 'rr': angle
        }
        
        # Set all wheel velocities to target lin_x
        wheel_velocities = {
            'fl': lin_x, 'fr': lin_x,
            'rl': lin_x, 'rr': lin_x
        }
        
    # No motion
    else:
        wheel_velocities = {'fl': 0.0, 'fr': 0.0, 'rl': 0.0, 'rr': 0.0}
        steering_angles = {'fl': 0.0, 'fr': 0.0, 'rl': 0.0, 'rr': 0.0}


    return wheel_velocities, steering_angles

def controller(model, data):
    global pos_pid
    global vel_pid
    global target_lin_x
    global target_ang_z

    # 1. Calculate target wheel commands from desired chassis motion
    wheel_velocities, steering_angles = calculate_wheel_commands(target_lin_x, target_ang_z)
    
    # 2. Apply PID to set steering angles
    if pos_pid:
        pos_pid.set_steering_angle(steering_angles)

    # 3. Apply PID to set wheel velocities
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
# cam.azimuth = 90
# cam.elevation = -45
# cam.distance = 2
# cam.lookat = np.array([0.0, 0.0, 0])

#initialize the controller
init_controller(model,data)

#set the controller
#mj.set_mjcb_control(controller)

while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0/60.0):
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
