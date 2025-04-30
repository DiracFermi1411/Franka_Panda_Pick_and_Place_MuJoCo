"""
Configuration parameters for the pick and place task.
"""

# Path to the MuJoCo model file
MODEL_PATH = '/home/dheeraj/Desktop/Pick_and_Place/Simple-MuJoCo-PickNPlace/asset/panda/franka_panda_w_objs.xml'

# Joint names for the Panda arm
ARM_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7"
]

# Gripper joint names
GRIPPER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]

# End-effector body name
EEF_BODY_NAME = "panda_eef"

# Block name
BLOCK_NAME = "red_box"

# Gripper positions
GRIPPER_OPEN_POS = 0.04  # Maximum opening
GRIPPER_CLOSED_POS = 0.00 # Fully closed

# Pick and place positions
PICKUP_POS = [0.5, 0.5, 1.09]   # Position of the red box
PLACE_POS = [0, -0.5, 1.09]    # Target position

# IK parameters
IK_ERROR_THRESHOLD = 2e-2         # Acceptable error for IK
IK_MAX_ITERATIONS = 3000          # Maximum iterations for optimization
IK_POSITION_WEIGHT = 1.0          # Weight for position error in IK objective
IK_ORIENTATION_WEIGHT = 0.5       # Weight for orientation error in IK objective
IK_NUM_DIRECT_LINE_SAMPLES = 5    # Number of samples for direct line approach

# Path execution parameters
PATH_INTERPOLATION_STEPS = 30     # Steps for path interpolation
PATH_STEP_TIME = 0.1              # Time between steps (slower = more visible)

# Approach vector (negative z-axis for approaching from above)
APPROACH_VECTOR = [0, 0, -1]

# Hover height above pickup/place positions
HOVER_HEIGHT = 0.15
