# PID parameters for pitch control
KP_PITCH = 500
KI_PITCH = 2
KD_PITCH = 0.1

# PID parameters for yaw control
KP_YAW = 20
KI_YAW = 0.1
KD_YAW = 5

# Control parameters
FORCE_MAGNITUDE = 1  # Adjust this value based on desired acceleration
TURNING_SPEED = 1  # Adjust for desired turning sharpness

# camera rendering parameters
CAMERA_DISTANCE = 4
CAMERA_YAW = -50
CAMERA_PITCH = -35
CAMERA_TARGET = [-2, -2, 0.5]

# terrain parameters
USE_TERRAIN = False

# target position for the PID navigation environment
TARGET_POSITION = [-3, -1, 0.5]

# maximum yaw adjustment and force for the PID navigation environment
MAX_YAW_ADJUSTMENT = 0.5
MAX_FORCE = 5