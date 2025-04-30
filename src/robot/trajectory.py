"""
Trajectory functions for executing robot paths.
"""

import time
import numpy as np
from config import PATH_INTERPOLATION_STEPS, PATH_STEP_TIME
from src.utils.math_utils import interpolate_path
from src.robot.robot_control import set_joint_angles
from src.utils.viewer import update_viewer
from src.utils.input_utils import wait_for_input
from src.utils.viewer import run_viewer_while_waiting

def execute_path_with_viewer(model, data, path, joint_names=None, viewer=None, step_time=PATH_STEP_TIME, 
                           pause_at_end=False, message=None):
    """
    Execute a path of joint angles with visualization and optional pause
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        path: Array of joint angles to follow
        joint_names: List of joint names
        viewer: MuJoCo viewer instance
        step_time: Time to wait between steps (seconds)
        pause_at_end: Whether to pause at the end of the path
        message: Message to display when pausing
    """
    for i, angles in enumerate(path):
        # Set joint angles
        set_joint_angles(model, data, angles, joint_names)
        
        # Update viewer if available
        if viewer is not None:
            update_viewer(viewer)
        
        # Wait between steps (slower simulation)
        time.sleep(step_time)
        
        # Print progress every 10%
        if i % max(1, len(path) // 10) == 0:
            print(f"Progress: {i+1}/{len(path)} steps")
    
    # Pause at the end if requested
    if pause_at_end:
        if message:
            print(message)
        
        if viewer is not None:
            def wait_func():
                wait_for_input("Viewing the robot state. Press Enter to continue...")
            
            run_viewer_while_waiting(viewer, wait_func)
        else:
            wait_for_input()

def move_to_joint_target(model, data, target_joints, current_joints=None, 
                        num_steps=PATH_INTERPOLATION_STEPS, viewer=None, 
                        step_time=PATH_STEP_TIME, pause_at_end=False, message=None,
                        joint_names=None):
    """
    Move the robot from current joint angles to target joint angles
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        target_joints: Target joint angles
        current_joints: Current joint angles (if None, use current robot state)
        num_steps: Number of interpolation steps
        viewer: MuJoCo viewer instance
        step_time: Time to wait between steps
        pause_at_end: Whether to pause at the end of the path
        message: Message to display when pausing
        joint_names: List of joint names
    """
    # If current joints not provided, get from robot
    if current_joints is None:
        from src.robot.robot_control import get_current_joint_angles
        current_joints = get_current_joint_angles(model, data, joint_names)
    
    # Interpolate path
    path = interpolate_path(current_joints, target_joints, num_steps)
    
    # Execute path
    execute_path_with_viewer(model, data, path, joint_names, viewer, step_time, pause_at_end, message)
    
    return True
