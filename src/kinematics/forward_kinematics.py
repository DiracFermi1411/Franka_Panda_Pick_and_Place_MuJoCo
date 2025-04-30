"""
Forward kinematics for the Franka Panda robot.
"""

import numpy as np
import mujoco
from src.utils.math_utils import rotation_matrix_to_quat

def forward_kinematics_with_orientation(model, data, joint_angles, joint_names=None):
    """
    Compute forward kinematics for given joint angles with orientation
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        joint_angles: Array of joint angles
        joint_names: List of joint names
        
    Returns:
        eef_pos: End-effector position
        eef_quat: End-effector orientation as quaternion [w, x, y, z]
    """
    # Store original joint positions
    original_qpos = data.qpos.copy()
    
    # If no joint names are specified, default to the Panda arm joints
    if joint_names is None:
        from config import ARM_JOINT_NAMES
        joint_names = ARM_JOINT_NAMES
    
    # Set joint angles
    for i, name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id == -1:
            raise ValueError(f"Joint '{name}' not found")
        
        qpos_addr = model.jnt_qposadr[joint_id]
        data.qpos[qpos_addr] = joint_angles[i]
    
    # Forward kinematics
    mujoco.mj_forward(model, data)
    
    # Get end-effector position
    from config import EEF_BODY_NAME
    eef_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EEF_BODY_NAME)
    if eef_id == -1:
        raise ValueError(f"End-effector body '{EEF_BODY_NAME}' not found")
    
    eef_pos = data.xpos[eef_id].copy()
    
    # Get end-effector orientation
    eef_rot_mat = data.xmat[eef_id].reshape(3, 3).copy()
    eef_quat = rotation_matrix_to_quat(eef_rot_mat)
    
    # Restore original joint positions
    data.qpos[:] = original_qpos
    mujoco.mj_forward(model, data)
    
    return eef_pos, eef_quat

def get_current_ee_pose(model, data, eef_name=None):
    """
    Get the current end-effector pose from the robot
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        eef_name: End-effector body name
    
    Returns:
        pos: Current end-effector position
        quat: Current end-effector orientation as quaternion [w, x, y, z]
    """
    if eef_name is None:
        from config import EEF_BODY_NAME
        eef_name = EEF_BODY_NAME
    
    eef_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, eef_name)
    if eef_id == -1:
        raise ValueError(f"End-effector body '{eef_name}' not found")
    
    # Get position
    pos = data.xpos[eef_id].copy()
    
    # Get orientation
    rot_mat = data.xmat[eef_id].reshape(3, 3).copy()
    quat = rotation_matrix_to_quat(rot_mat)
    
    return pos, quat

def get_block_orientation(model, data, block_name=None):
    """
    Get the orientation of the block
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        block_name: Name of the block body
        
    Returns:
        quat: Orientation quaternion [w, x, y, z]
    """
    if block_name is None:
        from config import BLOCK_NAME
        block_name = BLOCK_NAME
    
    # Try to find the block as a body
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, block_name)
    
    if body_id != -1:
        # Get orientation matrix
        rot_mat = data.xmat[body_id].reshape(3, 3).copy()
        # Convert to quaternion
        quat = rotation_matrix_to_quat(rot_mat)
        return quat
    
    # If not found as a body, try as a geom
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, block_name)
    
    if geom_id != -1:
        # Get orientation matrix
        rot_mat = data.geom_xmat[geom_id].reshape(3, 3).copy()
        # Convert to quaternion
        quat = rotation_matrix_to_quat(rot_mat)
        return quat
    
    # If not found, return identity quaternion
    print(f"Warning: Block '{block_name}' not found as body or geom")
    return np.array([1.0, 0.0, 0.0, 0.0])

def print_current_pose(model, data, eef_name=None):
    """
    Print the current end-effector pose
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        eef_name: End-effector body name
    """
    try:
        pos, quat = get_current_ee_pose(model, data, eef_name)
        print(f"Current position: {pos}")
        print(f"Current orientation (quaternion): {quat}")
    except ValueError as e:
        print(f"Error: {e}")
