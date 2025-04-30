"""
Inverse kinematics for the Franka Panda robot.
"""

import numpy as np
import mujoco
from scipy.optimize import minimize

from config import (
    ARM_JOINT_NAMES, 
    IK_NUM_DIRECT_LINE_SAMPLES,
    IK_ERROR_THRESHOLD,
    IK_MAX_ITERATIONS,
    IK_POSITION_WEIGHT,
    IK_ORIENTATION_WEIGHT
)
from src.kinematics.forward_kinematics import (
    forward_kinematics_with_orientation,
    get_current_ee_pose
)

def objective_function_with_orientation(joint_angles, model, data, target_pos, target_quat, joint_names,
                                         pos_weight=IK_POSITION_WEIGHT, ori_weight=IK_ORIENTATION_WEIGHT):
    """
    Objective function for IK with orientation control
    
    Args:
        joint_angles: Array of joint angles
        model: MuJoCo model
        data: MuJoCo data
        target_pos: Target position
        target_quat: Target orientation as quaternion [w, x, y, z]
        joint_names: List of joint names
        pos_weight: Weight for position error
        ori_weight: Weight for orientation error
        
    Returns:
        error: Weighted sum of position and orientation errors
    """
    # Compute forward kinematics
    eef_pos, eef_quat = forward_kinematics_with_orientation(model, data, joint_angles, joint_names)
    
    # Position error
    pos_error = np.sum((eef_pos - target_pos)**2)
    
    # Orientation error - using quaternion dot product
    quat_dot = np.abs(np.dot(eef_quat, target_quat))
    quat_dot = min(quat_dot, 1.0)  # Ensure it's not > 1 due to numerical error
    ori_error = 1.0 - quat_dot
    
    # Combined error
    error = pos_weight * pos_error + ori_weight * ori_error
    
    return error

def get_joint_limits(model, joint_names):
    """
    Get joint limits for specified joints
    
    Args:
        model: MuJoCo model
        joint_names: List of joint names
        
    Returns:
        bounds: List of (min, max) tuples for each joint
    """
    bounds = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id == -1:
            raise ValueError(f"Joint '{name}' not found")
        
        # Get joint limits from the model
        lower_limit = model.jnt_range[joint_id][0]
        upper_limit = model.jnt_range[joint_id][1]
        bounds.append((lower_limit, upper_limit))
    
    return bounds

def generate_direct_line_guesses(model, data, current_pos, target_pos, num_samples=IK_NUM_DIRECT_LINE_SAMPLES):
    """
    Generate joint angle guesses along a direct line from current position to target position
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        current_pos: Current end-effector position
        target_pos: Target position
        num_samples: Number of samples along the line
        
    Returns:
        joint_guesses: List of joint angle arrays
    """
    # Store original joint positions
    original_qpos = data.qpos.copy()
    
    # Get current joint angles for Panda arm
    joint_names = ARM_JOINT_NAMES
    
    current_angles = np.zeros(len(joint_names))
    for i, name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_addr = model.jnt_qposadr[joint_id]
        current_angles[i] = data.qpos[qpos_addr]
    
    # Create position samples along the direct line
    positions = []
    for i in range(num_samples):
        t = i / (num_samples - 1)  # 0 to 1
        pos = current_pos * (1 - t) + target_pos * t
        positions.append(pos)
    
    # Initialize list of joint guesses
    joint_guesses = [current_angles]  # Start with current angles
    
    print("Generating direct line approach guesses...")
    
    # For each position along the line (skip the first since it's the current position)
    for i in range(1, num_samples):
        pos = positions[i]
        print(f"Generating guess {i}/{num_samples-1} at position {pos}")
        
        # Try simple IK for this intermediate position
        # We'll use a simple Jacobian-based approach for these guesses
        
        # Use previous guess as starting point
        prev_guess = joint_guesses[-1].copy()
        
        # Apply the previous guess to get a starting point
        for j, name in enumerate(joint_names):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_addr = model.jnt_qposadr[joint_id]
            data.qpos[qpos_addr] = prev_guess[j]
        
        mujoco.mj_forward(model, data)
        
        # Get end-effector ID
        from config import EEF_BODY_NAME
        eef_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EEF_BODY_NAME)
        
        # Simple IK iteration for getting a reasonable guess
        max_iter = 20
        step_size = 0.1
        
        for _ in range(max_iter):
            # Get current end-effector position
            mujoco.mj_forward(model, data)
            curr_pos = data.xpos[eef_id].copy()
            
            # Compute error
            error = pos - curr_pos
            error_norm = np.linalg.norm(error)
            
            if error_norm < 0.01:  # Good enough for a guess
                break
            
            # Get Jacobian
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, None, eef_id)
            
            # Extract columns for arm joints
            jac = np.zeros((3, len(joint_names)))
            for j, name in enumerate(joint_names):
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                jac[:, j] = jacp[:, model.jnt_dofadr[joint_id]]
            
            # Pseudo-inverse Jacobian method
            lambda_val = 0.1  # Damping factor
            jac_t = jac.T
            delta_q = jac_t @ np.linalg.solve(jac @ jac_t + lambda_val * np.eye(3), error)
            
            # Update joint angles
            for j, name in enumerate(joint_names):
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                qpos_addr = model.jnt_qposadr[joint_id]
                data.qpos[qpos_addr] += step_size * delta_q[j]
                
                # Apply joint limits
                if data.qpos[qpos_addr] < model.jnt_range[joint_id][0]:
                    data.qpos[qpos_addr] = model.jnt_range[joint_id][0]
                elif data.qpos[qpos_addr] > model.jnt_range[joint_id][1]:
                    data.qpos[qpos_addr] = model.jnt_range[joint_id][1]
        
        # Extract the guess
        guess = np.zeros(len(joint_names))
        for j, name in enumerate(joint_names):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_addr = model.jnt_qposadr[joint_id]
            guess[j] = data.qpos[qpos_addr]
        
        joint_guesses.append(guess)
    
    # Restore original positions
    data.qpos[:] = original_qpos
    mujoco.mj_forward(model, data)
    
    return joint_guesses

def inverse_kinematics_with_orientation(model, data, target_pos, target_quat, joint_names=None, 
                                        max_iter=IK_MAX_ITERATIONS, error_threshold=IK_ERROR_THRESHOLD, 
                                        initial_angles=None):
    """
    Compute inverse kinematics with orientation control
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        target_pos: Target position [x, y, z]
        target_quat: Target orientation as quaternion [w, x, y, z]
        joint_names: List of joint names
        max_iter: Maximum number of iterations
        error_threshold: Maximum acceptable error
        initial_angles: Initial joint angles to start from (if None, use current)
        
    Returns:
        success: Boolean indicating if IK was successful
        joint_angles: Array of joint angles
        achieved_error: Final error value
    """
    # If no joint names are specified, default to the Panda arm joints
    if joint_names is None:
        joint_names = ARM_JOINT_NAMES
    
    # Get joint limits
    bounds = get_joint_limits(model, joint_names)
    
    # Get current end-effector position
    current_pos, _ = get_current_ee_pose(model, data)
    
    # Generate direct line approach guesses
    direct_line_guesses = generate_direct_line_guesses(model, data, current_pos, target_pos)
    
    # Get initial joint angles if not provided
    if initial_angles is None:
        initial_angles = np.zeros(len(joint_names))
        for i, name in enumerate(joint_names):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_addr = model.jnt_qposadr[joint_id]
            initial_angles[i] = data.qpos[qpos_addr]
    
    # Try all guesses and find the best one
    best_error = float('inf')
    best_solution = None
    
    # First try our direct line guesses
    for i, guess in enumerate(direct_line_guesses):
        print(f"Trying direct line guess {i+1}/{len(direct_line_guesses)}")
        
        result = minimize(
            objective_function_with_orientation, 
            guess, 
            args=(model, data, target_pos, target_quat, joint_names),
            method='Powell',
            bounds=bounds,
            options={'maxiter': max_iter // 2, 'disp': True}  # Use fewer iterations per guess
        )
        
        if result.fun < best_error:
            best_error = result.fun
            best_solution = result.x
            print(f"Found better solution with error: {best_error}")
    
    # Then refine the best solution with a longer optimization
    if best_solution is not None:
        print("Refining best solution...")
        result = minimize(
            objective_function_with_orientation, 
            best_solution, 
            args=(model, data, target_pos, target_quat, joint_names),
            method='Powell',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': True}
        )
        
        best_error = result.fun
        best_solution = result.x
    
    # Check if the optimization was successful
    success = best_error < error_threshold
    
    if success:
        print(f"IK successful with error: {best_error}")
    else:
        print(f"IK failed with error: {best_error}")
    
    return success, best_solution, best_error
