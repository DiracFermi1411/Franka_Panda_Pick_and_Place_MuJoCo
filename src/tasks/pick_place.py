"""
Pick and place task implementation for the Franka Panda robot.
"""

import numpy as np
from config import (
    ARM_JOINT_NAMES, 
    GRIPPER_JOINT_NAMES, 
    EEF_BODY_NAME,
    HOVER_HEIGHT,
    APPROACH_VECTOR
)
from src.utils.input_utils import wait_for_input
from src.utils.math_utils import get_aligned_gripper_quat
from src.kinematics.forward_kinematics import (
    print_current_pose, 
    get_current_ee_pose
)
from src.kinematics.inverse_kinematics import inverse_kinematics_with_orientation
from src.robot.robot_control import control_gripper,control_gripper_until_contact, set_joint_angles
from src.robot.trajectory import move_to_joint_target

def pick_and_place_with_direct_approach(model, data, pickup_pos, place_pos, error_threshold=None, viewer=None):
    """
    Complete pick and place operation with direct line approach
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        pickup_pos: Position to pick up object [x, y, z]
        place_pos: Position to place object [x, y, z]
        error_threshold: Maximum acceptable error for IK
        viewer: MuJoCo viewer instance
    """
    # Use default error threshold if not specified
    if error_threshold is None:
        from config import IK_ERROR_THRESHOLD
        error_threshold = IK_ERROR_THRESHOLD
    
    # Wait for user to start the simulation
    print("\n=== INTERACTIVE PICK AND PLACE SIMULATION WITH DIRECT LINE APPROACH ===")
    print("This simulation will run step by step with pauses between major steps.")
    print("The gripper will use a direct line approach to each target position.")
    wait_for_input("Press Enter to start the simulation...")
    
    # Define joint names for the arm and gripper
    arm_joints = ARM_JOINT_NAMES
    gripper_joints = GRIPPER_JOINT_NAMES
    
    # Compute the gripper orientation for approach
    # We want the gripper to approach from above (-z direction)
    gripper_quat = get_aligned_gripper_quat(np.array(APPROACH_VECTOR))
    print(f"Aligned gripper orientation (quaternion): {gripper_quat}")
    
    # 1. Move to a starting pose (slightly above pickup position)
    hover_pos = np.array(pickup_pos).copy()
    hover_pos[2] += HOVER_HEIGHT  # Height above the pickup position
    
    print("\nStep 1: Moving to hover position with direct line approach")
    success, hover_joints, error = inverse_kinematics_with_orientation(
        model, data, hover_pos, gripper_quat, arm_joints, error_threshold=error_threshold
    )
    
    if not success:
        print(f"Warning: Could not reach hover position exactly, error: {error}")
    
    # Open gripper
    print("Opening gripper...")
    control_gripper(model, data, True, gripper_joints)
    
    # Move to hover position
    set_joint_angles(model, data, hover_joints, arm_joints)
    
    print("Current robot pose:")
    print_current_pose(model, data)
    
    # Pause to observe initial position
    if viewer is not None:
        wait_for_input("Initial position reached. Press Enter to continue to pickup position...")
    
    # 2. Move down to pickup position
    print("\nStep 2: Moving down to pickup position")
    success, pickup_joints, error = inverse_kinematics_with_orientation(
        model, data, pickup_pos, gripper_quat, arm_joints, 
        error_threshold=error_threshold
    )
    
    if not success:
        print(f"Warning: Could not reach pickup position exactly, error: {error}")
    
    # Move from hover to pickup using trajectory
    move_to_joint_target(
        model, data, 
        pickup_joints, 
        hover_joints, 
        viewer=viewer, 
        pause_at_end=True, 
        message="Reached pickup position."
    )
    
    print("Current robot pose at pickup:")
    print_current_pose(model, data)
    
    # 3. Close gripper to grab object
    print("\nStep 3: Closing gripper to grab object")
    block_width = 0.02  # Width of your block from config
    stopping_width = block_width * 0.95  # Stop slightly tighter than block width
    final_width, contacted = control_gripper_until_contact(
        model, data, 
        target_width=0.035,  # Slightly less than block width
        target_object="red_box"
    )

    # Check if grasp was successful
    if contacted:
        print("Object successfully grasped")
    else:
        print("Warning: No contact detected, grasp may be unsuccessful")
    
    # # 4. Move back up to hover position
    # print("\nStep 4: Moving back up with object")
    
    # # Move from pickup to hover using trajectory
    # move_to_joint_target(
    #     model, data, 
    #     hover_joints, 
    #     pickup_joints, 
    #     viewer=viewer
    # )
    
    # # 5. Move to hover position above place position
    # place_hover_pos = np.array(place_pos).copy()
    # place_hover_pos[2] += HOVER_HEIGHT  # Height above the place position
    
    # print("\nStep 5: Moving to hover position above place position")
    # success, place_hover_joints, error = inverse_kinematics_with_orientation(
    #     model, data, place_hover_pos, gripper_quat, arm_joints, 
    #     error_threshold=error_threshold
    # )
    
    # if not success:
    #     print(f"Warning: Could not reach place hover position exactly, error: {error}")
    
    # # Move from pickup hover to place hover using trajectory
    # move_to_joint_target(
    #     model, data, 
    #     place_hover_joints, 
    #     hover_joints, 
    #     viewer=viewer, 
    #     pause_at_end=True, 
    #     message="Reached hover position above place position."
    # )
    
    # 6. Move down to place position
    print("\nStep 6: Moving down to place position")
    success, place_joints, error = inverse_kinematics_with_orientation(
        model, data, place_pos, gripper_quat, arm_joints, 
        error_threshold=error_threshold
    )
    
    if not success:
        print(f"Warning: Could not reach place position exactly, error: {error}")
    
    # Move from place hover to place using trajectory
    move_to_joint_target(
        model, data, 
        place_joints, 
        
        , 
        viewer=viewer
    )
    
    print("Current robot pose at place:")
    print_current_pose(model, data)
    
    # 7. Open gripper to release object
    print("\nStep 7: Opening gripper to release object")
    control_gripper(model, data, True, gripper_joints)
    
    # Pause to observe gripper opening
    if viewer is not None:
        wait_for_input("Object released. Press Enter to continue...")
    
    # 8. Move back up to final hover position
    print("\nStep 8: Moving back up from place position")
    
    # Move from place to place hover using trajectory
    move_to_joint_target(
        model, data, 
        place_hover_joints, 
        place_joints, 
        viewer=viewer, 
        pause_at_end=True, 
        message="Reached final position."
    )
    
    print("\nPick and place operation completed!")
    return True
