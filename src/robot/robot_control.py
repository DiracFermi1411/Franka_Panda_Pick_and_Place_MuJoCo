"""
Robot control functions for the Franka Panda robot.
"""

import mujoco
import numpy as np
from config import ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS

def set_joint_angles(model, data, joint_angles, joint_names=None):
    """
    Set joint angles for the robot
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        joint_angles: Array of joint angles
        joint_names: List of joint names
    """
    if joint_names is None:
        joint_names = ARM_JOINT_NAMES
    
    for i, name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id == -1:
            print(f"Warning: Joint {name} not found")
            continue
        
        qpos_addr = model.jnt_qposadr[joint_id]
        data.qpos[qpos_addr] = joint_angles[i]
    
    # Update the model
    mujoco.mj_forward(model, data)

def control_gripper(model, data, gripper_open, gripper_joint_names=None):
    """
    Control the gripper (open or close)
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        gripper_open: Boolean, True to open, False to close
        gripper_joint_names: List of gripper joint names
    """
    if gripper_joint_names is None:
        gripper_joint_names = GRIPPER_JOINT_NAMES
    
    # Set gripper positions
    gripper_pos = GRIPPER_OPEN_POS if gripper_open else GRIPPER_CLOSED_POS
    
    for name in gripper_joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id == -1:
            print(f"Warning: Gripper joint {name} not found")
            continue
        
        qpos_addr = model.jnt_qposadr[joint_id]
        
        # Set the appropriate value based on the joint
        if name == "panda_finger_joint1":
            data.qpos[qpos_addr] = gripper_pos
        elif name == "panda_finger_joint2":
            data.qpos[qpos_addr] = -gripper_pos  # Right finger moves in opposite direction
    
    # Update the model
    mujoco.mj_forward(model, data)

def set_gripper(model, data, option="open"):
    """
    Set the gripper to a predefined position by directly modifying joint positions
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        option: "open" for fully open gripper, "close" for fully closed gripper
    """
    # Get joint IDs for the finger joints
    finger1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "panda_finger_joint1")
    finger2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "panda_finger_joint2")
    
    if finger1_id == -1 or finger2_id == -1:
        print("Warning: Gripper joints not found")
        return
    
    # Get addresses in qpos
    finger1_addr = model.jnt_qposadr[finger1_id]
    finger2_addr = model.jnt_qposadr[finger2_id]
    
    # Set gripper positions based on option
    if option == "open":
        # Set to maximum opening positions
        data.qpos[finger1_addr] = 0.04  # Max open value 
        data.qpos[finger2_addr] = -0.04  # Max open value (negative for second finger)
    elif option == "close":
        # Set to minimum (closed) positions
        data.qpos[finger1_addr] = 0.0
        data.qpos[finger2_addr] = 0.0
    
    # Update the simulation
    mujoco.mj_forward(model, data)
    
    # Run several simulation steps to stabilize
    for _ in range(50):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)


def control_gripper_until_contact(model, data, target_width=None, max_steps=200, step_size=0.0005, 
                                 gripper_joint_names=None, force_threshold=1.0, target_object="red_box"):
    """
    Gradually close the gripper until it makes contact with an object or reaches target width
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        target_width: Target width to reach (if None, close until contact)
        max_steps: Maximum number of closing steps
        step_size: How much to close the gripper in each step
        gripper_joint_names: List of gripper joint names
        force_threshold: Contact force threshold to detect object contact
        target_object: Name of the object to grasp
        
    Returns:
        final_width: The final gripper width achieved
        contacted: Whether contact was detected
    """
    if gripper_joint_names is None:
        from config import GRIPPER_JOINT_NAMES
        gripper_joint_names = GRIPPER_JOINT_NAMES
    
    # Ensure max_steps is an integer
    max_steps_int = int(max_steps) if not isinstance(max_steps, list) else 200
    
    # Constants from gripper geometry
    FINGER_TIP_OFFSET = 0.017  # Finger base separation
    target_width = 0.04 + 0.005  # Block width + small margin    
    # Start with gripper fully open
    from config import GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS
    current_joint_pos = GRIPPER_OPEN_POS
    contacted = False
    
    print("Gradually closing gripper until contact or target width...")
    
    # Get the joint IDs and addresses
    joint_ids = {}
    joint_addrs = {}
    for name in gripper_joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id == -1:
            print(f"Warning: Gripper joint {name} not found")
            continue
        joint_ids[name] = joint_id
        joint_addrs[name] = model.jnt_qposadr[joint_id]
    
    # Gradually close the gripper
    for step in range(max_steps_int):
        # Reduce joint position
        current_joint_pos -= step_size
        if current_joint_pos < GRIPPER_CLOSED_POS:
            current_joint_pos = GRIPPER_CLOSED_POS
        
        # Calculate current width
        current_width = (current_joint_pos * 2) + FINGER_TIP_OFFSET
        
        # Check if we've reached target width
        if target_width is not None and current_width <= target_width:
            print(f"Reached target width: {target_width:.4f}m")
            break
        
        # Apply joint positions
        for name in gripper_joint_names:
            if name not in joint_addrs:
                continue
                
            if name == "panda_finger_joint1":
                data.qpos[joint_addrs[name]] = current_joint_pos
            elif name == "panda_finger_joint2":
                data.qpos[joint_addrs[name]] = -current_joint_pos
        
        # Step simulation and update
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        
        # Check for contacts
        finger1_contacted = False
        finger2_contacted = False
        
        for i in range(data.ncon):
            contact = data.contact[i]
            geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            
            # Handle case where name is None
            geom1 = geom1 if geom1 is not None else f"geom{contact.geom1}"
            geom2 = geom2 if geom2 is not None else f"geom{contact.geom2}"
            
            # Check for contact with collision geoms
            # Make sure we're detecting contact with collision geoms (not visual geoms)
            finger1_in_contact = (("finger1" in str(geom1) and "_collision" in str(geom1)) or 
                                 ("finger1" in str(geom2) and "_collision" in str(geom2)))
                                 
            finger2_in_contact = (("finger2" in str(geom1) and "_collision" in str(geom1)) or 
                                 ("finger2" in str(geom2) and "_collision" in str(geom2)))
                                 
            # Check if object is involved in the contact
            object_involved = ((target_object in str(geom1)) or (target_object in str(geom2)))
            
            # Update contact flags
            if finger1_in_contact and object_involved:
                finger1_contacted = True
                print(f"Left finger contacted {target_object}")
                
            if finger2_in_contact and object_involved:
                finger2_contacted = True
                print(f"Right finger contacted {target_object}")
            
            # If both fingers contacted the object, we have a good grasp
            if finger1_contacted and finger2_contacted:
                print(f"Both fingers in contact with {target_object} at width: {current_width:.4f}m")
                contacted = True
                break
        
        # Check if we need to exit the loop
        if contacted:
            break
            
        # If we've reached minimum gripper position, stop
        if current_joint_pos == GRIPPER_CLOSED_POS:
            print(f"Reached minimum gripper width: {current_width:.4f}m")
            break
    
    # Run a few more steps to stabilize
    for _ in range(10):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    
    # Report final state
    final_width = (current_joint_pos * 2) + FINGER_TIP_OFFSET
    if not contacted and target_width is None:
        print(f"No contact detected. Final width: {final_width:.4f}m")
    
    return final_width, contacted

def get_current_joint_angles(model, data, joint_names=None):
    """
    Get current joint angles
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        joint_names: List of joint names
        
    Returns:
        joint_angles: Array of current joint angles
    """
    if joint_names is None:
        joint_names = ARM_JOINT_NAMES
    
    joint_angles = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id == -1:
            print(f"Warning: Joint {name} not found")
            continue
        
        qpos_addr = model.jnt_qposadr[joint_id]
        joint_angles.append(data.qpos[qpos_addr])
    
    return joint_angles
