"""
Math utility functions for robotics operations.
"""

import numpy as np

def rotation_matrix_to_quat(R):
    """
    Convert rotation matrix to quaternion (w, x, y, z)
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        quat: Quaternion as [w, x, y, z]
    """
    # Algorithm from http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        S = np.sqrt(trace+1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    
    return np.array([qw, qx, qy, qz])

def quat_to_rotation_matrix(quat):
    """
    Convert quaternion to rotation matrix
    
    Args:
        quat: Quaternion as [w, x, y, z]
        
    Returns:
        R: 3x3 rotation matrix
    """
    w, x, y, z = quat
    
    xx, xy, xz = x*x, x*y, x*z
    yy, yz, zz = y*y, y*z, z*z
    wx, wy, wz = w*x, w*y, w*z
    
    R = np.array([
        [1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)],
        [2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)]
    ])
    
    return R

def quat_multiply(q1, q2):
    """
    Multiply two quaternions
    
    Args:
        q1, q2: Quaternions in format [w, x, y, z]
        
    Returns:
        result: Quaternion product as [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])

def get_aligned_gripper_quat(approach_vec=None):
    """
    Get quaternion for gripper alignment with approach vector
    
    Args:
        approach_vec: Direction to approach from (if None, approaches from above)
        
    Returns:
        quat: Orientation quaternion [w, x, y, z]
    """
    # If no approach vector specified, approach from above (negative z)
    if approach_vec is None:
        approach_vec = np.array([0, 0, -1])  # -z direction
    else:
        # Normalize the approach vector
        approach_vec = approach_vec / np.linalg.norm(approach_vec)
    
    # We want the gripper's z-axis to align with the approach vector
    z_axis = approach_vec
    
    # Create an arbitrary vector that's not parallel to z_axis
    # If z_axis is close to [1,0,0], use [0,1,0], otherwise use [1,0,0]
    if abs(np.dot(z_axis, np.array([1, 0, 0]))) > 0.9:
        arbitrary_vec = np.array([0, 1, 0])
    else:
        arbitrary_vec = np.array([1, 0, 0])
    
    # The y_axis is perpendicular to both z_axis and arbitrary_vec
    y_axis = np.cross(z_axis, arbitrary_vec)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # The x_axis completes the right-handed coordinate system
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Construct rotation matrix with these axes
    R = np.column_stack((x_axis, y_axis, z_axis))
    
    # Convert to quaternion
    quat = rotation_matrix_to_quat(R)
    
    return quat

def interpolate_path(start_angles, end_angles, num_steps=50):
    """
    Interpolate a path between start and end joint angles
    
    Args:
        start_angles: Starting joint angles
        end_angles: Ending joint angles
        num_steps: Number of interpolation steps
        
    Returns:
        path: Array of shape (num_steps, len(start_angles)) with interpolated joint angles
    """
    path = np.zeros((num_steps, len(start_angles)))
    
    for i in range(num_steps):
        t = i / (num_steps - 1)  # Interpolation parameter [0, 1]
        path[i, :] = (1 - t) * start_angles + t * end_angles
    
    return path
