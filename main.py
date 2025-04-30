"""
Main entry point for the Franka Panda pick and place application.
"""

import time
import mujoco
import numpy as np
from config import MODEL_PATH, PICKUP_POS, PLACE_POS
from src.utils.viewer import create_viewer, update_viewer, close_viewer
from src.utils.input_utils import wait_for_input
from src.tasks.pick_place import pick_and_place_with_direct_approach

def main():
    """Main function to run the pick and place task."""
    # Load the Franka Panda model
    try:
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        
        # Set physics parameters for better contact handling
        model.opt.iterations = 50       # More solver iterations
        model.opt.tolerance = 1e-10     # Lower tolerance
        model.opt.gravity[2] = -9.81    # Make sure gravity is properly set
        model.opt.timestep = 0.002      # Smaller timestep
        
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading MuJoCo model: {e}")
        print(f"Make sure the model file exists at: {MODEL_PATH}")
        return False
    # stabilize_scene(model, data)
    # Create viewer
    viewer = create_viewer(model, data)
    
    # Define pickup and place positions as numpy arrays
    pickup_pos = np.array(PICKUP_POS)
    place_pos = np.array(PLACE_POS)
    
    try:
        # Show the initial state
        if viewer is not None:
            update_viewer(viewer)
            print("Initial state shown. Press Enter to begin...")
            wait_for_input()
        
        # Perform pick and place operation with direct line approach
        success = pick_and_place_with_direct_approach(
            model, data, pickup_pos, place_pos, viewer=viewer
        )
        
        # Keep the viewer open to observe the final state
        if viewer is not None and success:
            print("Simulation complete. Press Ctrl+C to exit...")
            while update_viewer(viewer):
                time.sleep(0.1)
                
        return success
    
    except KeyboardInterrupt:
        print("Operation interrupted by user")
        return False
    
    finally:
        if viewer is not None:
            close_viewer(viewer)

if __name__ == "__main__":
    success = main()
    if success:
        print("Pick and place task completed successfully.")
    else:
        print("Pick and place task was not completed successfully.")
