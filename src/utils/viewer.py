"""
Viewer utilities for MuJoCo visualization.
"""

import time

# Try different viewers based on what's available
try:
    # Try mujoco-python-viewer first
    from mujoco_viewer import MujocoViewer
    HAS_VIEWER = True
    
    def create_viewer(model, data):
        """Create a MuJoCo viewer using mujoco-python-viewer."""
        return MujocoViewer(model, data)
    
    def update_viewer(viewer):
        """Update the viewer and return whether it's still running."""
        viewer.render()
        return viewer.is_alive
    
    def close_viewer(viewer):
        """Close the viewer."""
        viewer.close()
    
except ImportError:
    try:
        # Try older mujoco-py viewer
        import mujoco_py
        HAS_VIEWER = True
        
        def create_viewer(model, data):
            """Create a MuJoCo viewer using mujoco-py."""
            sim = mujoco_py.MjSim(model)
            viewer = mujoco_py.MjViewer(sim)
            return viewer
        
        def update_viewer(viewer):
            """Update the viewer and always return True (no is_alive check in mujoco-py)."""
            viewer.render()
            return True
        
        def close_viewer(viewer):
            """No specific close function in mujoco-py."""
            pass
            
    except ImportError:
        print("WARNING: No MuJoCo viewer found. Visualization disabled.")
        HAS_VIEWER = False
        
        def create_viewer(model, data):
            """Dummy function that returns None when no viewer is available."""
            print("Viewer not available. Install mujoco-python-viewer package.")
            return None
        
        def update_viewer(viewer):
            """Dummy function that always returns False when no viewer is available."""
            return False
        
        def close_viewer(viewer):
            """Dummy close function when no viewer is available."""
            pass

def run_viewer_while_waiting(viewer, wait_func):
    """
    Run the viewer while waiting for a function to complete.
    
    Args:
        viewer: MuJoCo viewer instance
        wait_func: Function to execute while viewer is running
    """
    if viewer is not None and HAS_VIEWER:
        import threading
        
        # Create a thread for waiting
        wait_thread = threading.Thread(target=wait_func)
        wait_thread.daemon = True
        wait_thread.start()
        
        # Keep updating the viewer while waiting
        while wait_thread.is_alive():
            update_viewer(viewer)
            time.sleep(0.01)
    else:
        # If no viewer, just execute the wait function
        wait_func()
