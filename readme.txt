# Panda Pick and Place

A modular implementation of a pick and place operation using a Franka Panda robot in MuJoCo.

## Features

- Direct line approach for natural grasping motions
- Multi-stage optimization for inverse kinematics
- Interactive step-by-step visualization
- Modular code structure for easy extension

## Setup

1. Install the required dependencies:
```bash
pip install mujoco numpy scipy
pip install mujoco-python-viewer  # For visualization
```

2. Run the main script:
```bash
python main.py
```

## Project Structure

- `main.py`: Entry point for the application
- `config.py`: Configuration parameters
- `src/`: Source code
  - `utils/`: Utility functions
  - `kinematics/`: Forward and inverse kinematics
  - `robot/`: Robot control functions
  - `tasks/`: High-level tasks (e.g., pick and place)
- `assets/`: MuJoCo model files


Current robot pose at pickup:
Current position: [0.49999824 0.50000162 1.0899991 ]
Current orientation (quaternion): [ 5.47511640e-06  1.00000000e+00 -4.81160722e-06  3.02186012e-06]