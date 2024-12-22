# Technovanza

## Warehouse Management and Object Tracking Project

This repository contains code and resources for a warehouse management project, focusing on object detection, tracking, and determining position and orientation. The project is divided into two main tasks:

---

### Task 1: Object Detection and Pose Estimation

This task focuses on detecting objects (specifically boxes), determining their position and orientation in images, and potentially tracking them over time.

#### Folder Structure:

```
Technovanza/
├── task_1/
│   ├── given_task_soln/
│   │   └── images/      # Input images for the given task
│   ├── output_pics/     # Output images with detected objects and pose information
│   ├── program/         # Python scripts for object detection and pose estimation
│   │   └── finding_position_orientation.py
│   └── real_world_soln/ # Real-world application of task 1 in a warehouse setting
│       ├── images/      # Real-world warehouse images
│       │   └── top_view_warehouse.png
│       ├── output_pics/ # Output images with real-time tracking and warehouse layout
│       │   ├── realtime_tracking.jpeg
│       │   └── warehouse_output.png
│       └── program/     # Scripts for real-time tracking and warehouse visualization
│           ├── calibrated_position_determination.py
│           └── warehouse_box_realtime.py
```

#### Scripts:

- **`finding_position_orientation.py`**: Implements the core logic for detecting objects in images and estimating their 3D position and orientation. Likely uses computer vision techniques such as:
  - Object Detection (e.g., using YOLO, SSD, or other detectors).
  - Pose Estimation (e.g., using Perspective-n-Point (PnP) algorithms).

- **`calibrated_position_determination.py`**: Handles camera calibration and precise object position determination in a warehouse setting.

- **`warehouse_box_realtime.py`**: Implements real-time object tracking and visualization within the warehouse.

#### Usage:

1. Place input images in `task_1/given_task_soln/images/`.
2. Run `finding_position_orientation.py`. Output images will be saved in `task_1/output_pics/`.
3. For real-world applications:
   - Place warehouse images in `task_1/real_world_soln/images/`.
   - Run scripts in `task_1/real_world_soln/program/`.

---

### Task 2: Maze Solving

This task focuses on finding the shortest path through a maze represented as an image.

#### Folder Structure:

```
Technovanza/
└── task_2/
    ├── given_task_soln/
    │   └── images/      # Input maze image
    │       └── mazeupd.jpg
    ├── output_pics/     # Output image with the shortest path marked
    │   └── maze_shortest_path.png
    └── program/         # Python script for maze solving
    │   └── maze_soln.py
    └── real_world_soln/ # Real-world warehouse navigation
        ├── images/      # Images of warehouse environment and obstacles
        │   ├── grid_large_object.jpeg
        │   ├── grid_small_object.jpeg
        │   ├── major_obstacle.jpeg
        │   ├── minor_obstacle.jpeg
        │   └── warehouse-top_view_1.jpg
        ├── output_pics/ # Shortest path visualization in the warehouse
        │   └── warehouse_shortest_path.png
        └── program/     # Python script for warehouse navigation
            └── warehouse_soln.py
```

#### Scripts:

- **`maze_soln.py`**: Solves mazes by finding the shortest path. Likely employs algorithms like A* Search. The script processes the maze image, finds the path, and visualizes the result.
- **`warehouse_soln.py`**: Extends maze-solving logic to navigate a warehouse, considering obstacles like large and small objects.

#### Usage:

**Maze Solving:**

1. Place the input maze image (`mazeupd.jpg`) in `task_2/given_task_soln/images/`.
2. Run `maze_soln.py` from `task_2/given_task_soln/program/`:
   ```bash
   cd task_2/given_task_soln/program/
   python maze_soln.py
   ```

**Warehouse Navigation:**

1. Place warehouse layout and obstacle images in `task_2/real_world_soln/images/`.
2. Run `warehouse_soln.py` from `task_2/real_world_soln/program/`:
   ```bash
   cd task_2/real_world_soln/program/
   python warehouse_soln.py
   ```

---

### Dependencies:

- **Python Libraries**:
  - OpenCV (`cv2`): For image processing.
  - NumPy (`numpy`): For numerical operations.
- Install dependencies using:
  ```bash
  pip install opencv-python numpy
  ```

---

### Notes:

1. **Camera Calibration**:
   - Task 1 assumes the camera is calibrated. If using your own images, update calibration parameters in the script.
2. **Troubleshooting**:
   - If an image cannot be read, verify the file path and ensure the image exists.
   - Use absolute paths if necessary.
3. **Output**:
   - Task 1: Detected objects are marked with bounding boxes, and their positions (X, Y, Z) are labeled in the output.
   - Task 2: Shortest paths are visualized on the maze or warehouse layout images.

---


