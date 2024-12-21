import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq

# Helper function: A* algorithm
def a_star(grid_map, start, goal):
    rows, cols = grid_map.shape
    # Include diagonal movements
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Up, Down, Left, Right
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonals: Top-left, Top-right, Bottom-left, Bottom-right
    open_set = []
    heapq.heappush(open_set, (0, start))  # (cost, (row, col))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            return reconstruct_path(came_from, current)
        
        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)
            
            # Check if the neighbor is within bounds and is not an obstacle
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid_map[neighbor] == 1:
                # Calculate step cost (√2 for diagonal steps, 1 otherwise)
                step_cost = np.sqrt(2) if abs(dr) == 1 and abs(dc) == 1 else 1
                tentative_g_score = g_score[current] + step_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return []  # Return an empty path if no path is found

# Heuristic function: Chebyshev distance
def heuristic(pos, goal):
    return max(abs(pos[0] - goal[0]), abs(pos[1] - goal[1]))

# Reconstruct the path from the 'came_from' dictionary
def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

# MAP GENERATION --------------------------------------------------------------------------------------------------------------------------------------------
# Step 1: Load the Image
image = cv2.imread("warehousePics/mazeupd.jpg", cv2.IMREAD_GRAYSCALE)

# Step 2: Preprocess the Image
# Resize the image for grid resolution (e.g., 100x100 grid)
image_resized = cv2.resize(image, (2000, 2000))

# Threshold the image to identify free and occupied spaces
_, binary_map = cv2.threshold(image_resized, 128, 255, cv2.THRESH_BINARY)

# Morphological operations to clean up noise
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)

# Step 3: Generate a Grid-Based Map
# Convert binary image to a grid map (1 = free space, 0 = obstacle)
grid_map = np.where(binary_map == 255, 1, 0)  # 255 -> Free space (1), 0 -> Obstacle (0)

# CURRENT & GOAL COORDINATES --------------------------------------------------------------------------------------------------------------------------------------------
# Define current and goal coordinates
current_position = (1979, 1553)  # (row, column) format
goal_position = (1983, 1623)     # (row, column) format

# Run the A* algorithm to find the shortest path
path = a_star(grid_map, current_position, goal_position)

# Mark the path on the map
for pos in path:
    if pos != current_position and pos != goal_position:
        grid_map[pos] = 5  # Mark the path with 5

# Mark current and goal positions
grid_map[current_position] = 2  # Current position
grid_map[goal_position] = 3     # Goal position

# Visualize the Map with the Shortest Path
plt.imshow(grid_map, cmap="gray")  # Use 'gray' colormap for black and white
plt.title("Map with Shortest Path")
plt.colorbar(label="0=Obstacle, 1=Free, 2=Current, 3=Goal, 5=Path")
plt.show()

# Export the Map
np.savetxt("warehouse_map_with_path.csv", grid_map, fmt='%d', delimiter=",")