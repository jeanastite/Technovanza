from collections import deque

# Reconstructs the path from the start to the goal
def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]  # Reverse the path to start from the beginning

# BFS finds a path from start to goal
def bfs_shortest_path(maze, start, goal):
    rows, cols = maze.shape
    queue = deque([start])
    visited = set()
    came_from = {}
    
    # Directions for movement: Right, Down, Left, Up
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while queue:
        current = queue.popleft()

        if current == goal:
            return reconstruct_path(came_from, current)

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)

            # Check bounds and if the neighbor is a valid path
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if maze[neighbor] == 255 and neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    queue.append(neighbor)

    return None  # Return None if no path is found

# Run BFS on the maze
shortest_path = bfs_shortest_path(thresh, tuple(start_coords), tuple(end_coords))

if shortest_path:
    # Overlay the path on the maze
    for x, y in shortest_path:
        image[x, y] = (0, 0, 255)  # Mark path in red

    # Display the solved maze
    plt.figure(figsize=(10, 10))
    plt.title("Solved Maze with Shortest Path")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    # Output the shortest path coordinates
    print("Shortest Path:", shortest_path)
else:
    print("No path found in the maze!")
