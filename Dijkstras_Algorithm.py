import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import heapq

# Define the grid world
grid_size = (10, 10)  # Size of the grid
obstacles = [(3, 3), (3, 4), (3, 5), (4, 5), (5, 5), (6, 2), (6, 3), (7, 3)]  # List of obstacle coordinates
start = (0, 0)  # Start position
goal = (9, 9)  # Goal position

# Define Dijkstra's algorithm
def dijkstra(grid_size, start, goal, obstacles):
    rows, cols = grid_size
    visited = set()
    costs = {start: 0}
    parent = {start: None}
    priority_queue = [(0, start)]
    
    while priority_queue:
        current_cost, current_node = heapq.heappop(priority_queue)
        if current_node in visited:
            continue
        visited.add(current_node)
        
        if current_node == goal:
            break
        
        # Explore neighbors
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, down, left, right
            neighbor = (current_node[0] + dx, current_node[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and neighbor not in obstacles:
                new_cost = current_cost + 1  # Uniform cost
                if neighbor not in costs or new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    parent[neighbor] = current_node
                    heapq.heappush(priority_queue, (new_cost, neighbor))
    
    # Reconstruct the path
    path = []
    if goal in parent:
        node = goal
        while node:
            path.append(node)
            node = parent[node]
        path.reverse()
    
    return path, visited

# Get the path and visited nodes
path, visited_nodes = dijkstra(grid_size, start, goal, obstacles)

# Visualization
fig, ax = plt.subplots()
ax.set_xlim(-0.5, grid_size[1] - 0.5)
ax.set_ylim(-0.5, grid_size[0] - 0.5)
ax.set_xticks(range(grid_size[1]))
ax.set_yticks(range(grid_size[0]))
ax.grid(True)

# Plot obstacles, start, and goal
for obs in obstacles:
    ax.add_patch(plt.Rectangle((obs[1] - 0.5, obs[0] - 0.5), 1, 1, color="black", alpha=0.8))  # Obstacles
ax.plot(start[1], start[0], 'go', markersize=10, label="Start")  # Start point
ax.plot(goal[1], goal[0], 'ro', markersize=10, label="Goal")  # Goal point
ax.legend()

visited_scatter, = ax.plot([], [], 'co', markersize=5, alpha=0.6, label="Visited Nodes")
path_line, = ax.plot([], [], 'm-', linewidth=2, label="Path")

visited_nodes_list = list(visited_nodes)  # Convert to a list for animation
visited_nodes_list.remove(start)

def init():
    visited_scatter.set_data([], [])
    path_line.set_data([], [])
    return visited_scatter, path_line

def update(frame):
    if frame < len(visited_nodes_list):
        current_visited = visited_nodes_list[:frame]
        visited_x = [node[1] for node in current_visited]
        visited_y = [node[0] for node in current_visited]
        visited_scatter.set_data(visited_x, visited_y)
    else:
        path_x = [node[1] for node in path]
        path_y = [node[0] for node in path]
        path_line.set_data(path_x, path_y)
    return visited_scatter, path_line

ani = FuncAnimation(fig, update, frames=len(visited_nodes_list) + 10, init_func=init, blit=True, repeat=False)

plt.show()
