import numpy as np
import matplotlib.pyplot as plt
import heapq
from matplotlib.animation import FuncAnimation

# Grid environment setup
grid_size = (10, 10)
obstacles = [(1, 1), (2, 1), (3, 3)]  # obstacles
start = (0, 0)  # Start position
goal = (6, 6)  # Goal position

# Create grid map
grid = np.zeros(grid_size)
for obs in obstacles:
    grid[obs] = 1  # Mark obstacles

# Initialize plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xticks(np.arange(0, grid.shape[1], 1))
ax.set_yticks(np.arange(0, grid.shape[0], 1))
ax.set_xlim(-0.5, grid.shape[1] - 0.5)
ax.set_ylim(grid.shape[0] - 0.5, -0.5)
ax.invert_yaxis()  # Invert y-axis to match grid coordinates
ax.grid(True, which='both', linestyle='--', linewidth=1)

# Function to plot the grid
def plot_grid(ax, grid, path=[], visited=[], obstacles=[]):
    ax.clear()  # Clear previous plot
    ax.imshow(grid, cmap="gray_r", origin="upper", extent=[0, grid.shape[1], 0, grid.shape[0]])

    # Plot obstacles, start, and goal
    for obs in obstacles:
        ax.scatter(obs[1], obs[0], color='black', marker='s', s=100, label='Obstacle')
    ax.scatter(start[1], start[0], color='red', marker='o', s=100, label='Start')
    ax.scatter(goal[1], goal[0], color='yellow', marker='o', s=100, label='Goal')

    # Plot visited nodes and path
    if visited:
        visited = np.array(visited)
        ax.scatter(visited[:, 1], visited[:, 0], color='blue', marker='x', label='Visited')
    if path:
        path = np.array(path)
        ax.plot(path[:, 1], path[:, 0], color='green', lw=3, marker='o', markersize=6, label='Path')

    # Set grid ticks and labels
    ax.set_xticks(np.arange(0, grid.shape[1], 1))
    ax.set_yticks(np.arange(0, grid.shape[0], 1))
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(grid.shape[0] - 0.5, -0.5)
    ax.invert_yaxis()

    ax.grid(True, which='both', linestyle='--', linewidth=1)
    ax.legend(loc='best')
    ax.set_title("A* Pathfinding Simulation")

# Heuristic function (Euclidean distance)
def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# A* Algorithm with animation
def a_star(grid, start, goal):
    open_list = []
    heapq.heappush(open_list, (0, start))  # Push start into priority queue
    g_costs = {start: 0}
    f_costs = {start: heuristic(start, goal)}
    parents = {start: None}
    visited = []
    path = []

    # Animation function
    def animate(i):
        nonlocal open_list, visited, path

        if open_list:
            _, current = heapq.heappop(open_list)

            # Reconstruct the path if goal is reached
            if current == goal:
                path = []
                while current:
                    path.append(current)
                    current = parents[current]
                path.reverse()
                return path, visited

            # Add current node to visited list
            visited.append(current)

            # Explore neighbors (4 directions: up, down, left, right)
            neighbors = [(current[0] - 1, current[1]), (current[0] + 1, current[1]),
                        (current[0], current[1] - 1), (current[0], current[1] + 1)]

            for neighbor in neighbors:
                if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1] and grid[neighbor] != 1:
                    tentative_g_cost = g_costs[current] + 1
                    if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                        g_costs[neighbor] = tentative_g_cost
                        f_costs[neighbor] = tentative_g_cost + heuristic(neighbor, goal)
                        parents[neighbor] = current
                        heapq.heappush(open_list, (f_costs[neighbor], neighbor))

        # Update plot with each frame
        plot_grid(ax, grid, path=path, visited=visited, obstacles=obstacles)

        # Stop animation once the path is found
        if path:
            ani.event_source.stop()  # Stop the animation once the goal is reached

        return path, visited

    # Run the animation
    ani = FuncAnimation(fig, animate, frames=100, interval=200, repeat=False)
    plt.show()

# Run the A* algorithm and animation
a_star(grid, start, goal)
