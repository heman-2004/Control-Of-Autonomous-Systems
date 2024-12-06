import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Environment parameters
start_point = np.array([0, 0])  # Starting point
target_point = np.array([9, 9])  # Target point
max_step = 0.5  # Maximum step size
target_radius = 0.5  # Radius within which the target is considered reached
iteration_limit = 1000  # Maximum number of iterations
x_limits = (0, 10)  # X-axis range
y_limits = (0, 10)  # Y-axis range

# Obstacles
obstacle_centers = [
    np.array([3, 3]),
    np.array([6, 6]),
    np.array([7, 2]),
    np.array([5, 8]), 
]
obstacle_sizes = [1, 1.5, 1, 0.8]  # Radii of obstacles

# Utility functions
def euclidean_distance(point1, point2):
    """Calculate distance between two points."""
    return np.linalg.norm(point1 - point2)

def check_collision(point):
    """Verify if a point collides with any obstacles."""
    for obs, radius in zip(obstacle_centers, obstacle_sizes):
        if euclidean_distance(point, obs) <= radius:
            return True
    return False

def random_free_sample():
    """Generate a random sample within the environment, avoiding collisions."""
    while True:
        sampled_point = np.array([
            np.random.uniform(*x_limits),
            np.random.uniform(*y_limits)
        ])
        if not check_collision(sampled_point):
            return sampled_point

def move_towards(start, target, max_step):
    """Move from 'start' towards 'target' by a step size."""
    vector = target - start
    dist = np.linalg.norm(vector)
    if dist < max_step:
        return target
    return start + max_step * (vector / dist)

# RRT Algorithm Implementation
class RapidlyExploringRandomTree:
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal
        self.tree_nodes = [start]  # List of nodes
        self.node_parents = {tuple(start): None}  # Parent nodes for path reconstruction
        self.edges = []  # Connections between nodes

    def closest_node(self, point):
        """Find the closest node to a given point."""
        return min(self.tree_nodes, key=lambda node: euclidean_distance(node, point))

    def add_tree_node(self, new_node, parent_node):
        """Add a new node to the tree."""
        self.tree_nodes.append(new_node)
        self.node_parents[tuple(new_node)] = tuple(parent_node)
        self.edges.append((parent_node, new_node))

    def reconstruct_path(self):
        """Reconstruct the path from start to goal."""
        path = []
        current_node = tuple(self.goal)
        while current_node is not None:
            path.append(np.array(current_node))
            current_node = self.node_parents.get(current_node)
        return path[::-1]

# RRT Execution
rrt_algorithm = RapidlyExploringRandomTree(start_point, target_point)
goal_reached = False

# Visualization
fig, axis = plt.subplots()
axis.set_xlim(x_limits)
axis.set_ylim(y_limits)
axis.set_title("RRT Path Planning with Visualization")
axis.set_xlabel("X-Axis")
axis.set_ylabel("Y-Axis")

# Draw obstacles
for center, size in zip(obstacle_centers, obstacle_sizes):
    obstacle_circle = plt.Circle(center, size, color='orange', alpha=0.7)  # New color for obstacles
    axis.add_artist(obstacle_circle)

# Visualization elements
tree_lines, = axis.plot([], [], 'purple', linewidth=0.5, label="Tree Connections")  # Tree connections in purple
path_line, = axis.plot([], [], 'lime', linewidth=2, label="Final Path")  # Final path in lime
axis.plot(start_point[0], start_point[1], 'bo', markersize=8, label="Start")  # Start marker in blue
axis.plot(target_point[0], target_point[1], 'mo', markersize=8, label="Target")  # Goal marker in magenta

axis.legend()

# Animation initialization
def initialize_animation():
    tree_lines.set_data([], [])
    path_line.set_data([], [])
    return tree_lines, path_line

# Animation update
def update_frame(frame):
    global goal_reached

    if goal_reached:
        # If goal reached, display the final path
        planned_path = rrt_algorithm.reconstruct_path()
        if len(planned_path) > 1:
            planned_path = np.array(planned_path)
            path_line.set_data(planned_path[:, 0], planned_path[:, 1])
        return tree_lines, path_line

    # Grow the tree
    sampled_point = random_free_sample()
    nearest_node = rrt_algorithm.closest_node(sampled_point)
    new_node = move_towards(nearest_node, sampled_point, max_step)

    if not check_collision(new_node):
        rrt_algorithm.add_tree_node(new_node, nearest_node)
        if euclidean_distance(new_node, target_point) < target_radius:
            rrt_algorithm.add_tree_node(target_point, new_node)
            goal_reached = True
            print("Target reached!")

    # Update tree connections for animation
    edge_array = np.array(rrt_algorithm.edges)
    if len(edge_array) > 0:
        x_coords = np.array([[edge[0][0], edge[1][0]] for edge in edge_array]).flatten()
        y_coords = np.array([[edge[0][1], edge[1][1]] for edge in edge_array]).flatten()
        tree_lines.set_data(x_coords, y_coords)

    return tree_lines, path_line

# Animate
animation = FuncAnimation(fig, update_frame, frames=iteration_limit, init_func=initialize_animation, blit=True, repeat=False)

plt.show()
