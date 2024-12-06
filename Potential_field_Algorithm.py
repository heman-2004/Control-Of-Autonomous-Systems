import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Potential Field Path Planning

# Parameters
attraction_gain = 1.0  # Gain for attraction force
repulsion_gain = 100.0  # Gain for repulsion force
goal_tolerance = 0.5  # Distance threshold to consider goal reached
repulsion_radius = 5.0  # Effective range of repulsion forces
move_step = 0.1  # Step size for robot movement

# Environment setup
target_position = np.array([8, 8])  # Target/goal position
barriers = [np.array([4, 4]), np.array([6, 6]), np.array([7, 2])]  # Obstacle positions
initial_position = np.array([0, 0])  # Starting position of the robot

# Helper Functions
def compute_attraction(position, target):
    """Calculate the attractive force toward the target."""
    return -attraction_gain * (position - target)

def compute_repulsion(position, obstacles):
    """Calculate the repulsive force from obstacles."""
    repulsion_force = np.zeros(2)
    for barrier in obstacles:
        dist = np.linalg.norm(position - barrier)
        if dist < repulsion_radius:
            repulsion = (
                repulsion_gain
                * (1 / dist - 1 / repulsion_radius)
                / (dist**2)
                * (position - barrier)
                / dist
            )
            repulsion_force += repulsion
    return repulsion_force

def calculate_total_force(position, target, obstacles):
    """Combine attractive and repulsive forces to get the total force."""
    attraction_force = compute_attraction(position, target)
    repulsion_force = compute_repulsion(position, obstacles)
    return attraction_force + repulsion_force

# Visualization setup
fig, axis = plt.subplots()
axis.set_xlim(-1, 10)
axis.set_ylim(-1, 10)
axis.set_title("Potential Field Path Planning")
axis.set_xlabel("X-axis")
axis.set_ylabel("Y-axis")

# Draw obstacles and target
for barrier in barriers:
    axis.add_artist(plt.Circle(barrier, 0.5, color="purple", alpha=0.5))  # Purple for obstacles
axis.plot(
    target_position[0], 
    target_position[1], 
    'y*',  # Yellow star for the target
    markersize=15, 
    label="Target"
)
axis.legend()

# Agent (robot) representation
robot, = axis.plot([], [], 'co', markersize=10, label="Robot")  # Cyan circle for the robot
trail, = axis.plot([], [], 'c--', linewidth=1, alpha=0.6)  # Cyan dashed line for the trail

trajectory = [initial_position]  # Store robot's path

def initialize_animation():
    """Initialize the animation."""
    robot.set_data([initial_position[0]], [initial_position[1]])  # x, y must be sequences
    trail.set_data([], [])  # Initialize the trail as empty
    return robot, trail

def update_frame(frame):
    """Update the robot's position during the animation."""
    global trajectory
    current_position = trajectory[-1]
    if np.linalg.norm(current_position - target_position) < goal_tolerance:
        return robot, trail  # Stop updating if goal is reached
    total_force = calculate_total_force(current_position, target_position, barriers)
    new_position = current_position + move_step * total_force / np.linalg.norm(total_force)
    trajectory.append(new_position)
    robot.set_data([new_position[0]], [new_position[1]])  # x, y must be sequences
    trail.set_data(
        [pos[0] for pos in trajectory], 
        [pos[1] for pos in trajectory]
    )  # Update trail with the full path
    return robot, trail

# Create animation
animation = FuncAnimation(
    fig,
    update_frame,
    frames=200,
    init_func=initialize_animation,
    blit=True,
    repeat=False
)

plt.show()
