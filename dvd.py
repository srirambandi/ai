import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Parameters
grid_height = 50
grid_width = 60
tile_size = 1  # Tile size of 1x1

# Initialize the grid
grid = np.zeros((grid_height, grid_width))

# Initial position of the tile
x, y = 0, 0

# Direction of movement (diagonal)
dx, dy = 1, 1

# Number of frames for the animation
frames = 300

# Create a figure and axis
fig, ax = plt.subplots()

# Function to draw the grid and the tile
def draw_tile(i):
    ax.clear()  # Clear previous frame
    global x, y, dx, dy

    # Update position
    x += dx
    y += dy

    # Reflect from the boundaries
    if x == grid_width - tile_size or x == 0:
        dx *= -1
    if y == grid_height - tile_size or y == 0:
        dy *= -1

    # Draw the grid
    ax.imshow(grid, cmap='gray', extent=(0, grid_width, 0, grid_height))

    # Draw the tile
    square = plt.Rectangle((x, y), tile_size, tile_size, color='white')
    ax.add_patch(square)

    # Set the limits and remove the axes for a nicer look
    ax.set_xlim(0, grid_width)
    ax.set_ylim(0, grid_height)
    ax.axis('off')

# Create an animation
ani = animation.FuncAnimation(fig, draw_tile, frames=frames, interval=50)

# Display the animation
plt.show()
