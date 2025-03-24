import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import os

# Step 1: Load all CSV file paths in sorted order
csv_files = sorted(glob.glob("output/temperature_*.csv"))

# Step 2: Load the first file to get dimensions
initial_data = np.loadtxt(csv_files[0], delimiter=",")
fig, ax = plt.subplots()
im = ax.imshow(initial_data, cmap='inferno', origin='lower', extent=[0, 10, 0, 10], animated=True)
plt.colorbar(im, ax=ax, label="Temperature (K)")
ax.set_title("2D Heat Diffusion Animation")
ax.set_xlabel("x")
ax.set_ylabel("y")

# Step 3: Animation update function
def update(frame):
    data = np.loadtxt(csv_files[frame], delimiter=",")
    im.set_array(data)
    ax.set_title(f"Timestep: {frame+1}")
    return [im]

# Step 4: Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(csv_files), interval=50, blit=True)

plt.tight_layout()
plt.show()
