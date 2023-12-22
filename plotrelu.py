import matplotlib.pyplot as plt
import numpy as np

# Define the ReLU function
def relu(x):
    return np.maximum(0, x)

# Generate some sample input values
x = np.linspace(-5, 5, 100)

# Calculate the corresponding output values
y = relu(x)

# Create a new figure and axes
fig, ax = plt.subplots()

# Plot the ReLU function


# Set the x and y axis limits to run through the origin
ax.axhline(0, color='black', lw=2, linestyle = '--')
ax.axvline(0, color='black', lw=2, linestyle = '--')
ax.plot(x, y)
# Set the x and y axis labels
ax.set_xlabel('Input (x)')
ax.set_ylabel('Output (y)')

# Set the title of the plot
ax.set_title('ReLU Function')
plt.grid(axis = 'both')
plt.ylim(-5,5)
plt.xlim(-5,5)
# Display the plot
plt.show()
