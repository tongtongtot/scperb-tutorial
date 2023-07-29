import numpy as np
import matplotlib.pyplot as plt

# Parameters for the normal distribution
mu = 0         # Mean
sigma = 1      # Standard deviation

# Generate data points for the normal distribution
x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, color='lightorange', linewidth=2)  # Use 'lightorange' for the curve color
plt.xlabel('X')
plt.ylabel('Probability Density')
plt.title('Normal Distribution')
plt.grid(True)

# Show the plot
plt.show()
