import numpy as np
import matplotlib.pyplot as plt

# Define A, B, C, D and the target function
def target_fun(d1, d2):
    A = np.exp(-(d1 - 0.25)**2 / 0.05)
    B = np.exp(-(d1 - 0.75)**2 / 0.05)
    C = np.exp(-(d2 - 0.25)**2 / 0.05)
    D = np.exp(-(d2 - 0.75)**2 / 0.05)
    target = -np.log(100 * (A * D - B * C)**2 + 10 * (A**2 + B**2 + C**2 + D**2) + 1)
    return target

# Set evaluation grid
d1_seq = np.linspace(0, 1, 5001)
d2_seq = np.linspace(0, 1, 5001)
D1, D2 = np.meshgrid(d1_seq, d2_seq)

# Evaluate target on the grid
target_matrix = target_fun(D1, D2)

# Find design parameters with minimal target value (maximizing information gain)
min_val = np.min(target_matrix)
print(f"Minimum value: {min_val}")
print("Optimal design parameter combinations are:")

rows, cols = np.where(target_matrix == min_val)
for r, c in zip(rows, cols):
    print(f"d1 = {D1[r, c]:.4f}, d2 = {D2[r, c]:.4f}")

# Plot target heatmap
plt.figure(figsize=(8, 6))
plt.imshow(target_matrix, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
plt.colorbar(label='Target Value')

opt_d1 = D1[rows, cols]
opt_d2 = D2[rows, cols]
plt.scatter(opt_d2, opt_d1, color='red', marker='*', s=150, edgecolor='black', zorder=5)
plt.xlabel('d2')
plt.ylabel('d1')
plt.title('Log-Determinant of Posterior Covariance')
plt.savefig('target_heatmap_linear.png', dpi=300, bbox_inches='tight')
plt.show()