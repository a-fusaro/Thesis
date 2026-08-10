import numpy as np
# Use logsumexp to calculate log of sum of exponentials in a numerically stable way  
from scipy.special import logsumexp
import matplotlib.pyplot as plt

# Specify chosen values  
SIGMA_VAL = 0.1  # Sigma of the forward model  
VAR_EXP = 0.05  # Variance appearing in exponential term of forward model  

# Prior Configuration 
PRIOR_MEAN = np.array([0.5, 2.0])  # 1st component: location, 2nd component: strength  
PRIOR_COV = np.array([
    [0.1, 0.0],  # Smaller variance in location
    [0.0, 1.0]   # Higher variance in strength
])  

# Setup grid for d_1 and d_2  
d_seq = np.arange(0, 1.01, 0.02)  # So 0, 0.02, ..., 0.98, 1  
N_grid = len(d_seq)  

# Specify nested Monte Carlo sample sizes  
N = 1000  # Outer loop samples  
M = 2000  # Inner loop samples  

# Sampling from the multivariate Gaussian prior
def sample_prior(num_samples):  
    return np.random.multivariate_normal(PRIOR_MEAN, PRIOR_COV, size=num_samples)

# Implementation of forward model evaluating the specified G(d, theta)  
def forward_model(d, theta):  
    # theta[:, 0] is location, theta[:, 1] is strength  
    theta_L = theta[:, 0]  
    theta_S = theta[:, 1]  
    # Calculate both entries of G(d, theta)  
    g1 = theta_S * np.exp(-((d[0] - theta_L)**2) / VAR_EXP)  
    g2 = theta_S * np.exp(-((d[1] - theta_L)**2) / VAR_EXP)  
    return np.column_stack((g1, g2))  

# Computation of EIG using double-loop MC-estimation  
def compute_eig_for_design(d, theta_outer):  
    # Generate true forward model outputs and add noise for the outer loop  
    G_outer = forward_model(d, theta_outer)     
    # Noise is sampled from built-in normal distribution sampler  
    noise = np.random.normal(0, np.sqrt(SIGMA_VAL), size=(N, 2))  
    y_outer = G_outer + noise     
    total_eig = 0.0  
    # Constant term for the 2D Gaussian log-likelihood  
    log_norm_constant = -np.log(2 * np.pi) - 0.5 * np.log(SIGMA_VAL**2)   
      
    # Track the pointwise EIG elements to estimate standard deviation  
    pointwise_eig = np.zeros(N)  
      
    # Loop over outer samples (i)  
    for i in range(N):  
        y_i = y_outer[i]  
        # Term (a): Log-likelihood log p(y^(i) | \theta^(i), d)  
        diff_outer = y_i - G_outer[i]  
        log_lik_outer = log_norm_constant - 0.5 * np.sum(diff_outer**2) / SIGMA_VAL  
            
        # Term (b): Log-evidence log((1/M)*\sump(y^(i)|\theta^(j), d) )  
        # Draw M inner samples using the updated fast sampler  
        theta_inner = sample_prior(M)  
        G_inner = forward_model(d, theta_inner)     
        # Calculation of log-likelihoods for all M inner samples against y_i  
        diff_inner = y_i - G_inner     
        log_liks_inner = log_norm_constant - 0.5 * np.sum(diff_inner**2, axis=1) / SIGMA_VAL  
        # Use logsumexp to safely calculate log(mean(exp(log_liks)))  
        log_evidence = logsumexp(log_liks_inner) - np.log(M)  
        # Add the value to the sum  
        pointwise_eig[i] = log_lik_outer - log_evidence         
    total_eig = np.sum(pointwise_eig)  
    mean_eig = total_eig / N  
    std_dev = np.std(pointwise_eig)  
      
    return mean_eig, std_dev  

# Grid EIG evaluation  
print("Sampling from outer prior distribution...")  
np.random.seed(123)  

# Pre-sample the outer thetas using the implemented sampler
theta_outer_all = sample_prior(N)  

# Initialize a matrix to store EIG values  
eig_grid = np.zeros((N_grid, N_grid))  
# Initialize a matrix to store standard deviation values  
std_grid = np.zeros((N_grid, N_grid))  

print(f"Starting classical double-loop grid search over {N_grid}x{N_grid} design points...")  
for i in range(N_grid):  
    for j in range(i, N_grid):  # Use symmetry: compute only for j >= i  
        d = np.array([d_seq[i], d_seq[j]])  
        # Compute EIG for one design parameter  
        val, sd = compute_eig_for_design(d, theta_outer_all)  
        eig_grid[i, j] = val  
        std_grid[i, j] = sd  
        if i != j:  
            eig_grid[j, i] = val  # Again uses symmetry: U(d1, d2) = U(d2, d1)  
            std_grid[j, i] = sd  
    print(f"Progress: Finished row {i + 1}/{N_grid} (d1 = {d_seq[i]:.2f})")  

# Find the optimal design  
opt_idx = np.unravel_index(np.argmax(eig_grid), eig_grid.shape)  
d1_opt = d_seq[opt_idx[0]]  
d2_opt = d_seq[opt_idx[1]]  
max_eig = eig_grid[opt_idx]  
opt_sd = std_grid[opt_idx]  

print("\n--- Optimization Result ---")  
print(f"Optimal Sensor Placements: d1 = {d1_opt:.2f}, d2 = {d2_opt:.2f}")  
print(f"Maximum Expected Information Gain: {max_eig:.4f} (SD: {opt_sd:.4f})")  

# Plot heatmap  
print("\nGenerating EIG and Standard Deviation Heatmaps...")  
fig, axes = plt.subplots(1, 2, figsize=(14, 5))  

# Left plot: Expected Information Gain  
im0 = axes[0].imshow(eig_grid, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')  
fig.colorbar(im0, ax=axes[0], label='EIG')  
axes[0].set_xlabel('d2')  
axes[0].set_ylabel('d1')  
axes[0].set_title('Expected Information Gain (EIG)')  

# Mark the optimal designs with a star  
axes[0].scatter(d2_opt, d1_opt, color='red', marker='*', s=200, edgecolors='black', zorder=5)  
if d1_opt != d2_opt:  
    axes[0].scatter(d1_opt, d2_opt, color='red', marker='*', s=200, edgecolors='black', zorder=5)  

# Right plot: Standard Deviation  
im1 = axes[1].imshow(std_grid, origin='lower', extent=[0, 1, 0, 1], cmap='magma')  
fig.colorbar(im1, ax=axes[1], label='Standard Deviation')  
axes[1].set_xlabel('d2')  
axes[1].set_ylabel('d1')  
axes[1].set_title('Standard Deviation Estimation')  
plt.tight_layout()  
plt.savefig('target_heatmap_nonlinear.png', dpi=300, bbox_inches='tight')  
plt.show()