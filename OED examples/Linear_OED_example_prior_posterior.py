import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

# True theta 
theta_true = np.array([1, 3])

# Three design cases: [d1, d2]
cases = {
    '1': [0.25, 0.75],
    '2': [0.55, 0.8],
    '3': [0.45, 0.45]
}

# Generate prior samples 
K_samples = 7500000 
np.random.seed(77)
prior_mean, prior_var = 2.0, 1.0
prior_std = np.sqrt(prior_var)
theta_samples = np.random.normal(prior_mean, prior_std, size=(K_samples, 2))

# Setup 2x2 plot
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

# Plot prior Density using the analytical formula across the [0, 4] grid space
xedges = np.linspace(0, 4, 51)
yedges = np.linspace(0, 4, 51)
X, Y = np.meshgrid(xedges, yedges)
prior_pdf = (1 / (2 * np.pi * prior_var)) * np.exp(-0.5 * ((X - prior_mean)**2 + (Y - prior_mean)**2) / prior_var)
im0 = axes[0].imshow(prior_pdf, origin='lower', extent=[0, 4, 0, 4], cmap='inferno', aspect='auto')
axes[0].set_title('Prior Parameter Density')
fig.colorbar(im0, ax=axes[0])

# Initialize table for later use
table_data = {}

# Loop through three design cases
for i, (name, d) in enumerate(cases.items(), start=1):
    # Forward model coefficients
    A = np.exp(-(d[0] - 0.25)**2 / 0.05)
    B = np.exp(-(d[0] - 0.75)**2 / 0.05)
    C = np.exp(-(d[1] - 0.25)**2 / 0.05)
    D = np.exp(-(d[1] - 0.75)**2 / 0.05)
    
    # Posterior Estimation for given true_theta 
    y_true = np.array([theta_true[0] * A + theta_true[1] * B, theta_true[0] * C + theta_true[1] * D])
    noise_sigma = 0.1
    y_obs = y_true + np.random.normal(0, noise_sigma, size=2)
    
    pred_1 = theta_samples[:, 0] * A + theta_samples[:, 1] * B
    pred_2 = theta_samples[:, 0] * C + theta_samples[:, 1] * D
    rss = (y_obs[0] - pred_1)**2 + (y_obs[1] - pred_2)**2
    log_liks = -0.5 * rss / (noise_sigma**2)
    
    log_prior = -0.5 * np.log(2 * np.pi * prior_var) - 0.5 * ((theta_samples - prior_mean)**2) / prior_var
    log_numerator = log_liks + log_prior[:, 0] + log_prior[:, 1]
    
    weights = np.exp(log_numerator - logsumexp(log_numerator))
    
    post_mean = np.sum(theta_samples * weights[:, None], axis=0)
    post_var = np.sum(weights[:, None] * (theta_samples - post_mean)**2, axis=0)
    post_std = np.sqrt(post_var)
    mse = np.mean((post_mean - theta_true)**2)

    # Expected Information Gain (EIG) Calculation
    N_outer = 5000 
    M_inner = 20000 
    
    theta_outer = theta_samples[:N_outer]
    theta_inner = theta_samples[N_outer : N_outer + M_inner]
    
    # Forward pass evaluations for EIG components
    G1_outer = theta_outer[:, 0] * A + theta_outer[:, 1] * B
    G2_outer = theta_outer[:, 0] * C + theta_outer[:, 1] * D
    G_outer = np.column_stack((G1_outer, G2_outer))
    
    G1_inner = theta_inner[:, 0] * A + theta_inner[:, 1] * B
    G2_inner = theta_inner[:, 0] * C + theta_inner[:, 1] * D
    G_inner = np.column_stack((G1_inner, G2_inner))
    
    # Simulate outer loop noise and observations
    noise_outer = np.random.normal(0, noise_sigma, size=(N_outer, 2))
    Y_outer = G_outer + noise_outer
    
    # Evaluate log p(y^n|theta^n,d)
    log_lik_outer = -np.log(2 * np.pi * (noise_sigma**2)) - 0.5 * np.sum((Y_outer - G_outer)**2, axis=1) / (noise_sigma**2)
    
    # Evaluate log p(y^n|d) using inner samples via broadcasting
    rss_matrix = np.sum((Y_outer[:, None, :] - G_inner[None, :, :])**2, axis=2)
    log_lik_inner_matrix = -np.log(2 * np.pi * (noise_sigma**2)) - 0.5 * rss_matrix / (noise_sigma**2)
    log_evidence = logsumexp(log_lik_inner_matrix, axis=1) - np.log(M_inner)
    
    # EIG calculation via Monte Carlo estimator using mean difference
    eig = np.mean(log_lik_outer - log_evidence)
    
    table_data[name] = {'mse': mse, 'std_A': post_std[0], 'std_B': post_std[1], 'eig': eig}
    
    # Continuous grid density plotting updated to [0, 4] range limits
    counts, x, y = np.histogram2d(theta_samples[:, 0], theta_samples[:, 1], bins=50, range=[[0, 4], [0, 4]], weights=weights)
    im = axes[i].imshow(counts.T, origin='lower', extent=[0, 4, 0, 4], cmap='inferno', aspect='auto')
    
    # Mark true theta and posterior mean
    axes[i].scatter(theta_true[0], theta_true[1], color='cyan', marker='*', s=250, label=r'True $\theta^*$', zorder=10)
    axes[i].scatter(post_mean[0], post_mean[1], color='red', marker='+', s=250, linewidths=3, label='Post Mean', zorder=11)
    axes[i].set_title(f"Case {name}: d=({d[0]:.2f}, {d[1]:.2f})")
    axes[i].legend()
    fig.colorbar(im, ax=axes[i])

# Formatting subplots and plotting
for ax in axes:
    ax.set_xlabel(r'$\theta_A$')
    ax.set_ylabel(r'$\theta_B$')
plt.tight_layout()
plt.savefig('prior_posterior_comparison_linear.png', dpi=300, bbox_inches='tight')
plt.show()

# Print updated results table
print(f"\n{'Design Case':<15} | {'EIG':<10} | {'MSE':<12} | {'Post Std (theta_A)':<20} | {'Post Std (theta_B)':<20}")
print("-" * 88)
for name, data in table_data.items():
    print(f"{name:<15} | {data['eig']:<10.4f} | {data['mse']:<12.6f} | {data['std_A']:<20.6f} | {data['std_B']:<20.6f}")