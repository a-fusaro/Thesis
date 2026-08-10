import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

# Specify chosen values  
SIGMA_VAL = 0.1 
VAR_EXP = 0.05 

# Updated Prior Configuration (No truncation)
PRIOR_MEAN = np.array([0.5, 2.0])  # 1st component: location, 2nd component: strength   
PRIOR_COV = np.array([ 
    [0.1, 0.0],  # Smaller variance in location 
    [0.0, 1.0]   # Higher variance in strength 
])   

# Implementation of forward model evaluating the specified G(d, theta)  
def forward_model(d, theta): 
    theta_L = theta[:, 0] 
    theta_S = theta[:, 1] 
    g1 = theta_S * np.exp(-((d[0] - theta_L)**2) / VAR_EXP) 
    g2 = theta_S * np.exp(-((d[1] - theta_L)**2) / VAR_EXP) 
    return np.column_stack((g1, g2)) 

# Sample from the multivariate Gaussian prior
def sample_prior(num_samples):   
    return np.random.multivariate_normal(PRIOR_MEAN, PRIOR_COV, size=num_samples) 

# True theta
theta_L_true, theta_S_true = 0.4, 2.5  
theta_true = np.array([theta_L_true, theta_S_true])

# Adjusted Case 1 to the new optimal design (0.36, 0.64)
cases = {
    '1': np.array([0.36, 0.64]),
    '2': np.array([0.10, 0.80]),
    '3': np.array([0.60, 0.60])
}

# Generate prior samples without truncation
K_samples = 7500000 
np.random.seed(77)
theta_samples = sample_prior(K_samples)

# Setup Subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Plot analytical prior density over updated theta_S grid [0, 4.0]
xedges_p = np.linspace(0, 1.0, 51)
yedges_p = np.linspace(0, 4.0, 51)  
X, Y = np.meshgrid(xedges_p, yedges_p)

var_L = PRIOR_COV[0, 0]
var_S = PRIOR_COV[1, 1]
pdf_x = (1.0 / (np.sqrt(2 * np.pi * var_L))) * np.exp(-0.5 * ((X - PRIOR_MEAN[0])**2) / var_L)
pdf_y = (1.0 / (np.sqrt(2 * np.pi * var_S))) * np.exp(-0.5 * ((Y - PRIOR_MEAN[1])**2) / var_S)
exact_prior_pdf = pdf_x * pdf_y

im0 = axes[0].imshow(
    exact_prior_pdf,
    origin='lower',
    extent=[xedges_p[0], xedges_p[-1], yedges_p[0], yedges_p[-1]],
    cmap='inferno',
    aspect='auto',
)
axes[0].set_title('Prior Parameter Density')
fig.colorbar(im0, ax=axes[0])

table_data = {}

# Loop through design cases 
for i, (name, d) in enumerate(cases.items(), start=1):
    # Posterior estimation for true_theta 
    G_true = forward_model(d, np.array([[theta_L_true, theta_S_true]]))
    noise = np.random.normal(0, np.sqrt(SIGMA_VAL), size=(1, 2))
    y_obs = G_true + noise
    
    G_pred = forward_model(d, theta_samples)
    rss = (y_obs[0, 0] - G_pred[:, 0])**2 + (y_obs[0, 1] - G_pred[:, 1])**2
    log_liks = -0.5 * rss / SIGMA_VAL
    
    # Updated analytic Gaussian log-prior matching your new PRIOR_COV
    log_prior_L = -0.5 * np.log(2 * np.pi * var_L) - 0.5 * ((theta_samples[:, 0] - PRIOR_MEAN[0])**2) / var_L
    log_prior_S = -0.5 * np.log(2 * np.pi * var_S) - 0.5 * ((theta_samples[:, 1] - PRIOR_MEAN[1])**2) / var_S
    log_numerator = log_liks + log_prior_L + log_prior_S
    
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
    
    G_outer = forward_model(d, theta_outer)
    G_inner = forward_model(d, theta_inner)
    
    # Simulate data for outer loop 
    noise_outer = np.random.normal(0, np.sqrt(SIGMA_VAL), size=(N_outer, 2))
    Y_outer = G_outer + noise_outer
    
    # Evaluate log p(y^n | theta^n, d)
    log_lik_outer = -np.log(2 * np.pi * SIGMA_VAL) - 0.5 * np.sum((Y_outer - G_outer)**2, axis=1) / SIGMA_VAL
    
    # Evaluate log p(y^n | d) using Inner samples via broadcasting
    rss_matrix = np.sum((Y_outer[:, None, :] - G_inner[None, :, :])**2, axis=2)
    log_lik_inner_matrix = -np.log(2 * np.pi * SIGMA_VAL) - 0.5 * rss_matrix / SIGMA_VAL
    
    # Compute log evidence for each outer sample
    log_evidence = logsumexp(log_lik_inner_matrix, axis=1) - np.log(M_inner)
    
    # EIG calculation
    eig = np.mean(log_lik_outer - log_evidence)
    
    table_data[name] = {'mse': mse, 'std_L': post_std[0], 'std_S': post_std[1], 'eig': eig}
    
    # Plotting setup
    counts, xedges, yedges = np.histogram2d(
        theta_samples[:, 0],
        theta_samples[:, 1],
        bins=50,
        range=[[0, 1.0], [0, 4.0]],
        weights=weights,
    )
    im = axes[i].imshow(
        counts.T,
        origin='lower',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap='inferno',
        aspect='auto',
    )
    axes[i].scatter(
        theta_L_true,
        theta_S_true,
        color='cyan',
        marker='*',
        s=250,
        label=r'True $\theta^*$',
        zorder=10,
    )
    axes[i].scatter(
        post_mean[0],
        post_mean[1],
        color='red',
        marker='+',
        s=250,
        linewidths=3,
        label='Post Mean',
        zorder=11,
    )
    axes[i].set_title(f"Case {name}: d=({d[0]:.2f}, {d[1]:.2f})")
    axes[i].legend()
    fig.colorbar(im, ax=axes[i])

for ax in axes:
    ax.set_xlabel(r'Location Parameter $\theta_L$')
    ax.set_ylabel(r'Strength Parameter $\theta_S$')
plt.tight_layout()
plt.savefig('prior_posterior_comparison_nonlinear.png', dpi=300, bbox_inches='tight')
plt.show()

# Print updated results table
print(f"\n{'Design Case':<15} | {'EIG':<10} | {'MSE':<12} | {'Post Std (theta_L)':<20} | {'Post Std (theta_S)':<20}")
print("-" * 88)
for name, data in table_data.items():
    print(f"{name:<15} | {data['eig']:<10.4f} | {data['mse']:<12.6f} | {data['std_L']:<20.6f} | {data['std_S']:<20.6f}")