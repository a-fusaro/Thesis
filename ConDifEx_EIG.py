import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm

np.typeDict = np.sctypeDict

# Pull framework from the PGsOED package of Wanggang Shen
from oed_package.pg_soed import PGsOED

##############################################################################
# ENVIRONMENT AND INITIALIZATION
##############################################################################
random_state = 2021
np.random.seed(random_state)
torch.manual_seed(random_state)
n_stage = 2
n_param = 2
n_design = 2
n_obs = 1
prior_type = "uniform"
prior_loc = 0
prior_scale = 1
prior_info = [(prior_type, prior_loc, prior_scale), (prior_type, prior_loc, prior_scale)]
design_bounds = [(-0.25, 0.25), (-0.25, 0.25)]
noise_loc = 0
noise_base_scale = 0.05
noise_ratio_scale = 0.05
noise_info = [(noise_loc, noise_base_scale, noise_ratio_scale)]
n_phys_state = 2
init_phys_state = (0.5, 0.5)
n_grid = 50
post_rvs_method = "Rejection"

# Load provided network surrogates
conv_diff_nets = (
    torch.load("conv_diff_net_t0.05.pt", weights_only=False),
    torch.load("conv_diff_net_t0.2.pt", weights_only=False),
)


def conv_diff_model(stage, theta, d, xp=None):
    n_sample = max(len(theta), len(d), len(xp))
    X = torch.zeros(n_sample, 4).double()
    X[:, :2] = torch.from_numpy(theta)
    X[:, 2:] = torch.from_numpy(xp + d)
    return conv_diff_nets[stage](X).detach().numpy()


# Define rewards and physical state transition
# Rewards are 0 as we do not impose a movement penalty, just the KL-divergence later on
def reward_fun(stage, xb, xp, d, y):
    return 0


# As the design is vehicle displacement, we add it to the vehicle location to get the next vehicle location
def phys_state_fun(xp, stage, d, y):
    return xp + d


phys_state_info = (n_phys_state, init_phys_state, phys_state_fun)

# Instantiate the problem by plugging all parameters in
soed = PGsOED(
    model_fun=conv_diff_model,
    reward_fun=reward_fun,
    prior_info=prior_info,
    design_bounds=design_bounds,
    noise_info=noise_info,
    phys_state_info=phys_state_info,
    n_stage=n_stage,
    n_param=n_param,
    n_design=n_design,
    n_obs=n_obs,
    n_grid=n_grid,
    post_rvs_method=post_rvs_method,
    random_state=random_state,
)

##############################################################################
# DEFINITIONS AND SAMPLING SETUP
##############################################################################
# Set up the grid of possible sensor vehicle locations
grid_res = 30
x_space = np.linspace(0, 1, grid_res)
y_space = np.linspace(0, 1, grid_res)
X_mesh, Y_mesh = np.meshgrid(x_space, y_space)

# Initialization
utility_t0 = np.zeros_like(X_mesh)
utility_t1 = np.zeros_like(X_mesh)

xb_init = soed.get_xb(None)
xp_fixed = np.array([[0.5, 0.5]])

# Set up sample sizes
n_particles = min(3500, xb_init.shape[0]) # Vorher 3500
n_y_samples = 15

sample_indices = np.random.choice(xb_init.shape[0], size=n_particles, replace=False)
thetas = xb_init[sample_indices, :2]

##############################################################################
# EIG COMPUTATION
##############################################################################
# Set up batches to ensure the code works without crashing
batch_size = 500

print("Calculating EIG ...")
prog = 0

for r in range(grid_res):
    for c in range(grid_res):
        # Track progress
        prog += 1
        print(f"Progress: {prog}/{grid_res**2}")

        target = np.array([X_mesh[r, c], Y_mesh[r, c]], dtype=np.float32)
        d_input = target.reshape(1, -1) - 0.5

        for stage, utility_matrix in [(0, utility_t0), (1, utility_t1)]:

            # Assess forward model to obtain measurement without noise
            G = (
                soed.m_f(
                    stage,
                    thetas,
                    np.repeat(d_input, n_particles, axis=0),
                    np.repeat(xp_fixed, n_particles, axis=0),
                )
                .flatten()
                .astype(np.float32)
            )

            # Noise std, depends on measurement scale
            noise_std = (noise_base_scale * (1.0 + np.abs(G))).astype(np.float32)

            # Simulated observation
            y_sim = np.random.normal(
                loc=G, scale=noise_std, size=(n_y_samples, n_particles)
            ).astype(np.float32)

            # Calculate conditional log-likelihoods: log p(y^{(i,j)} | theta^{(i)})
            err_diag = (y_sim - G[np.newaxis, :]) / noise_std[np.newaxis, :]
            p_y_conditional = norm.pdf(err_diag) / noise_std[np.newaxis, :]
            p_y_conditional = np.clip(p_y_conditional, 1e-15, None)

            # Batched marginal calculation: sum_k p(y^{(i,j)} | theta^{(k)})
            # Accumulate likelihoods in batches to avoid ceashing
            p_y_marginal_sum = np.zeros((n_y_samples, n_particles), dtype=np.float32)

            for b in range(0, n_particles, batch_size):
                G_b = G[b : b + batch_size]
                noise_std_b = noise_std[b : b + batch_size]

                err_b = (
                    y_sim[:, :, np.newaxis] - G_b[np.newaxis, np.newaxis, :]
                ) / noise_std_b[np.newaxis, np.newaxis, :]
                p_b = norm.pdf(err_b) / noise_std_b[np.newaxis, np.newaxis, :]

                # Sum over batch candidates
                p_y_marginal_sum += np.sum(p_b, axis=2, dtype=np.float32)

            p_y_marginal = p_y_marginal_sum / n_particles
            p_y_marginal = np.clip(p_y_marginal, 1e-15, None)

            # Mean Expected Information Gain across samples
            utility_matrix[r, c] = np.mean(
                np.log(p_y_conditional) - np.log(p_y_marginal)
            )

# Identify maximum
idx_max_t0 = np.unravel_index(np.argmax(utility_t0), utility_t0.shape)
max_x_t0, max_y_t0 = X_mesh[idx_max_t0], Y_mesh[idx_max_t0]

idx_max_t1 = np.unravel_index(np.argmax(utility_t1), utility_t1.shape)
max_x_t1, max_y_t1 = X_mesh[idx_max_t1], Y_mesh[idx_max_t1]

##############################################################################
# PLOTTING AND EXPORT
##############################################################################
# Calculate global minimum and maximum across both utility matrices to generate shared scale
vmin = min(np.min(utility_t0), np.min(utility_t1))
vmax = max(np.max(utility_t0), np.max(utility_t1))

# Create a shared set of levels
shared_levels = np.linspace(vmin, vmax, 21)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
ticks_f10 = np.arange(0.0, 1.1, 0.1)
tick_labels = [f"{t:.1f}" for t in ticks_f10]

# Subplot (a) t = 0.05
ax0 = axes[0]
ax0.set_aspect("equal")
# Apply shared levels, vmin, and vmax
cf0 = ax0.contourf(X_mesh, Y_mesh, utility_t0, cmap="viridis", levels=shared_levels, vmin=vmin, vmax=vmax)
cbar0 = plt.colorbar(cf0, ax=ax0, fraction=0.046, pad=0.04)
cbar0.ax.tick_params(labelsize=8)
ax0.scatter(
    max_x_t0,
    max_y_t0,
    color="red",
    marker="*",
    s=150,
    edgecolor="white",
    linewidth=0.7,
    label="Max EIG",
)
ax0.set_title("(a) t = 0.05", fontsize=11, y=-0.18)
ax0.set_xlabel("$Z_x$", fontsize=10)
ax0.set_ylabel("$Z_y$", fontsize=10)
ax0.legend(loc="upper right", fontsize=8)

# Subplot (b) t = 0.2
ax1 = axes[1]
ax1.set_aspect("equal")
cf1 = ax1.contourf(X_mesh, Y_mesh, utility_t1, cmap="viridis", levels=shared_levels, vmin=vmin, vmax=vmax)
cbar1 = plt.colorbar(cf1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.ax.tick_params(labelsize=8)
ax1.scatter(
    max_x_t1,
    max_y_t1,
    color="red",
    marker="*",
    s=150,
    edgecolor="white",
    linewidth=0.7,
    label="Max EIG",
)
ax1.set_title("(b) t = 0.2", fontsize=11, y=-0.18)
ax1.set_xlabel("$Z_x$", fontsize=10)
ax1.set_ylabel("$Z_y$", fontsize=10)

# Formatting rules
for ax in axes:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(ticks_f10)
    ax.set_yticks(ticks_f10)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.grid(
        True,
        which="both",
        linestyle="-",
        color="gray",
        linewidth=0.3,
        alpha=0.5,
    )

plt.suptitle(
    "EIG versus sensor location if conducting a single experiment at t = 0.05 or t = 0.2.",
    fontsize=11,
    y=0.98,
)
plt.tight_layout()
plt.savefig("figure_expected_utility.png", dpi=300)
plt.show()