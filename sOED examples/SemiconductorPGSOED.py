import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

np.typeDict = np.sctypeDict

from oed_package.pg_soed import PGsOED

# ============================================================================
# LOAD TRAINED GAUSSIAN SURROGATE DNN MODEL 
# ============================================================================
class ForwardPDESurrogate(nn.Module):
    def __init__(self, input_dim=3):
        super(ForwardPDESurrogate, self).__init__()
        self.register_buffer('mu', torch.zeros(input_dim))
        self.register_buffer('sigma', torch.ones(input_dim))
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 40),
            nn.GELU(),
            nn.Linear(40, 80),
            nn.GELU(),
            nn.Linear(80, 40),
            nn.GELU(),
            nn.Linear(40, 20),
            nn.GELU(),
            nn.Linear(20, 10),
            nn.GELU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        sigma_safe = torch.clamp(self.sigma, min=1e-8)
        x_scaled = (x - self.mu) / sigma_safe
        return self.network(x_scaled)

surrogate_model = ForwardPDESurrogate(input_dim=3)
model_path = "semiconductor_gaussian_pde_surrogate.pt"
if os.path.exists(model_path):
    surrogate_model.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"Successfully loaded surrogate model from {model_path}")
else:
    print(f"Warning: {model_path} not found. Ensure you run the 3-input training script first.")
surrogate_model.eval()

# ============================================================================
# PARAMETERS & PROBLEM CONFIGURATION 
# ============================================================================
n_stage = 8       
n_param = 2        
n_design = 1       
n_obs = 1          
n_phys_state = 1   
n_grid = 50        

# Priors for the unknown parameter theta
prior_info = [
    ("uniform", 0.2, 0.6),   # theta_N_y: [0.2, 0.8]
    ("uniform", -0.8, 0.6),  # theta_P_y: [-0.8, -0.2]
]

design_bounds = [(-0.75, 0.75)]  

noise_loc = 0.0
noise_base_scale = 0.05 
noise_ratio_scale = 0.05 
noise_info = [(noise_loc, noise_base_scale, noise_ratio_scale)]

init_phys_state = (0.0,)        
post_rvs_method = "Rejection"        

random_state = 2026
np.random.seed(random_state)
torch.manual_seed(random_state)

def semiconductor_surrogate_model(stage, theta, d, xp=None):
    n_sample = max(len(theta), len(d), len(xp) if xp is not None else 0)
    t_Ny = theta[:, 0]
    t_Py = theta[:, 1]
    U = xp.flatten() if (xp is not None and len(xp) > 0) else np.zeros(n_sample)
        
    X_input = torch.zeros(n_sample, 3, dtype=torch.float32)
    X_input[:, 0] = torch.tensor(t_Ny, dtype=torch.float32)
    X_input[:, 1] = torch.tensor(t_Py, dtype=torch.float32)
    X_input[:, 2] = torch.tensor(U, dtype=torch.float32)
    
    with torch.no_grad():
        preds = surrogate_model(X_input)
        
    return preds.detach().numpy()

# Movement penalty as negative reward
def reward_fun(stage, xb, xp, d, y):
    if d is None:
        return 0.0
    movement_penalty = 0.15  
    return -movement_penalty * float(np.sum(np.square(d)))

def phys_state_fun(xp, stage, d, y):
    new_xp = np.array(xp) + np.array(d)
    return np.clip(new_xp, -4.0, 4.0)  

phys_state_info = (n_phys_state, init_phys_state, phys_state_fun)

# ============================================================================
# INITIALIZE PG-SOED FRAMEWORK & TRAIN
# ============================================================================
soed = PGsOED(
    model_fun=semiconductor_surrogate_model,
    n_stage=n_stage,
    n_param=n_param,
    n_design=n_design,
    n_obs=n_obs,
    prior_info=prior_info,
    design_bounds=design_bounds,
    noise_info=noise_info,
    reward_fun=reward_fun,
    phys_state_info=phys_state_info,
    n_grid=n_grid,
    post_rvs_method=post_rvs_method,
    random_state=random_state,
    actor_dimns=[80, 80],
    critic_dimns=[80, 80],
)

soed.initialize()

actor_optimizer = optim.Adam(soed.actor_net.parameters(), lr=3e-3)
actor_lr_scheduler = optim.lr_scheduler.ExponentialLR(actor_optimizer, gamma=0.98) 

n_critic_update = 100
critic_optimizer = optim.Adam(soed.critic_net.parameters(), lr=5e-3)
critic_lr_scheduler = optim.lr_scheduler.ExponentialLR(critic_optimizer, gamma=0.98) 

soed.soed(
    n_update=100, 
    n_traj=1000,
    actor_optimizer=actor_optimizer,
    actor_lr_scheduler=actor_lr_scheduler,
    n_critic_update=n_critic_update,
    critic_optimizer=critic_optimizer,
    critic_lr_scheduler=critic_lr_scheduler,
    design_noise_scale=0.3,
    design_noise_decay=0.98, 
)

# ============================================================================
# POSTERIOR CONTOUR PLOTS
# ============================================================================

def plot_semiconductor_trajectories(
    soed_instance, 
    title, 
    file_suffix, 
    theta_val, 
    agent_instance=None,
    x_lims=(0.2, 0.8),  
    y_lims=(-0.8, -0.2), 
    noise_base_scale=0.05
):
    if agent_instance is None:
        agent_instance = soed_instance

    np.random.seed(112)
    
    d_hist = np.zeros((soed_instance.n_stage, soed_instance.n_design))
    y_hist = np.zeros((soed_instance.n_stage, soed_instance.n_obs))
    xp = np.array(soed_instance.init_xp)
    
    xb_list = []
    
    for i in range(soed_instance.n_stage):
        if hasattr(agent_instance, 'get_design'):
            try:
                d_hist[i] = agent_instance.get_design(i, d_hist[:i], y_hist[:i])
            except TypeError:
                d_hist[i] = agent_instance.get_design(i, d_hist=d_hist[:i], y_hist=y_hist[:i])
        else:
            d_hist[i] = soed_instance.get_design(i, d_hist[:i], y_hist[:i])
            
        G = soed_instance.m_f(
            i,
            theta_val.reshape(1, -1),
            d_hist[i].reshape(1, -1),
            xp.reshape(1, -1),
        ).flatten()

        noise_std = noise_base_scale * (1.0 + np.abs(G))
        y_hist[i] = np.random.normal(loc=G, scale=noise_std)
        
        xp = soed_instance.xp_f(xp, i, d_hist[i], y_hist[i])
        
        xb_stage = soed_instance.get_xb(d_hist=d_hist[:i+1], y_hist=y_hist[:i+1])
        xb_list.append(xb_stage)

    all_densities = np.concatenate([xb[:, -1] for xb in xb_list])
    vmin, vmax = np.min(all_densities), np.max(all_densities)
    shared_levels = np.linspace(vmin, vmax, 16)

    fig, axes = plt.subplots(4, 2, figsize=(10, 20))
    axes = axes.flatten()

    for i, xb in enumerate(xb_list):
        n_grid = int(np.sqrt(xb.shape[0]))
        ax = axes[i]
        ax.set_aspect("auto")
        
        cf = ax.contourf(
            xb[:, 0].reshape(n_grid, n_grid),
            xb[:, 1].reshape(n_grid, n_grid),
            xb[:, -1].reshape(n_grid, n_grid),
            cmap="viridis",
            levels=shared_levels,
            vmin=vmin,
            vmax=vmax
        )
        
        cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        
        ax.scatter(
            theta_val[0],
            theta_val[1],
            marker="*",
            s=150,
            c="magenta",
            edgecolors="black",
            zorder=5,
            label="True Parameters"
        )
        
        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("$\\theta_{N,y}$", fontsize=10)
        ax.set_ylabel("$\\theta_{P,y}$", fontsize=10)
        ax.set_title(f"$p(\\theta|I_{{{i}}})$", fontsize=11)
        ax.grid(True, ls="--", alpha=0.5)
        
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"{title}, True $\\theta = ({theta_val[0]}, {theta_val[1]})$", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"semiconductor_trajectory_{file_suffix}.png", dpi=300)
    plt.show()

true_theta = np.array([0.65, -0.3]) 

plot_semiconductor_trajectories(
    agent_instance=soed,            
    soed_instance=soed,             
    title="PG-sOED posterior contours", 
    file_suffix="pg_soed_gaussian_8stage", # UPDATED suffix 
    theta_val=true_theta, 
    x_lims=(0.2, 0.8),  
    y_lims=(-0.8, -0.2),
    noise_base_scale=noise_base_scale
)

# ============================================================================
# EVALUATION APPLIED VOLTAGE & INFORMATION GAIN REWARD COLLECTION
# ============================================================================
print("Running built-in assessment across 10,000 trajectories...")

soed.asses(n_traj=10000)

final_rewards = soed.rewards_hist[:, -1]
design_history = soed.dcs_hist  

mean_reward = np.mean(final_rewards)
std_reward = np.std(final_rewards)
min_reward = np.min(final_rewards)
max_reward = np.max(final_rewards)

print("\n+" + "-"*35 + "+")
print(f"| {'EVALUATION METRICS (10,000 TRAJ)':^33} |")
print("+" + "-"*19 + "+" + "-"*15 + "+")
print(f"| {'Metric':<17} | {'Value':<13} |")
print("+" + "-"*19 + "+" + "-"*15 + "+")
print(f"| {'Mean Reward':<17} | {mean_reward:<13.6f} |")
print(f"| {'Std Dev':<17} | {std_reward:<13.6f} |")
print(f"| {'Min Reward':<17} | {min_reward:<13.6f} |")
print(f"| {'Max Reward':<17} | {max_reward:<13.6f} |")
print("+" + "-"*19 + "+" + "-"*15 + "+\n")

# ============================================================================
# PHYSICAL STATE (BIAS VOLTAGE U_k) ALL 10,000 TRAJECTORIES (DENSITY ENSEMBLE)
# ============================================================================
plt.figure(figsize=(12, 6))

n_traj_eval = design_history.shape[0]
U_states = np.zeros((n_traj_eval, n_stage + 1)) 
current_U = np.full(n_traj_eval, init_phys_state[0])

for k in range(n_stage):
    U_states[:, k] = current_U
    current_U = np.clip(current_U + design_history[:, k, 0], -4.0, 4.0)

U_states[:, n_stage] = current_U

stages = np.arange(n_stage + 1)

# Plot the trajectories using a low alpha for density visualization
plt.plot(
    stages, 
    U_states.T,  
    color="navy", 
    linewidth=1.0, 
    alpha=0.015
)

# Baseline reference line at U_0 = 0
plt.axhline(0.0, color="red", linestyle="--", linewidth=1.2, alpha=0.7)

plt.xlim(0, n_stage)
plt.ylim(-4.1, 4.1)
plt.xticks(stages, [f"Stage {k}" for k in stages], fontsize=11)
plt.xlabel("Stage $k$", fontsize=12)
plt.ylabel("Physical State (Voltage $U_k$)", fontsize=12)
plt.title("Physical state trajectories ($U_k$)", fontsize=14)
plt.grid(True, ls=":", alpha=0.5)

plt.tight_layout()
plt.savefig("semiconductor_design_trajectories_all_10k.png", dpi=300)
plt.show()

# ============================================================================
# REWARD HISTOGRAM PLOT
# ============================================================================
plt.figure(figsize=(6, 4))
bins_reward = np.linspace(0, 6, 80)

plt.hist(final_rewards, alpha=0.7, bins=bins_reward, color='c', label='PG-sOED', edgecolor='none')
plt.xlim(0, 3)
plt.xlabel('Reward', fontsize=11)
plt.ylabel('Counts', fontsize=11)
plt.title('Histogram of rewards', fontsize=11)
plt.legend(loc='upper right', fontsize=9)
plt.grid(True, ls=':', alpha=0.5)

plt.tight_layout()
plt.savefig("figure_rewards_histograms.png", dpi=300)
plt.show()