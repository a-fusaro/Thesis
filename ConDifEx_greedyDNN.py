import copy
import numpy as np
np.typeDict = np.sctypeDict 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================================
# PARAMETERS & PROBLEM CONFIGURATION
# ============================================================================
n_stage = 2
n_param = 2
n_design = 2
n_obs = 1
n_phys_state = 2
n_grid = 50

prior_type = "uniform"
prior_loc = 0
prior_scale = 1
prior_info = [(prior_type, prior_loc, prior_scale), (prior_type, prior_loc, prior_scale)]

design_bounds = [(-0.25, 0.25), (-0.25, 0.25)]

noise_loc = 0
noise_base_scale = 0.05
noise_ratio_scale = 0.05
noise_info = [(noise_loc, noise_base_scale, noise_ratio_scale)]

init_phys_state = (0.5, 0.5)
post_rvs_method = "Rejection"

random_state = 2021
np.random.seed(random_state)
torch.manual_seed(random_state)

from oed_package.pg_soed import PGsOED

def conv_diff_model(stage, theta, d, xp=None):
    n_sample = max(len(theta), len(d), len(xp))
    X = torch.zeros(n_sample, 4).double()
    X[:, :2] = torch.from_numpy(theta)
    X[:, 2:] = torch.from_numpy(xp + d)
    return conv_diff_nets[stage](X).detach().numpy()

def reward_fun(stage, xb, xp, d, y):
    return 0
    
def phys_state_fun(xp, stage, d, y):
    return np.clip(xp + d, 0.0, 1.0)

phys_state_info = (n_phys_state, init_phys_state, phys_state_fun)

# ============================================================================
# INDEPENDENT EVALUATOR FUNCTION
# ============================================================================
def evaluate_agent_independently(soed_instance, agent_get_design, n_traj=100):
    rewards_hist = np.zeros((n_traj, soed_instance.n_stage))
    dcs_hist = np.zeros((n_traj, soed_instance.n_stage, soed_instance.n_design))
    
    for ep in range(n_traj):
        theta = np.random.uniform(0, 1, size=(soed_instance.n_param,))
        xp = np.array(soed_instance.init_xp)
        d_hist = []
        y_hist = []
        
        for t in range(soed_instance.n_stage):
            d = agent_get_design(t, d_hist, y_hist)
            
            G = soed_instance.m_f(t, theta.reshape(1, -1), d.reshape(1, -1), xp.reshape(1, -1)).flatten()
            noise_std = noise_base_scale * (1.0 + np.abs(G))
            y = np.random.normal(loc=G, scale=noise_std)
            
            d_hist.append(d)
            y_hist.append(y)
            xp = soed_instance.xp_f(xp, t, d, y)
            
            xb = soed_instance.get_xb(d_hist=np.array(d_hist), y_hist=np.array(y_hist))
            p = xb[:, -1]
            reward = np.sum(p * np.log(p + 1e-12)) / (soed_instance.n_grid ** 2)
            
            rewards_hist[ep, t] = reward
            dcs_hist[ep, t, :] = d
            
    return rewards_hist, dcs_hist


# ============================================================================
# GREEDY AND BATCH AGENT IMPLEMENTATIONS
# ============================================================================
class PGGreedyAgent:
    def __init__(self, soed_instance):
        self.soed = soed_instance
        # Single-DNN architecture matching the paper
        self.policy_net = copy.deepcopy(soed_instance.actor_net).double()
        self.critic_net = copy.deepcopy(soed_instance.critic_net).double()
        self.rewards_hist = None
        self.dcs_hist = None
        
        # Calculate the dimension: N + (N-1)*Nd + (N-1)*Ny
        self.N = self.soed.n_stage
        self.Nd = self.soed.n_design
        self.Ny = getattr(self.soed, 'n_obs', 1) # Fallback to 1 if not explicitly set
        self.target_dim = self.N + (self.N - 1) * self.Nd + (self.N - 1) * self.Ny
        
        # Modify the first layer of the nets to accept the calculated dimension
        self._adapt_input_layer(self.policy_net, self.target_dim)
        self._adapt_input_layer(self.critic_net, self.target_dim)

    def _adapt_input_layer(self, net, target_dim):
        for name, module in net.named_children():
            if isinstance(module, nn.Linear):
                setattr(net, name, nn.Linear(target_dim, module.out_features).double())
                return True
            else:
                if self._adapt_input_layer(module, target_dim): 
                    return True
        return False

    def _get_input_tensor(self, stage, d_hist=None, y_hist=None):
        # One-hot encoding of the stage index
        ek = np.zeros(self.N)
        ek[int(stage)] = 1.0
        
        # Zero-padded design history
        d_pad = np.zeros((self.N - 1) * self.Nd)
        if d_hist is not None and len(d_hist) > 0:
            d_flat = np.array(d_hist).flatten()
            d_pad[:len(d_flat)] = d_flat
            
        # Zero-padded observation history
        y_pad = np.zeros((self.N - 1) * self.Ny)
        if y_hist is not None and len(y_hist) > 0:
            y_flat = np.array(y_hist).flatten()
            y_pad[:len(y_flat)] = y_flat
            
        # Concatenate to match exactly the specified equation of the paper
        feat = np.concatenate([ek, d_pad, y_pad])
        return torch.tensor(feat, dtype=torch.float64).unsqueeze(0)
    
    def get_design(self, stage, d_hist=None, y_hist=None):
        if isinstance(d_hist, dict):
            y_hist = d_hist.get('y_hist', [])
            d_hist = d_hist.get('d_hist', [])
            
        inp = self._get_input_tensor(stage, d_hist, y_hist)
        
        with torch.no_grad():
            out = self.policy_net(inp).cpu().numpy().flatten()
            
        for j in range(self.soed.n_design):
            bnd = self.soed.design_bounds[j]
            out[j] = np.clip(out[j], bnd[0], bnd[1])
        return out

    def train_offline(self, num_updates=500, batch_size=1000, noise_base_scale=0.05):
        print("\nTraining PG-Greedy Agent:")
        
        opt_actor = optim.Adam(self.policy_net.parameters(), lr=0.005)
        opt_critic = optim.Adam(self.critic_net.parameters(), lr=0.005)
        
        scheduler_actor = optim.lr_scheduler.ExponentialLR(opt_actor, gamma=0.995)
        scheduler_critic = optim.lr_scheduler.ExponentialLR(opt_critic, gamma=0.995)
        
        design_noise_scale = 0.1
        design_noise_decay = 0.995
        
        for update in tqdm(range(num_updates), desc="Greedy Updates"):
            opt_actor.zero_grad()
            opt_critic.zero_grad()
            
            log_probs_list = [[] for _ in range(self.soed.n_stage)]
            values_list = [[] for _ in range(self.soed.n_stage)]
            rewards_list = [[] for _ in range(self.soed.n_stage)]
            
            for ep in range(batch_size):
                theta = np.random.uniform(0, 1, size=(self.soed.n_param,))
                xp = np.array(self.soed.init_xp)
                d_hist_ep, y_hist_ep = [], []
                prev_kl = 0.0
                
                for t in range(self.soed.n_stage):
                    # Same tensor builder handles both dynamically
                    inp_tensor = self._get_input_tensor(t, d_hist_ep, y_hist_ep)
                    
                    action_mean = self.policy_net(inp_tensor)
                    value = self.critic_net(inp_tensor)
                    
                    noise = torch.randn_like(action_mean, dtype=torch.float64) * design_noise_scale
                    action_taken = (action_mean + noise).detach()
                    
                    log_term = torch.tensor([np.log(2 * np.pi * design_noise_scale**2)], dtype=torch.float64)
                    log_prob = -0.5 * torch.sum(((action_taken - action_mean) / design_noise_scale)**2 + log_term)
                    
                    d = action_taken.cpu().numpy().flatten()
                    for j in range(self.soed.n_design):
                        bnd = self.soed.design_bounds[j]
                        d[j] = np.clip(d[j], bnd[0], bnd[1])
                    d_hist_ep.append(d)
                    
                    G = self.soed.m_f(t, theta.reshape(1,-1), d.reshape(1,-1), xp.reshape(1,-1)).flatten()
                    obs_noise_std = noise_base_scale * (1.0 + np.abs(G))
                    y = np.random.normal(loc=G, scale=obs_noise_std)
                    y_hist_ep.append(y)
                    xp = self.soed.xp_f(xp, t, d, y)
                    
                    xb_curr = self.soed.get_xb(d_hist=np.array(d_hist_ep), y_hist=np.array(y_hist_ep))
                    p_curr = xb_curr[:, -1]
                    curr_kl = float(np.sum(p_curr * np.log(p_curr + 1e-12)) / (self.soed.n_grid ** 2))
                    
                    step_reward = curr_kl - prev_kl
                    prev_kl = curr_kl
                    
                    log_probs_list[t].append(log_prob)
                    values_list[t].append(value)
                    rewards_list[t].append(step_reward)
            
            # Loss calculation according to the equation specified in the paper
            actor_loss = 0.0
            critic_loss = 0.0
            
            for t in range(self.soed.n_stage):
                st_log_probs = torch.stack(log_probs_list[t]).double()
                st_values = torch.cat(values_list[t]).squeeze().double()
                st_rewards = torch.tensor(rewards_list[t], dtype=torch.float64)
                
                # Calculate advantage 
                adv = st_rewards - st_values.detach()
                    
                actor_loss += -(st_log_probs * adv).mean()
                critic_loss += F.mse_loss(st_values, st_rewards)
            
            actor_loss.backward()
            critic_loss.backward()
            opt_actor.step()
            opt_critic.step()
            scheduler_actor.step()
            scheduler_critic.step()
            
            design_noise_scale *= design_noise_decay

    def asses(self, n_episodes=10000):
        self.rewards_hist, self.dcs_hist = evaluate_agent_independently(self.soed, self.get_design, n_episodes)
        return f"PG-Greedy Average Terminal Reward: {np.mean(self.rewards_hist[:, -1]):.4f}"
    
class PGBatchAgent:
    def __init__(self, soed_instance):
        self.soed = soed_instance
        self.policy_net = copy.deepcopy(soed_instance.actor_net).double()
        self.critic_net = copy.deepcopy(soed_instance.critic_net).double()
        self.rewards_hist = None
        self.dcs_hist = None
        
        # Modify input layers to accept one-hot encoded stage index e_{k+1}
        self._reduce_to_one_hot(self.policy_net, self.soed.n_stage)
        self._reduce_to_one_hot(self.critic_net, self.soed.n_stage)

    def _reduce_to_one_hot(self, net, n_stages):
        for name, module in net.named_children():
            if isinstance(module, nn.Linear):
                setattr(net, name, nn.Linear(n_stages, module.out_features).double())
                return True
            else:
                if self._reduce_to_one_hot(module, n_stages): 
                    return True
        return False

    def _get_input_tensor(self, stage):
        # Generate one-hot vector for stage index e_{k+1}
        one_hot = np.zeros(self.soed.n_stage)
        one_hot[int(stage)] = 1.0
        return torch.tensor(one_hot, dtype=torch.float64).unsqueeze(0)

    def get_design(self, stage, d_hist=None, y_hist=None):
        inp = self._get_input_tensor(stage)
        with torch.no_grad():
            out = self.policy_net(inp).cpu().numpy().flatten()
            
        for j in range(self.soed.n_design):
            bnd = self.soed.design_bounds[j]
            out[j] = np.clip(out[j], bnd[0], bnd[1])
        return out

    def train_offline(self, num_updates=500, batch_size=1000, noise_base_scale=0.05):
        print("\nTraining PG-Batch Agent:")
        design_noise_scale = 0.1
        design_noise_decay = 0.995
        
        opt_actor = optim.Adam(self.policy_net.parameters(), lr=0.005)
        opt_critic = optim.Adam(self.critic_net.parameters(), lr=0.005)
        scheduler_actor = optim.lr_scheduler.ExponentialLR(opt_actor, gamma=0.995)
        scheduler_critic = optim.lr_scheduler.ExponentialLR(opt_critic, gamma=0.995)
        
        for update in tqdm(range(num_updates), desc="Batch Updates"):
            opt_actor.zero_grad()
            opt_critic.zero_grad()
            
            log_probs = [[] for _ in range(self.soed.n_stage)]
            values = [[] for _ in range(self.soed.n_stage)]
            term_rewards = []
            
            for ep in range(batch_size):
                theta = np.random.uniform(0, 1, size=(self.soed.n_param,))
                xp = np.array(self.soed.init_xp)
                d_hist_ep, y_hist_ep, ep_log_probs, ep_values = [], [], [], []
                
                for t in range(self.soed.n_stage):
                    inp = self._get_input_tensor(t)
                    action_mean = self.policy_net(inp)
                    value = self.critic_net(inp)
                    
                    noise = torch.randn_like(action_mean, dtype=torch.float64) * design_noise_scale
                    action_taken = (action_mean + noise).detach()
                    
                    log_term = torch.tensor([np.log(2 * np.pi * design_noise_scale**2)], dtype=torch.float64)
                    log_prob = -0.5 * torch.sum(((action_taken - action_mean) / design_noise_scale)**2 + log_term)
                    
                    d = action_taken.cpu().numpy().flatten()
                    for j in range(self.soed.n_design):
                        bnd = self.soed.design_bounds[j]
                        d[j] = np.clip(d[j], bnd[0], bnd[1])
                    d_hist_ep.append(d)
                    
                    G = self.soed.m_f(t, theta.reshape(1,-1), d.reshape(1,-1), xp.reshape(1,-1)).flatten()
                    noise_std = noise_base_scale * (1.0 + np.abs(G))
                    y = np.random.normal(loc=G, scale=noise_std)
                    y_hist_ep.append(y)
                    xp = self.soed.xp_f(xp, t, d, y)
                    
                    ep_log_probs.append(log_prob)
                    ep_values.append(value)
                    
                # Terminal reward calculation at stage N
                xb = self.soed.get_xb(d_hist=np.array(d_hist_ep), y_hist=np.array(y_hist_ep))
                p = xb[:, -1]
                term_rewards.append(float(np.sum(p * np.log(p + 1e-12)) / (self.soed.n_grid ** 2)))
                
                for t in range(self.soed.n_stage):
                    log_probs[t].append(ep_log_probs[t])
                    values[t].append(ep_values[t])
                    
            actor_loss = torch.tensor(0.0, dtype=torch.float64)
            critic_loss = torch.tensor(0.0, dtype=torch.float64)
            term_rewards_tensor = torch.tensor(term_rewards, dtype=torch.float64)
            
            for t in range(self.soed.n_stage):
                st_log_probs = torch.stack(log_probs[t]).double()
                st_values = torch.cat(values[t]).squeeze().double()
                
                # Calculate advantage
                adv = term_rewards_tensor - st_values.detach()
                
                actor_loss = actor_loss - (st_log_probs * adv).mean()
                critic_loss = critic_loss + F.mse_loss(st_values, term_rewards_tensor)
            
            actor_loss.backward()
            critic_loss.backward()
            opt_actor.step()
            opt_critic.step()
            scheduler_actor.step()
            scheduler_critic.step()
            
            design_noise_scale *= design_noise_decay

    def asses(self, n_episodes=10000):
        self.rewards_hist, self.dcs_hist = evaluate_agent_independently(self.soed, self.get_design, n_episodes)
        return f"PG-Batch Average Terminal Reward: {np.mean(self.rewards_hist[:, -1]):.4f}"
    
# ============================================================================
# INITIALIZATION AND TRAINING 
# ============================================================================
conv_diff_nets = (torch.load("conv_diff_net_t0.05.pt", weights_only=False),
                  torch.load("conv_diff_net_t0.2.pt", weights_only=False))

soed = PGsOED(model_fun=conv_diff_model, n_stage=n_stage, n_param=n_param,
              n_design=n_design, n_obs=n_obs, prior_info=prior_info,
              design_bounds=design_bounds, noise_info=noise_info,
              reward_fun=reward_fun, phys_state_info=phys_state_info,
              n_grid=n_grid, post_rvs_method=post_rvs_method,
              random_state=random_state, actor_dimns=[80, 80], critic_dimns=[80, 80])

soed.initialize()

greedy_agent = PGGreedyAgent(soed)
batch_agent = PGBatchAgent(soed)

actor_optimizer = optim.Adam(soed.actor_net.parameters(), lr=0.005)
actor_lr_scheduler = optim.lr_scheduler.ExponentialLR(actor_optimizer, gamma=0.995)

n_critic_update = 100
critic_optimizer = optim.Adam(soed.critic_net.parameters(), lr=0.005)
critic_lr_scheduler = optim.lr_scheduler.ExponentialLR(critic_optimizer, gamma=0.995)

print("Training PG-sOED...")
soed.soed(n_update = 300, n_traj = 1000,
          actor_optimizer=actor_optimizer, actor_lr_scheduler=actor_lr_scheduler,
          n_critic_update=n_critic_update, critic_optimizer=critic_optimizer,
          critic_lr_scheduler=critic_lr_scheduler,
          design_noise_scale = 0.1, design_noise_decay = 0.995)

print("Evaluating PG-sOED Policy...")
soed.asses(10000)

greedy_agent.train_offline(num_updates = 300, batch_size = 1000) 
print("Evaluating PG-Greedy Policy Baseline...")
print(greedy_agent.asses(n_episodes = 10000))

batch_agent.train_offline(num_updates = 300, batch_size = 1000) 
print("Evaluating PG-Batch Policy Baseline...")
print(batch_agent.asses(n_episodes = 10000))

# ============================================================================
# POSTERIOR CONTOUR PLOTS 
# ============================================================================
def simulate_trajectory(agent_instance, theta_val):
    np.random.seed(112)
    d_hist = np.zeros((soed.n_stage, soed.n_design))
    y_hist = np.zeros((soed.n_stage, soed.n_obs))
    xp = np.array(soed.init_xp)
    
    history = []
    
    for i in range(soed.n_stage + 1):
        if i > 0:
            xb = soed.get_xb(d_hist=d_hist[:i], y_hist=y_hist[:i])
            history.append({
                'xb': xb,
                'xp': xp.copy(),
                'd': d_hist[i - 1].copy()
            })
            
        if i < soed.n_stage:
            if hasattr(agent_instance, 'get_design'):
                d_hist[i] = agent_instance.get_design(i, d_hist[:i], y_hist[:i])
            else:
                d_hist[i] = agent_instance.get_design(i, d_hist=d_hist[:i], y_hist=y_hist[:i])
            
            G = soed.m_f(i, theta_val.reshape(1, -1), d_hist[i].reshape(1, -1), xp.reshape(1, -1)).flatten()
            
            noise_std = noise_base_scale * (1.0 + np.abs(G))
            y_hist[i] = np.random.normal(loc=G, scale=noise_std)
            xp = soed.xp_f(xp, i, d_hist[i], y_hist[i])
            
    return history


def plot_saved_trajectory(title, file_suffix, theta_val, history, vmin, vmax, levels):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ticks = np.arange(0.0, 1.1, 0.2)
    
    for i, stage_data in enumerate(history):
        xb = stage_data['xb']
        xp = stage_data['xp']
        d = stage_data['d']
        
        ax = axes[i]
        ax.set_aspect('equal')
        
        # Apply the shared vmin, vmax, and explicit levels
        cf = ax.contourf(
            xb[:, 0].reshape(n_grid, n_grid), 
            xb[:, 1].reshape(n_grid, n_grid),
            xb[:, -1].reshape(n_grid, n_grid), 
            cmap='viridis', 
            levels=levels,
            vmin=vmin,
            vmax=vmax
        )
        cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        
        ax.scatter(theta_val[0], theta_val[1], marker='*', s=150, c='magenta', zorder=5)
        ax.scatter(xp[0], xp[1], marker='o', s=80, c='orangered', zorder=4)
        ax.plot([xp[0] - d[0], xp[0]], [xp[1] - d[1], xp[1]], color='orangered', linewidth=1.5, zorder=3)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(labelsize=8)
        ax.set_xlabel('$Z_x$', fontsize=10)
        ax.set_ylabel('$Z_y$', fontsize=10)
        ax.set_title(f"$p(\\theta|I_{i+1})$", fontsize=11)
        ax.grid(True, ls='--', alpha=0.5)
        
    fig.suptitle(f"{title}, $\\theta=({theta_val[0]}, {theta_val[1]})$", fontsize=11, y=0.98)
    plt.tight_layout()
    plt.savefig(f"figure_trajectory_{file_suffix}.png", dpi=300)
    plt.show()

# Provide our theta
theta_val = np.array([0.65, 0.7])
agents_to_run = [
    (soed, "(a) PG-sOED", "pg_soed"),
    (greedy_agent, "(b) Greedy", "greedy"),
    (batch_agent, "(c) Batch", "batch")
]

all_trajectories = []

# Run the simulations and collect the history
for agent, title, suffix in agents_to_run:
    hist = simulate_trajectory(agent, theta_val)
    all_trajectories.append((title, suffix, hist))

pg_batch_min, pg_batch_max = np.inf, -np.inf
greedy_min, greedy_max = np.inf, -np.inf

# Find min and max for each respective group to generate shared scales
for title, suffix, hist in all_trajectories:
    for stage_data in hist:
        density_vals = stage_data['xb'][:, -1]
        c_min = np.min(density_vals)
        c_max = np.max(density_vals)
        
        if suffix in ['pg_soed', 'batch']:
            pg_batch_min = min(pg_batch_min, c_min)
            pg_batch_max = max(pg_batch_max, c_max)
        elif suffix == 'greedy':
            greedy_min = min(greedy_min, c_min)
            greedy_max = max(greedy_max, c_max)

pg_batch_levels = np.linspace(pg_batch_min, pg_batch_max, 16)
greedy_levels = np.linspace(greedy_min, greedy_max, 16)

# Generate the plots
for title, suffix, hist in all_trajectories:
    if suffix in ['pg_soed', 'batch']:
        plot_saved_trajectory(
            title, suffix, theta_val, hist, 
            pg_batch_min, pg_batch_max, pg_batch_levels
        )
    elif suffix == 'greedy':
        plot_saved_trajectory(
            title, suffix, theta_val, hist, 
            greedy_min, greedy_max, greedy_levels
        )

# ============================================================================
# REWARD HISTOGRAMS PLOT
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
bins = np.linspace(0, 6, 80)

axes[0].hist(soed.rewards_hist[:, -1], alpha=0.7, bins=bins, color='c', label='PG-sOED', edgecolor='none')
axes[0].hist(greedy_agent.rewards_hist[:, -1], alpha=0.7, bins=bins, color='coral', label='Greedy', edgecolor='none')
axes[0].set_xlim(0, 6)
axes[0].set_xlabel('Reward', fontsize=11)
axes[0].set_ylabel('Counts', fontsize=11)
axes[0].set_title('(a) sOED versus greedy', fontsize=11)
axes[0].legend(loc='upper right', fontsize=9)
axes[0].grid(True, ls=':', alpha=0.5)

axes[1].hist(soed.rewards_hist[:, -1], alpha=0.7, bins=bins, color='steelblue', label='PG-sOED', edgecolor='none')
axes[1].hist(batch_agent.rewards_hist[:, -1], alpha=0.7, bins=bins, color='plum', label='Batch', edgecolor='none')
axes[1].set_xlim(0, 6)
axes[1].set_xlabel('Reward', fontsize=11)
axes[1].set_ylabel('Counts', fontsize=11)
axes[1].set_title('(b) sOED versus batch', fontsize=11)
axes[1].legend(loc='upper right', fontsize=9)
axes[1].grid(True, ls=':', alpha=0.5)

plt.suptitle("Histograms of total rewards from evaluation episodes", fontsize=12, y=0.98)
plt.tight_layout()
plt.savefig("figure_rewards_histograms.png", dpi=300)
plt.show()

# ============================================================================
# EPISODE TRACES PLOT
# ============================================================================
def extract_positions(agent_hist):
    n_episodes = agent_hist.shape[0]
    x1 = np.zeros((n_episodes, 2))
    x2 = np.zeros((n_episodes, 2))
    
    for ep in range(n_episodes):
        x0 = np.array(soed.init_xp)
        
        # Physical boundary clipping [0.0, 1.0]
        x1[ep] = np.clip(x0 + agent_hist[ep, 0, :], 0.0, 1.0)
        x2[ep] = np.clip(x1[ep] + agent_hist[ep, 1, :], 0.0, 1.0)
        
    return x1, x2

pg_x1, pg_x2 = extract_positions(soed.dcs_hist)
gr_x1, gr_x2 = extract_positions(greedy_agent.dcs_hist)
bt_x1, bt_x2 = extract_positions(batch_agent.dcs_hist)

fig, axes = plt.subplots(2, 3, figsize=(10, 7.5))

axis_ticks = np.linspace(0.0, 1.0, 11)
tick_labels = ['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']

# Row 1: x1,p
axes[0, 0].scatter(pg_x1[:, 0], pg_x1[:, 1], c='teal', s=15, alpha=0.6, edgecolors='none')
axes[0, 0].set_title("PG-sOED", fontsize=12)
axes[0, 0].set_ylabel('$x_{1,p}^{(2)}$', fontsize=11)
axes[0, 0].set_xlabel('$x_{1,p}^{(1)}$', fontsize=11)

axes[0, 1].scatter(gr_x1[:, 0], gr_x1[:, 1], c='orangered', s=15, alpha=0.6, edgecolors='none')
axes[0, 1].set_title("Greedy", fontsize=12)
axes[0, 1].set_xlabel('$x_{1,p}^{(1)}$', fontsize=11)

axes[0, 2].scatter(bt_x1[:, 0], bt_x1[:, 1], c='purple', s=15, alpha=0.6, edgecolors='none')
axes[0, 2].set_title("Batch", fontsize=12)
axes[0, 2].set_xlabel('$x_{1,p}^{(1)}$', fontsize=11)

# Row 2: x2,p
axes[1, 0].scatter(pg_x2[:, 0], pg_x2[:, 1], c='teal', s=15, alpha=0.6, edgecolors='none')
axes[1, 0].set_ylabel('$x_{2,p}^{(2)}$', fontsize=11)
axes[1, 0].set_xlabel('$x_{2,p}^{(1)}$', fontsize=11)

axes[1, 1].scatter(gr_x2[:, 0], gr_x2[:, 1], c='orangered', s=15, alpha=0.6, edgecolors='none')
axes[1, 1].set_xlabel('$x_{2,p}^{(1)}$', fontsize=11)

axes[1, 2].scatter(bt_x2[:, 0], bt_x2[:, 1], c='purple', s=15, alpha=0.6, edgecolors='none')
axes[1, 2].set_xlabel('$x_{2,p}^{(1)}$', fontsize=11)

for ax in axes.flat:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(axis_ticks)
    ax.set_yticks(axis_ticks)
    ax.set_xticklabels(tick_labels, fontsize=6)
    ax.set_yticklabels(tick_labels, fontsize=6)
    ax.grid(True, which='both', linestyle='-', color='lightgray', linewidth=0.5)
    ax.set_aspect('equal')

plt.suptitle("Vehicle locations of episodes obtained from PG-sOED, greedy, and batch designs.", fontsize=11, y=0.98)
plt.tight_layout()
plt.savefig("figure_node_distributions.png", dpi=300)
plt.show()

# ============================================================================
# RESULTS SUMMARY TABLE
# ============================================================================
# Fallback check to ensure PG-sOED rewards_hist is populated
if not hasattr(soed, 'rewards_hist') or soed.rewards_hist is None:
    soed.rewards_hist, _ = evaluate_agent_independently(soed, soed.actor_net_predict, n_traj=10000)

results = [
    ("PG-sOED", np.mean(soed.rewards_hist[:, -1]), np.std(soed.rewards_hist[:, -1])),
    ("PG-Greedy", np.mean(greedy_agent.rewards_hist[:, -1]), np.std(greedy_agent.rewards_hist[:, -1])),
    ("PG-Batch", np.mean(batch_agent.rewards_hist[:, -1]), np.std(batch_agent.rewards_hist[:, -1]))
]

print("\n" + "="*42)
print(f"{'Method':<12} | {'Mean Reward':<12} | {'Std Reward':<12}")
print("-" * 42)
for method, mean, std in results:
    print(f"{method:<12} | {mean:<12.4f} | {std:<12.4f}")
print("="*42)