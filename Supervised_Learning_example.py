import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# We consider the following six cases:
CASES = {
    1: {"depth": 8, "width": 128, "activation": nn.ReLU(), "num_points": 15, "learning_rate": 0.025,   "weight_decay": 0.0},
    2: {"depth": 8, "width": 128, "activation": nn.ReLU(), "num_points": 15, "learning_rate": 0.00001, "weight_decay": 0.0},
    3: {"depth": 8, "width": 128, "activation": nn.ReLU(), "num_points": 15, "learning_rate": 0.0025,  "weight_decay": 0.0},
    4: {"depth": 8, "width": 128, "activation": nn.ReLU(), "num_points": 15, "learning_rate": 0.0025,  "weight_decay": 0.02},
    5: {"depth": 8, "width": 128, "activation": nn.ReLU(), "num_points": 15, "learning_rate": 0.0025,  "weight_decay": 0.003},
    6: {"depth": 8, "width": 128, "activation": nn.SiLU(), "num_points": 15, "learning_rate": 0.0025,  "weight_decay": 0.00075},
}

# Construct DNN using given depth, width and activation function
class DNN(nn.Module):
    def __init__(self, depth, width, activation):
        super(DNN, self).__init__()
        layers = []
        layers.append(nn.Linear(1, width))
        layers.append(activation)
        for _ in range(depth - 2):
            layers.append(nn.Linear(width, width))
            layers.append(activation)   
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Define the true complex target function
def true_function(x):
    return np.sin(x) * np.exp(-0.1 * x)

# Setup a large grid on [-2, 10] for evaluation
x_big_grid_np = np.linspace(-2, 10, 1000)
X_big_grid = torch.tensor(x_big_grid_np, dtype=torch.float32).unsqueeze(1)
y_true_grid = true_function(x_big_grid_np)

# Initialize the 3x2 figure layout
fig, axes = plt.subplots(3, 2, figsize=(15, 15), sharex=True, sharey=True)
axes = axes.flatten()

# Loop through all 6 cases 
for idx, case_id in enumerate(sorted(CASES.keys())):
    cfg = CASES[case_id]
    ax = axes[idx]

    # Keep same seed
    np.random.seed(77)
    torch.manual_seed(77)

    # Generate training data
    x_train_np = np.random.uniform(-2, 10, cfg["num_points"]) 
    noise = np.random.normal(0, 0.1, cfg["num_points"])
    y_train_np = true_function(x_train_np) + noise
    X_train = torch.tensor(x_train_np, dtype=torch.float32).unsqueeze(1)
    Y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)

    # Initialize model and optimizer
    model = DNN(depth=cfg["depth"], width=cfg["width"], activation=cfg["activation"])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    # Train model
    epochs = 2000
    for epoch in range(epochs):
        predictions = model(X_train)
        loss = criterion(predictions, Y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate model
    model.eval()
    with torch.no_grad():
        Y_pred = model(X_big_grid).numpy().ravel()

    # Calculate L2 norm error
    l2_error = np.sqrt(np.trapz((y_true_grid - Y_pred) ** 2, x_big_grid_np))
    print(f"Case {case_id} -> L2-norm error: {l2_error:.4f}")

    # Add plots to the current subplot axis
    ax.plot(x_big_grid_np, y_true_grid, 'g--', alpha=0.7, label=r"$f^*(x)$ (True)")
    ax.scatter(x_train_np, y_train_np, color='red', s=40, zorder=5, label="Data (Noisy)")
    ax.plot(x_big_grid_np, Y_pred, 'b-', label=r"$f(x, \lambda)$ (DNN)")

    # Titles and formatting per subplot
    ax.set_title(f"Case {case_id} (L2 Error: {l2_error:.3f})", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)

    # Only show labels on bottom rows and left columns to prevent layout clutter
    if idx >= 4:
        ax.set_xlabel("Input $x$")
    if idx % 2 == 0:
        ax.set_ylabel("Output $y$")

# Plot adjustments
axes[0].legend(loc="upper right")
plt.tight_layout()
plt.savefig('DNN_All_Cases_Comparison.png', dpi=300, bbox_inches='tight')
plt.show()