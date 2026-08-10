import numpy as np

# Setup environment
GRID_SIZE = 5
START_STATE = (0, 0)
GOAL_STATE = (4, 4)

# Wall coordinates
WALL_STATES = [(0, 1), (0, 4), (1, 2), (2, 0), (4, 2)]

# Poison square coordinate
PIT_STATES = [(3, 3)]

# Set of actions to choose from at each state: Left, Up, Right, Down
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
NUM_ACTIONS = len(ACTIONS)

def step(state, action_idx):
    # Calculates transitions and rewards
    if state == GOAL_STATE:
        # first entry tracks position, second entry tracks reward/penalty, last entry tracks if we reached the goal
        return state, 0, True

    # Transition
    move = ACTIONS[action_idx]
    next_state = (state[0] + move[0], state[1] + move[1])

    # Specify penalties
    if (
        next_state[0] < 0
        or next_state[0] >= GRID_SIZE
        or next_state[1] < 0
        or next_state[1] >= GRID_SIZE
        or next_state in WALL_STATES
    ):
        return state, -0.5, False  # Bumping penalty, remain at same square
    if next_state == GOAL_STATE:
        return next_state, 50.0, True  # Positive target reward
    if next_state in PIT_STATES:
        return next_state, -20.0, False  # Negative penalty from poison

    return next_state, -0.1, False  # Step cost

# Initialize policy parameters
all_states = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
non_wall_states = [s for s in all_states if s not in WALL_STATES]

# Actor parameters (policy logits) and critic parameters (state values)
actor_params = {s: np.zeros(NUM_ACTIONS) for s in non_wall_states}
critic_values = {s: 0.0 for s in non_wall_states}

def get_action_probabilities(state):
    # Softmax function turning policy parameters into action choices
    p = actor_params[state]
    exp_p = np.exp(p - np.max(p))
    return exp_p / np.sum(exp_p)

# Training loop matching the actor critic algorithm
alpha_actor = 0.05   # alpha^w
alpha_critic = 0.05  # alpha^nu
gamma = 1            # gamma
num_episodes = 500   # E

print("Training the Actor-Critic model ...\n")

# Loop for episodes
for episode in range(num_episodes):
    S = START_STATE
    I = 1.0  # Tracks discounting (would only be relevant if gamma < 1)
    trajectory = []
    done = False
    step_count = 0

    # Loop for t
    while not done and step_count < 30:
        # Generate an action A from pi_w(.|S)
        probs = get_action_probabilities(S)
        A = np.random.choice(NUM_ACTIONS, p=probs)

        # Take action A, observe S_prime (next_state) and R (reward)
        S_prime, R, done = step(S, A)
        trajectory.append((S, A, R, S_prime))

        # Calculate delta = R + gamma * v(S_prime) - v(S)
        # Handling the environment boundary condition for walls
        if S_prime in WALL_STATES:
            v_S_prime = critic_values[S]
        elif done:  # If S_prime is terminal, v(S_prime) = 0
            v_S_prime = 0.0
        else:
            v_S_prime = critic_values[S_prime]

        delta = R + gamma * v_S_prime - critic_values[S]

        # Update critic parameters nu
        # The tabular value gradient \nabla_\nu v(S,\nu) is exactly 1
        grad_v = 1.0
        critic_values[S] += alpha_critic * delta * grad_v

        # Update policy parameters w
        # The analytical Softmax gradient \nabla_w \ln \pi_w(A|S)
        grad_ln_pi = -probs
        grad_ln_pi[A] += 1.0  # The partial derivative of the chosen action has an additional 1 as derived

        # w = w + alpha^w * I * delta * gradient
        actor_params[S] += alpha_actor * I * delta * grad_ln_pi

        # Update I and S
        I = gamma * I
        S = S_prime
        step_count += 1

    if (episode + 1) % 10 == 0 or episode == 0:
        total_raw_reward = sum(x[2] for x in trajectory)
        print(f"Episode {episode + 1:4d}: Total Reward = {total_raw_reward:6.1f} | Steps Traveled = {step_count}")

# Illustrate resulting optimized policy
# For each state plot action with highest probability
action_symbols = ['↑', '→', '↓', '←']
for r in range(GRID_SIZE):
    row_str = ""
    for c in range(GRID_SIZE):
        state = (r, c)
        if state in WALL_STATES:
            row_str += " [W] "
        else:
            probs = get_action_probabilities(state)
            best_action = np.argmax(probs)
            symbol = action_symbols[best_action]

            if state == START_STATE:
                row_str += f" S{symbol}  "
            elif state == GOAL_STATE:
                row_str += f" G{symbol}  "
            elif state in PIT_STATES:
                row_str += f" P{symbol}  "
            else:
                row_str += f"  {symbol}  "
    print(row_str)