# Parameters Setup

## best model 4:

episodes = 20000
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
bufffer_capacity = 100000
gamma = 0.99
batch_size = 64
learning_rate = 1e-4
epsilon_start = 1.0
epsilon_min = 0.01
epsilon_decay = 1e5
target_update = 1000
print_step = 10
live_reward = 0.1
pass_reward = 3
death_reward = -3


## best model 5:

episodes = 20000
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
bufffer_capacity = 100000
gamma = 0.99
batch_size = 64
learning_rate = 1e-4
epsilon_start = 1.0
epsilon_min = 0.01
epsilon_decay = 3e5
target_update = 1000
print_step = 10
live_reward = 0.01
pass_reward = 3
death_reward = -3