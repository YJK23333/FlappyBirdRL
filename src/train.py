import torch
import gymnasium as gym
import utils.replay_buffer as replay_buffer
import agent.dqn_agent as dqn_agent
import flappy_bird_gymnasium
from env.flappy_env import FlappyBirdEnv

# Parameters settings
state_dim = 4
action_dim = 2
bufffer_capacity = 100000
gamma = 0.99
batch_size = 64
episodes = 10000
learning_rate = 1e-4
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.9995
target_update = 500



env = FlappyBirdEnv()
agent = dqn_agent.DQNAgent(state_dim,action_dim)
buffer = replay_buffer.ReplayBuffer(bufffer_capacity)

# Main training loop
for episode in range(episodes):
    state = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state,reward,done = env.step(action)
        buffer.push(state,action,reward,next_state,done)

        agent.train(buffer)

        state = next_state
        episode_reward += reward

    # Decay epsilon
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    if episode % target_update == 0:
        agent.update_target()
    if episode % 50 == 0:
        print(f"Episode {episode}  Reward {episode_reward:.2f}  epsilon {agent.epsilon:.3f}")