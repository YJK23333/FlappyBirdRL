import torch
import gymnasium as gym
import utils.replay_buffer as replay_buffer
import agent.dqn_agent as dqn_agent
import flappy_bird_gymnasium
from env.flappy_env import FlappyBirdEnv
from utils.plot import plot_rewards

env = gym.make("FlappyBird-v0", render_mode="rgb_array")
state, _ = env.reset()

# Parameters settings
episodes = 10000
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
bufffer_capacity = 100000
gamma = 0.99
batch_size = 64
learning_rate = 1e-4
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 1e5
target_update = 500
print_step = 10

#env = FlappyBirdEnv()
agent = dqn_agent.DQNAgent(state_dim,action_dim,
                           epsilon=epsilon,epsilon_min=epsilon_min,
                           epsilon_decay=epsilon_decay,
                           learning_rate=learning_rate,
                           target_update=target_update)
buffer = replay_buffer.ReplayBuffer(bufffer_capacity)


best_reward = float('-inf')
rewards = []
# Main training loop
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state,reward,terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(state,action,reward,next_state,done)

        agent.train(buffer)

        state = next_state
        episode_reward += reward

    rewards.append(episode_reward)
    # save best model
    if best_reward < episode_reward:
        best_reward = episode_reward
        torch.save(agent.q_net.state_dict(), "./results/best_model.pth")

    # print training info
    if episode % print_step == 0:
        print(f"Episode {episode}  Reward {episode_reward:.2f}  epsilon {agent.epsilon:.3f}")

plot_rewards(rewards)