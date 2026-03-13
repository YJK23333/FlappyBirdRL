import torch
import gymnasium as gym
import utils.replay_buffer as replay_buffer
import agent.dqn_agent as dqn_agent
import flappy_bird_gymnasium
from env.flappy_env import FlappyBirdEnv


env = FlappyBirdEnv()

state_dim = 5
action_dim = 2

agent = dqn_agent.DQNAgent(state_dim,action_dim)

buffer = replay_buffer.ReplayBuffer()

num_episodes = 2000

for episode in range(num_episodes):

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

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    if episode % 20 == 0:
        agent.update_target()

    print(
        f"Episode {episode}  Reward {episode_reward:.2f}  epsilon {agent.epsilon:.3f}"
    )