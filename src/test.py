import torch
import cv2
import numpy as np
import gymnasium as gym
import flappy_bird_gymnasium

from agent.dqn_agent import DQNAgent
from env.flappy_env import FlappyBirdEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("FlappyBird-v0", render_mode="rgb_array")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim)
agent.q_net.load_state_dict(torch.load("./results/best_model2.pth", map_location=device))
agent.q_net.eval()

state, _ = env.reset()
frame = env.render()
height, width, _ = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

video = cv2.VideoWriter(
    "./mp4/flappybird_agent.mp4",
    fourcc,
    30,
    (width, height)
)

done = False

while not done:
    action = agent.select_action(state)
    next_state, action, terminated, truncted, _ = env.step(action)
    done = terminated or truncted
    frame = env.render()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video.write(frame)
    state = next_state

video.release()

env.close()

print("Done!")
