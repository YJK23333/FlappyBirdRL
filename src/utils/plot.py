import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(rewards):
    plt.plot(rewards, alpha=0.3, label="Raw Reward")
    plt.plot(moving_average(rewards), label="Smoothed Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Curve")
    plt.legend()
    plt.savefig("./pics/training_curve.png")
    plt.show()

def moving_average(data, window_size=20):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

