import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import model.dqn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        
        self.q_net = model.dqn.DQN(state_dim, action_dim).to(device)
        self.target_net = model.dqn.DQN(state_dim, action_dim).to(device)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.update_target()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        q = self.q_net(state)

        return q.argmax().item()
    
    def train(self, buffer):
        if(len(buffer) < self.batch_size):
            return
        
        state, action, reward, next_state, done = buffer.sample(self.batch_size)
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        reward = torch.FloatTensor(reward).to(device)
        action = torch.LongTensor(action).to(device)
        done = torch.FloatTensor(done).to(device)

        q_values = self.q_net(state)
        next_q_values = self.target_net(next_state)

        q = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q = next_q_values.max(1)[0]

        target = reward + self.gamma * next_q * (1 - done)
        loss = F.mse_loss(q, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()