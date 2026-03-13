import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import model.dqn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self, state_dim, action_dim, 
                 batch_size=64, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.01, 
                 epsilon_decay=0.997, learning_rate=1e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_net = model.dqn.DQN(state_dim, action_dim).to(device)
        self.target_net = model.dqn.DQN(state_dim, action_dim).to(device)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.update_target()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q = self.q_net(state)
        return q.argmax().item()
    
    def train(self, buffer):
        if(len(buffer) < self.batch_size):
            return
        
        state, action, reward, next_state, done = buffer.sample(self.batch_size)
        state = torch.tensor(state, dtype=torch.float32, device=device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
        reward = torch.tensor(reward, dtype=torch.float32, device=device)
        action = torch.tensor(action, dtype=torch.long, device=device)
        done = torch.tensor(done, dtype=torch.float32, device=device)

        q_values = self.q_net(state)
        next_q_values = self.target_net(next_state)

        q = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q = next_q_values.max(1)[0]

        target = reward + self.gamma * next_q * (1 - done)
        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
        self.optimizer.step()