import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
from collections import namedtuple,deque

class CNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(in_features = 64 * 5 * 5, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=4)

        self.dropout = nn.Dropout(0.5)

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn1(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.bn2(self.conv4(x)))

        x = x.view(x.size(0), -1)


        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x




class ReplayBuffer(object):
    """Replay buffer to store and sample experience tuples.
    """
    def __init__(self, capacity) -> None:
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def add(self, state, action, reward, next_state,done):

        experience = self.experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
 
    def __getitem__(self, idx):
        experience = self.buffer[idx]
        state = experience.state
        if not isinstance(state,torch.Tensor):
            state = torch.tensor(experience.state).float().to(self.device)

        if state.shape == (1,3,10,10):
            state = state.squeeze(0)
            assert(state.shape == (3,10,10))

        action = torch.tensor(experience.action).long().to(self.device)

        reward = torch.tensor(experience.reward).float().to(self.device)

        next_state = experience.next_state
        if not isinstance(next_state,torch.Tensor):
            next_state = torch.tensor(experience.next_state).float().to(self.device)
        if next_state.shape == (1,3,10,10):
            state = state.squeeze(0)
            assert(state.shape == (3,10,10))

        done = torch.tensor(experience.done).float().to(self.device)

        return (state, action, reward, next_state, done)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self,
                 model  = None,
                 model_path = None,
                 buffer_size=int(1e5),
                 batch_size=64,
                 gamma=0.99,
                 lr=0.001,
                 update_every=4,
                 do_update=False) -> None:
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        if model:
            self.q_network_local = model
            self.q_network_local = self.q_network_local.to(self.device)
            self.q_network_target = model
            self.q_network_target = self.q_network_target.to(self.device)
            if model_path:
                model_weights = torch.load(model_path,map_location=self.device)
                self.q_network_local.load_state_dict(model_weights)
                self.q_network_target.load_state_dict(model_weights)
        else: 
            raise NotImplementedError

        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=lr,weight_decay=1e-5)
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_every = update_every
        self.t_step = 0
    

        self.do_update = do_update
        self.q_network_local = self.q_network_local.to(self.device)
        self.q_network_target = self.q_network_target.to(self.device)


    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                dataloader = DataLoader(self.memory, batch_size=self.batch_size, shuffle=True)
                self.learn(dataloader, self.gamma)

    def act(self, state, eps=0.):
        self.q_network_local.eval()
        with torch.no_grad():
            action_values = self.q_network_local(state)
        self.q_network_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()), action_values.cpu().data.numpy().ravel()
        else:
            return random.choice(np.arange(4)), action_values.cpu().data.numpy()

    def learn(self, dataloader, gamma):
        for states, actions, rewards, next_states, dones in dataloader:
            Q_targets_next = self.q_network_target(next_states).detach().max(1)[0]

            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            Q_expected = self.q_network_local(states).gather(1, actions.unsqueeze(-1))


            Q_targets = Q_targets.unsqueeze(1)


            loss = F.mse_loss(Q_expected, Q_targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.soft_update(self.q_network_local, self.q_network_target, tau=1e-3)
    
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
             target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)