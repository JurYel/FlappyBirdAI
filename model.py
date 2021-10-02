import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as fn
import os
import config

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_QNet, self).__init__()

        # discoutn rate
        self.gamma = 0.99

        # epsilon value ϵ-greedy exploration
        self.init_epsilon = 0.1
        self.final_epsilon = 0.0001
        self.replay_memory_size = 10000
        self.num_iterations = 200000
        self.minibatch_size = 32
        self.episode_durations = []

        # construct model
        self.linear1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        self.linear3 = nn.Linear(hidden_size, output_size)
        # self.tanh = nn.Tanh()
        # self.sigmoi = nn.Sigmoid()


    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        # x = self.sigmoid(x)
        # x = fn.softmax(x, dim=1)
        return x

    def save(self, file_name = 'model.pth'):
        model_path = os.path.join(config.base_dir, "model")

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        file_name = os.path.join(model_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, axis=0)
            next_state = torch.unsqueeze(next_state, axis=0)
            action = torch.unsqueeze(action, axis=0)
            reward = torch.unsqueeze(reward, axis=0)
            done = (done, )

        # 1. predicted Q values with current state
        pred = self.model(state)

        # 2. Q_new = r + y * max(next predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action).item()] = Q_new
            # if action == 1 or action > 0.5:
            #     target[idx][0] = Q_new
            # act = torch.argmax(action).item()
            # if action[act] == True:
            #     target[idx][0] = Q_new
            # target[idx][0] = Q_new

        # PyTorch accumulates gradient by default
        # so they need to be reset in each pass
        self.optimizer.zero_grad()

        # calculate loss
        loss = self.criterion(target, pred)

        # do backward pass for updating gradients
        loss.backward()

        self.optimizer.step()
