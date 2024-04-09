from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from collections import deque
import random

class Agent:
    class Net(nn.Module):
        def __init__(self, input_shape=(4, 84, 84), n_actions=len(COMPLEX_MOVEMENT)):
            super(Agent.Net, self).__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            self.flattened_size = self._get_conv_output(input_shape)
            self.fc_layers = nn.Sequential(
                nn.Linear(self.flattened_size, 512),
                nn.ReLU(),
                nn.Linear(512, n_actions)
            )
        
        def forward(self, x):
            x = self.conv_layers(x)
            x = self.fc_layers(x)
            return x
        
        def _get_conv_output(self, shape):
            with torch.no_grad():
                input = torch.zeros(1, *shape)
                output = self.conv_layers(input)
                return int(np.prod(output.size()))
        
    def __init__(self):
        policy_net_path='model_weights_4.pth'
        stack_frames=4
        skip_frames=4
        # model_module = importlib.import_module("109062312_hw2_data")
        # Net = model_module.Net
        self.policy_net = Agent.Net()
        self.policy_net.load_state_dict(torch.load(policy_net_path))
        self.policy_net.eval()
        self.stack_frames = stack_frames
        self.skip_frames = skip_frames
        self.state_buffer = deque([], maxlen=stack_frames)
        self.resize = transforms.Compose([transforms.ToPILImage(), transforms.Resize((84, 84)), transforms.Grayscale(), transforms.ToTensor()])
        self.last_action = 0
        self.action_counter = 0
        self.epsilon = 0.01

    def act(self, observation):
        
        observation = observation.astype('uint8')

        if self.action_counter % self.skip_frames == 0:
            self.state_buffer.append(self.resize(observation).squeeze(0))
            
            if len(self.state_buffer) == self.stack_frames:
                if np.random.rand() <= self.epsilon:
                    self.last_action = random.randrange(len(COMPLEX_MOVEMENT))
                else:
                    state = torch.stack(list(self.state_buffer), dim=0).unsqueeze(0).cpu()
                    with torch.no_grad():
                        self.last_action = self.policy_net(state).max(1)[1].view(1, 1).item()
            self.action_counter = 0

        self.action_counter += 1
        return self.last_action