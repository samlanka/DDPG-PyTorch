
# coding: utf-8

# In[ ]:


import torch 
import torch.nn as nn
import numpy as np


# In[ ]:


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    w = 1./ np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-w, w)


# In[ ]:


HID_LAYER1 = 400
HID_LAYER2 = 300
WFINAL = 0.003


# In[ ]:
def obs2state(observation):
    l = [val.tolist() for val in list(observation.values())]
    l = [item for sublist in l for item in sublist]
    return torch.FloatTensor(l).view(1, -1)

class Actor(nn.Module):
    def __init__(self, env):
        super(Actor, self).__init__()
        self.stateDim = obs2state(env.reset().observation).size()[1]
        self.actionDim = env.action_spec().shape[0]
        
        self.norm0 = nn.BatchNorm1d(self.stateDim)
                                    
        self.fc1 = nn.Linear(self.stateDim, HID_LAYER1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
                                    
        self.bn1 = nn.BatchNorm1d(HID_LAYER1)
                                    
        self.fc2 = nn.Linear(HID_LAYER1, HID_LAYER2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
                                    
        self.bn2 = nn.BatchNorm1d(HID_LAYER2)
                                    
        self.fc3 = nn.Linear(HID_LAYER2, self.actionDim)
        self.fc3.weight.data.uniform_(-WFINAL, WFINAL)
        
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()
            
    def forward(self, ip):
        ip_norm = self.norm0(ip)                            
        h1 = self.ReLU(self.fc1(ip_norm))
        h1_norm = self.bn1(h1)
        h2 = self.ReLU(self.fc2(h1_norm))
        h2_norm = self.bn2(h2)
        action = (self.fc3(h2_norm))
        return action
        


# In[ ]:


class Critic(nn.Module):
    def __init__(self, env):
        super(Critic, self).__init__()
        self.stateDim = obs2state(env.reset().observation).size()[1]
        self.actionDim = env.action_spec().shape[0]
        
        self.fc1 = nn.Linear(self.stateDim, HID_LAYER1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        
        self.bn1 = nn.BatchNorm1d(HID_LAYER1)
        self.fc2 = nn.Linear(HID_LAYER1 + self.actionDim, HID_LAYER2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        
        self.fc3 = nn.Linear(HID_LAYER2, 1)
        self.fc3.weight.data.uniform_(-WFINAL, WFINAL)
        
        self.ReLU = nn.ReLU()
        
    def forward(self, ip, action):
        h1 = self.ReLU(self.fc1(ip))
        h1_norm = self.bn1(h1)
        h2 = self.ReLU(self.fc2(torch.cat([h1_norm, action], dim=1)))
        Qval = self.ReLU(self.fc3(h2))
        return Qval
        

