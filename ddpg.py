
# coding: utf-8

# In[ ]:


#Torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os

#Lib
import numpy as np
import random
from copy import deepcopy
from dm_control import suite
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
#from IPython.display import clear_output
from IPython import display

#Files
from noise import OrnsteinUhlenbeckActionNoise as OUNoise
from replaybuffer import Buffer
from actorcritic import Actor, Critic


# In[ ]:


ACTOR_LR = 0.0001
CRITIC_LR = 0.001
MINIBATCH_SIZE = 64
NUM_EPISODES = 10000
MU = 0
SIGMA = 0.2
CHECKPOINT_DIR = './checkpoints/'
BUFFER_SIZE = 1000000
DISCOUNT = 0.9
TAU = 0.001
WARMUP = 70

# In[ ]:
def obs2state(observation):
    l = [val.tolist() for val in list(observation.values())]
    l = [item for sublist in l for item in sublist]
    return torch.FloatTensor(l).view(1, -1)


class DDPG:
    def __init__(self, env):
        self.env = env
        self.stateDim = obs2state(env.reset().observation).size()[1]
        self.actionDim = env.action_spec().shape[0]
        self.actor = Actor(self.env).cuda()
        self.critic = Critic(self.env).cuda()
        self.targetActor = deepcopy(Actor(self.env)).cuda()
        self.targetCritic = deepcopy(Critic(self.env)).cuda()
        self.actorOptim = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.criticOptim = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.criticLoss = nn.MSELoss()
        self.noise = OUNoise(mu=np.zeros(self.actionDim), sigma=SIGMA*np.ones(self.actionDim))
        self.replayBuffer = Buffer(BUFFER_SIZE)
        self.batchSize = MINIBATCH_SIZE
        self.checkpoint_dir = CHECKPOINT_DIR
        self.discount = DISCOUNT
        self.warmup = WARMUP
        
    def getQTarget(self, nextStateBatch, rewardBatch, terminalBatch):
        targetBatch = [None] * self.batchSize
        print(type(targetBatch))
        #print(nextStateBatch)
        #print(rewardBatch)
        #print(terminalBatch)
        for i in range(self.batchSize):
            if terminalBatch[i]:
                target = torch.FloatTensor([rewardBatch[i]]).cuda()
            else:
                self.targetActor.eval()
                self.targetCritic.eval()
                actionNext = self.targetActor(nextStateBatch[i])
                qNext = self.targetCritic(nextStateBatch[i], actionNext)
                target = rewardBatch[i] + self.discount * qNext.data
                
            targetBatch[i] = target
        return Variable(torch.cat(targetBatch), requires_grad=False)
    
    def updateTargets(self, target, original):
        for targetParam, orgParam in zip(target.parameters(), original.parameters()):
            targetParam.data.copy_((1 - TAU)*targetParam.data + \
                                          TAU*orgParam.data)
    
    def train(self, start=0, end=NUM_EPISODES):
        tracklen = []
        for i in range(start, end):
            time_step = self.env.reset()  
            episodeLen = 0
            while not time_step.last():
                episodeLen += 1
                #print(episodeLen)
                display.clear_output(wait=True)
                plt.imshow(self.env.physics.render())
                plt.show()
                
                #fig, (ax1, ax2) = plt.subplots(2,1, figsize=(5,10))
                #ax1.imshow(self.env.physics.render())
                #ax2.set_title('Training Progress')
                #ax2.set_xlabel('No. of episodes')
                #ax2.set_ylabel('Episode duration')
                #ax2.plot(tracklen)
                
                #Update replay buffer
                curState = Variable(obs2state(time_step.observation)).cuda()
                self.actor.eval()
                action = self.actor(curState) + Variable(torch.FloatTensor(self.noise()), requires_grad=False).cuda()
                time_step = self.env.step(action.data)
                nextState = Variable(obs2state(time_step.observation)).cuda()
                reward = time_step.reward
                terminal = time_step.last()
                self.replayBuffer.append((curState, action, nextState, reward, terminal))
        
                self.actor.train()
                
                if len(self.replayBuffer) >= self.warmup:
                    curStateBatch, actionBatch, nextStateBatch, \
                    rewardBatch, terminalBatch = self.replayBuffer.sample_batch(self.batchSize)
                    curStateBatch = torch.cat(curStateBatch)
                    actionBatch = torch.cat(actionBatch)
                    
                    qTargetBatch = self.getQTarget(nextStateBatch, rewardBatch, terminalBatch)
                    qPredBatch = self.critic(curStateBatch, actionBatch)
                                
            #Critic update
                    self.criticOptim.zero_grad()
                    criticLoss = self.criticLoss(qPredBatch, qTargetBatch)
                    criticLoss.backward(retain_graph=True)
                    self.criticOptim.step()
            
            #Actor update
                    self.actorOptim.zero_grad()
                    actorLoss = -torch.mean(self.critic(curStateBatch, self.actor(curStateBatch)))
                    actorLoss.backward()
                    self.actorOptim.step()
            
            #Target Update
                    self.updateTargets(self.targetActor, self.actor)
                    self.updateTargets(self.targetCritic, self.critic)
            
            tracklen.append(episodeLen)
            #self.plotProgress(tracklen)

    def plotProgress(self, tracklen):
        plt.figure(2)
        plt.clf()
        plt.title('Training Progress')
        plt.xlabel('No. of episodes')
        plt.ylabel('Episode duration')
        plt.plot(tracklen)
       
        display.display(plt.gcf())
        

    def save_checkpoint(self, episode_num):
        checkpointName = self.checkpoint_dir + 'ep{}.pth.tar'.format(episode_num)
        checkpoint = {
            'episode': episode_num,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'targetActor': self.targetActor.state_dict(),
            'targetCritic': self.targetCritic.state_dict(),
            'actorOpt': self.actorOptim.state_dict(),
            'criticOpt': self.criticOptim.state_dict(),
            'replayBuffer': self.replayBuffer
        } 
        torch.save(checkpoint, checkpointName)
    
    def resume_train(self, checkpointName):
        if os.path.isfile(checkpointName):
            print("...Loading checkpoint")
            checkpoint = torch.load(checkpointName)
            start = checkpoint['episode']
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.targetActor.load_state_dict(checkpoint['targetActor'])
            self.targetCritic.load_state_dict(checkpoint['targetCritic'])
            self.actorOptim.load_state_dict(checkpoint['actorOptim'])
            self.criticOptim.load_state_dict(checkpoint['criticOptim'])
            self.replayBuffer = checkpoint['replayBuffer']
            
            self.train(start)
        else:
            raise OSError('Checkpoint not found')

