"""
Deep Deterministic Policy Gradient agent
Author: Sameera Lanka
Website: https://sameera-lanka.com
"""

# Torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim


# Lib
import numpy as np
import random
from copy import deepcopy
from dm_control import suite
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
#from IPython.display import clear_output
from IPython import display
import os

# Files
from noise import OrnsteinUhlenbeckActionNoise as OUNoise
from replaybuffer import Buffer
from actorcritic import Actor, Critic

# Hyperparameters
ACTOR_LR = 0.0001
CRITIC_LR = 0.001
MINIBATCH_SIZE = 64
NUM_EPISODES = 10000
MU = 0
SIGMA = 0.2
CHECKPOINT_DIR = './checkpoints/manipulator/'
BUFFER_SIZE = 1000000
DISCOUNT = 0.9
TAU = 0.001
WARMUP = 70
EPSILON = 1.0
EPSILON_DECAY = 1e-6


def obs2state(observation):
    """Converts observation dictionary to state tensor"""
    l1 = [val.tolist() for val in list(observation.values())]
    l2 = []
    for sublist in l1:
        try:
            l2.extend(sublist)
        except:
            l2.append(sublist)
    return torch.FloatTensor(l2).view(1, -1)



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
        self.noise = OUNoise(mu=np.zeros(self.actionDim), sigma=SIGMA)
        self.replayBuffer = Buffer(BUFFER_SIZE)
        self.batchSize = MINIBATCH_SIZE
        self.checkpoint_dir = CHECKPOINT_DIR
        self.discount = DISCOUNT
        self.warmup = WARMUP
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.rewardgraph = []
        self.start = 0
        self.end = NUM_EPISODES
        
        
    def getQTarget(self, nextStateBatch, rewardBatch, terminalBatch):       
        """Inputs: Batch of next states, rewards and terminal flags of size self.batchSize
            Calculates the target Q-value from reward and bootstraped Q-value of next state
            using the target actor and target critic
           Outputs: Batch of Q-value targets"""
        
        targetBatch = torch.FloatTensor(rewardBatch).cuda() 
        nonFinalMask = torch.ByteTensor(tuple(map(lambda s: s != True, terminalBatch)))
        nextStateBatch = torch.cat(nextStateBatch)
        nextActionBatch = self.targetActor(nextStateBatch)
        nextActionBatch.volatile = True
        qNext = self.targetCritic(nextStateBatch, nextActionBatch)  
        
        nonFinalMask = self.discount * nonFinalMask.type(torch.cuda.FloatTensor)
        targetBatch += nonFinalMask * qNext.squeeze().data
        
        return Variable(targetBatch, volatile=False)

    
    def updateTargets(self, target, original):
        """Weighted average update of the target network and original network
            Inputs: target actor(critic) and original actor(critic)"""
        
        for targetParam, orgParam in zip(target.parameters(), original.parameters()):
            targetParam.data.copy_((1 - TAU)*targetParam.data + \
                                          TAU*orgParam.data)

            
  
    def getMaxAction(self, curState):
        """Inputs: Current state of the episode
            Returns the action which maximizes the Q-value of the current state-action pair"""
       
        noise = self.epsilon * Variable(torch.FloatTensor(self.noise()), volatile=True).cuda()
        action = self.actor(curState)
        actionNoise = action + noise
        return actionNoise
        
        
    def train(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        print('Training started...')
        
        for i in range(self.start, self.end):
            time_step = self.env.reset()  
            ep_reward = 0
            
            while not time_step.last():
                
                #Visualize Training
                display.clear_output(wait=True)
                plt.imshow(self.env.physics.render())
                plt.show()
             
                # Get maximizing action
                curState = Variable(obs2state(time_step.observation), volatile=True).cuda()
                self.actor.eval()     
                action = self.getMaxAction(curState)
                curState.volatile = False
                action.volatile = False
                self.actor.train()
                
                # Step episode
                time_step = self.env.step(action.data)
                nextState = Variable(obs2state(time_step.observation), volatile=True).cuda()
                reward = time_step.reward
                ep_reward += reward
                terminal = time_step.last()
                
                # Update replay bufer
                self.replayBuffer.append((curState, action, nextState, reward, terminal))
                
                # Training loop
                if len(self.replayBuffer) >= self.warmup:
                    
                    curStateBatch, actionBatch, nextStateBatch, \
                    rewardBatch, terminalBatch = self.replayBuffer.sample_batch(self.batchSize)
                    curStateBatch = torch.cat(curStateBatch)
                    actionBatch = torch.cat(actionBatch)
                    
                    qPredBatch = self.critic(curStateBatch, actionBatch)
                    qTargetBatch = self.getQTarget(nextStateBatch, rewardBatch, terminalBatch)
                    
                # Critic update
                    self.criticOptim.zero_grad()
                    criticLoss = self.criticLoss(qPredBatch, qTargetBatch)
                    print('Critic Loss: {}'.format(criticLoss))
                    criticLoss.backward()
                    self.criticOptim.step()
            
                # Actor update
                    self.actorOptim.zero_grad()
                    actorLoss = -torch.mean(self.critic(curStateBatch, self.actor(curStateBatch)))
                    print('Actor Loss: {}'. format(actorLoss))
                    actorLoss.backward()
                    self.actorOptim.step()
                    
                # Update Targets                        
                    self.updateTargets(self.targetActor, self.actor)
                    self.updateTargets(self.targetCritic, self.critic)
                    self.epsilon -= self.epsilon_decay
                    
            if i % 20 == 0:
                self.save_checkpoint(i)
            self.rewardgraph.append(ep_reward)


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
            'replayBuffer': self.replayBuffer,
            'rewardgraph': self.rewardgraph,
            'epsilon': self.epsilon
            
        } 
        torch.save(checkpoint, checkpointName)
    
    def loadCheckpoint(self, checkpointName):
        if os.path.isfile(checkpointName):
            print("Loading checkpoint...")
            checkpoint = torch.load(checkpointName)
            self.start = checkpoint['episode'] + 1
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.targetActor.load_state_dict(checkpoint['targetActor'])
            self.targetCritic.load_state_dict(checkpoint['targetCritic'])
            self.actorOptim.load_state_dict(checkpoint['actorOpt'])
            self.criticOptim.load_state_dict(checkpoint['criticOpt'])
            self.replayBuffer = checkpoint['replayBuffer']
            self.rewardgraph = checkpoint['rewardgraph']
            self.epsilon = checkpoint['epsilon']
            print('Checkpoint loaded')
        else:
            raise OSError('Checkpoint not found')

