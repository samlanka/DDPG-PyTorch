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
import torch.nn.functional as F
import math


# Lib
import gym
import numpy as np
import random
from copy import deepcopy
#from dm_control import suite
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')
#from IPython.display import clear_output
#from IPython import display
import os

# Files
from noise import OrnsteinUhlenbeckActionNoise as OUNoise
from replaybuffer import Buffer
from actorcritic import Actor, Critic

# Hyperparameters
ACTOR_LR = 0.001
CRITIC_LR = 0.002
MINIBATCH_SIZE = 64
NUM_EPISODES = 500
MU = 0
SIGMA = 1
CHECKPOINT_DIR = 'Github_raw_500/DDPG-PyTorch'
BUFFER_SIZE = 1000000
DISCOUNT = 0.999
TAU = 0.001
WARMUP = 70 # >= MINIBATCH_SIZE
EPSILON = 1.0
EPSILON_DECAY = 1e-7
LOGSTEP = 10


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
        #self.stateDim = obs2state(env.reset().observation).size()[1]
        #self.actionDim = env.action_spec().shape[0]
        self.stateDim = env.observation_space.shape[0]
        self.actionDim = env.action_space.shape[0]
        self.actor = Actor(self.env)
        self.critic = Critic(self.env)
        self.targetActor = deepcopy(Actor(self.env))
        self.targetCritic = deepcopy(Critic(self.env))
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
        self.stepgraph = []
        self.start = 0
        self.end = NUM_EPISODES
        
        
    def getQTarget(self, nextStateBatch, rewardBatch, terminalBatch):       
        """Inputs: Batch of next states, rewards and terminal flags of size self.batchSize
            Calculates the target Q-value from reward and bootstraped Q-value of next state
            using the target actor and target critic
           Outputs: Batch of Q-value targets"""
        
        targetBatch = torch.FloatTensor(rewardBatch)
        nonFinalMask = torch.ByteTensor(tuple(map(lambda s: s != True, terminalBatch)))
        nextStateBatch = torch.cat(nextStateBatch)
        nextActionBatch = self.targetActor(nextStateBatch)
        nextActionBatch.volatile = False
        qNext = self.targetCritic(nextStateBatch, nextActionBatch)  
        
        nonFinalMask = self.discount * nonFinalMask.type(torch.FloatTensor)
        targetBatch += nonFinalMask * qNext.squeeze().data
        
        return Variable(targetBatch, volatile = False)

    
    def updateTargets(self, target, original):
        """Weighted average update of the target network and original network
            Inputs: target actor(critic) and original actor(critic)"""
        
        for targetParam, orgParam in zip(target.parameters(), original.parameters()):
            targetParam.data.copy_((1 - TAU)*targetParam.data + \
                                          TAU*orgParam.data)

            
  
    def getMaxAction(self, curState):
        """Inputs: Current state of the episode
            Returns the action which maximizes the Q-value of the current state-action pair"""
        
        #spec = self.env.action_spec()
        #minAct = Variable(torch.FloatTensor(spec.minimum), requires_grad=False)
        #maxAct = Variable(torch.FloatTensor(spec.maximum), requires_grad=False)  
        noise = self.epsilon * Variable(torch.FloatTensor(self.noise()), volatile=False)
        action = self.actor(curState)
        actionNoise = action + noise
        return actionNoise

    def play(self, showdata=False):
        print("Playing started...")
        for i in range(1):
            time_step = self.env.reset()
            step = 0
            begins = True
            while True:
                
                self.env.render()
 
                # Get maximizing action
                if begins:
                    curState = Variable(torch.FloatTensor(time_step).view(1, -1), volatile = False)
                    begins = False
                else :
                    curState = Variable(torch.FloatTensor(time_step[0]).view(1, -1), volatile = False)
                self.actor.eval()     
                action = self.getMaxAction(curState)
                #curState.volatile = False
                action.volatile = False

                if showdata:
                    Qsa = self.critic(curState, action)
                    print("action:", action, " on state:", curState)
                    print("      with Q(s,a)=", Qsa)
                    
                
                # Step episode
                time_step = self.env.step(action.data)
                nextState = Variable(torch.FloatTensor(time_step[0]).view(1, -1))
                #reward = time_step[1]
                reward = myreward(time_step)
                if showdata:
                    print("    and gets reward: ", reward)
                
                terminal = time_step[2]
                step += 1
                
                if terminal :
                    print("Succeed")
                    break
    
    def train(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        print('Training started...')
        
        for i in range(self.start, self.end):
            time_step = self.env.reset()  
            ep_reward = 0
            step = 0
            begins = True
            
            while True:
                
                #Visualize Training
                #display.clear_output(wait=True)
                if (i % LOGSTEP == 0) :
                    self.env.render()
                #plt.show()
             
                # Get maximizing action
                if begins:
                    curState = Variable(torch.FloatTensor(time_step).view(1, -1), volatile = False)
                    begins = False
                else :
                    curState = Variable(torch.FloatTensor(time_step[0]).view(1, -1), volatile = False)
                self.actor.eval()     
                action = self.getMaxAction(curState)
                #curState.volatile = False
                action.volatile = False
                self.actor.train()

                if i % LOGSTEP == 0:
                    Qsa = self.critic(curState, action)
                    print("action:", action, " on state ", curState)
                    print("      with Q(s,a)=", Qsa)
                    
                
                # Step episode
                time_step = self.env.step(action.data)
                nextState = Variable(torch.FloatTensor(time_step[0]).view(1, -1))
                #reward = time_step[1]
                reward = myreward(time_step)
                
                ep_reward += reward
                terminal = time_step[2]
                step += 1
                # Update replay bufer
                self.replayBuffer.append((curState, action, nextState, reward, terminal))
                
                # Training loop
                if len(self.replayBuffer) >= self.warmup:
                    
                    curStateBatch, actionBatch, nextStateBatch, \
                    rewardBatch, terminalBatch = self.replayBuffer.sample_batch(self.batchSize)
                    curStateBatch = torch.cat(curStateBatch)
                    actionBatch = torch.cat(actionBatch)

                    qPredBatch = self.critic(curStateBatch, actionBatch).reshape(-1)
                    qTargetBatch = self.getQTarget(nextStateBatch, rewardBatch, terminalBatch)

                # Critic update
                    self.criticOptim.zero_grad()
                    criticLoss = self.criticLoss(qPredBatch, qTargetBatch)
                    #criticLoss = F.smooth_l1_loss(qPredBatch, qTargetBatch)
                    #if step % 5 == 4 :
                    #    print('Critic Loss: {}'.format(criticLoss))
                    criticLoss.backward(retain_graph=True)
                    self.criticOptim.step()
            
                # Actor update
                    self.actorOptim.zero_grad()
                    actorLoss = -torch.mean(self.critic(curStateBatch, self.actor(curStateBatch)))
                    #if step % 5 == 4 :
                    #    print('Actor Loss: {}'. format(actorLoss))
                    actorLoss.backward(retain_graph=True)
                    self.actorOptim.step()
                    
                # Update Targets                        
                    self.updateTargets(self.targetActor, self.actor)
                    self.updateTargets(self.targetCritic, self.critic)
                    self.epsilon -= self.epsilon_decay

                if time_step[2] :
                    break
            print(i, ':', step)

            if i % 20 == 0:
                self.save_checkpoint(i)
            self.stepgraph.append(step)
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
            'epsilon': self.epsilon,
            'stepgraph': self.stepgraph
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
            self.stepgraph = checkpoint['stepgraph']
            print('Checkpoint loaded')
        else:
            raise OSError('Checkpoint not found')

def myreward(time_step):
    return time_step[1]
    next_state, reward, done, _ = time_step
    position = next_state[0]
    velocity = next_state[1]
    r = (3 * position + 2) * 0 + 9.8 * (math.sin(3 * position) + 1) + 0.5 * velocity**2
    if done:
        r += 100
    return torch.tensor([r])
