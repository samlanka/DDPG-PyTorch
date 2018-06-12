from ddpg import DDPG
import gym

env = gym.make('MountainCarContinuous-v0')
agent = DDPG(env)
agent.train()
env.close()
