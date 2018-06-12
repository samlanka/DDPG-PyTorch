from ddpg import DDPG
import gym
import matplotlib.pyplot as plt

env = gym.make('MountainCarContinuous-v0')
agent = DDPG(env)
ckdir = agent.checkpoint_dir
ckpoints = [480]

maxep = max(ckpoints)
for ckpoint in ckpoints:
    ckpointname = "Github_E_500/DDPG-PyTorch"+"ep{}.pth.tar".format(ckpoint)
    print(ckpointname)
    agent.loadCheckpoint(ckpointname)
    #agent.play(showdata=False)
    #print(agent.stepgraph)
    if ckpoint == maxep:
        n = agent.start-1
        eps = list(range(n))
        avgs = []
        for i in eps:
            if i+5<n:
                avgs.append(sum(agent.stepgraph[i:i+10])/10)
            else:
                avgs.append(agent.stepgraph[i])
        plt.title('DDPG with raw reward')
        plt.ylabel('number of steps to succeed')
        plt.xlabel('episode')
        plt.plot(eps, avgs)

plt.savefig("DDPG_E_10")
plt.show()
env.close()
