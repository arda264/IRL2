# Write your experiments in here! You can use the plotting helper 
# functions from the previous assignment if you want.
import ShortCutEnvironment as SCE
import ShortCutAgents as SCA
import matplotlib as plt

env = SCE.ShortcutEnvironment()

agent = SCA.QLearningAgent(env.action_size(), env.state_size())
agent.train(1000)

print(agent.Q)
