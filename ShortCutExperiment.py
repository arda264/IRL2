# Write your experiments in here! You can use the plotting helper 
# functions from the previous assignment if you want.
import ShortCutEnvironment as SCE
import ShortCutAgents as SCA
import matplotlib as plt

#For Q and SAARS we do it in env below.
env = SCE.ShortcutEnvironment()

#Check Q Agent

agent = SCA.QLearningAgent(env.action_size(), env.state_size())
agent.train(10000)

#print(agent.Q, "done")
#env.render()
env.render_greedy(agent.Q)

#Check SARSA

agent = SCA.SARSAAgent(env.action_size(), env.state_size())
agent.train(1000)

#print(agent.Q, "done")
#env.render()
env.render_greedy(agent.Q)

#Check ExpectedSARSA

agent = SCA.ExpectedSARSAAgent(env.action_size(), env.state_size())
agent.train(1000)

#print(agent.Q, "done")
#env.render()
env.render_greedy(agent.Q)

