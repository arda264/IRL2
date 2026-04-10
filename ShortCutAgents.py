import numpy as np
import ShortCutEnvironment as SCE

class QLearningAgent(object):
    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        self.Q = np.zeros((n_states, n_actions)) # -> q values
        self.n = np.zeros((n_states, n_actions)) # -> action counter
        
        
    def select_action(self, state):
        # TO DO: Implement policy
        best_action = np.argmax(self.Q[state]) #Naive agent copied

        probabilities = np.multiply(np.ones(self.n_actions), np.divide(self.epsilon, (self.n_actions -1))) #making every choice probability the same
        probabilities[best_action] = (1 - self.epsilon)#assigning the naive agents value to the probabilities, sums up to 1

        action = np.random.choice(self.n_actions, p = probabilities)#check out link above

        return action
        
    def update(self, state, next_state, action, reward, done): # Augment arguments if necessary
        # TO DO: Implement Q-learning update
        self.n[state, action] += 1
        
        future = (self.gamma * np.max(self.Q[next_state]))
        
        if done:
            future = 0
        
        self.Q[(state, action)] = self.Q[(state, action)] + self.alpha * ((reward + future) - self.Q[(state, action)])
    
    def train(self, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        env = SCE.ShortcutEnvironment()

        for _ in range(n_episodes):
            env.reset()
            state = env.y * env.c + env.x
            cumulative_reward = 0

            while not env.isdone:
                action = self.select_action(state)
                reward = env.step(action)                   
                next_state = env.y * env.c + env.x         
                done = env.isdone                            

                self.update(state, next_state, action, reward, done)

                state = next_state
                cumulative_reward += reward

            episode_returns.append(cumulative_reward)

        return episode_returns


class SARSAAgent(object):
    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary

        self.Q = np.zeros((n_states, n_actions)) #Initialize the action values Q(s,a) to 0.
        
    def select_action(self, state):
        # TO DO: Implement policy

        #Implement an ϵ-greedy policy for selecting an action.
        action = None

        best_action = np.argmax(self.Q[state, :])

        #from our previous assignment
        probabilities = np.multiply(np.ones(self.n_actions), np.divide(self.epsilon, (self.n_actions -1))) #making every choice probability the same
        probabilities[best_action] = (1 - self.epsilon) #assigning the naive agents value to the probabilities, sums up to 1

        action = np.random.choice(self.n_actions, p = probabilities)

        return action
        
    def update(self, state, action, reward, next_state, next_action, done): # Augment arguments if necessary
        # TO DO: Implement SARSA update

        #sarsa update Q(St, At) = Q(St, At) + alpha*[Rt+1 + gamma*Q(St+1, At+1) - Q(St, At)] 
        predict = self.Q[state, action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.Q[next_state, next_action]
        
        self.Q[state, action] += self.alpha * (target - predict)

    def train(self, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode

        episode_returns = []
        env = SCE.ShortcutEnvironment()

        for _ in range(n_episodes):
            env.reset()
            state = env.y * env.c + env.x
            cumulative_reward = 0
            
            action = self.select_action(state)

            while not env.isdone:
                reward = env.step(action)
                next_state = env.y * env.c + env.x
                next_action = self.select_action(next_state)
                done = env.isdone

                self.update(state, action, reward, next_state, next_action, done)

                state = next_state
                action = next_action
                
                cumulative_reward += reward

            episode_returns.append(cumulative_reward)

        return episode_returns

class ExpectedSARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary

        self.Q = np.zeros((n_states, n_actions)) #Initialize the action values Q(s,a) to 0.
        
    def select_action(self, state):
        # TO DO: Implement policy

        #Implement an ϵ-greedy policy for selecting an action.
        action = None

        best_action = np.argmax(self.Q[state, :])

        #from our previous assignment
        probabilities = np.multiply(np.ones(self.n_actions), np.divide(self.epsilon, (self.n_actions -1))) #making every choice probability the same
        probabilities[best_action] = (1 - self.epsilon) #assigning the naive agents value to the probabilities, sums up to 1

        action = np.random.choice(self.n_actions, p = probabilities)

        return action
        
    def update(self, state, action, reward, done, next_state): # Augment arguments if necessary
        # TO DO: Implement Expected SARSA update

        #expected sarsa update Q(St, At) = Q(St, At) + alpha * [Rt+1 + gamma * sum(pi(a|St+1) * Q(St+1, a)) - Q(St, At)]
        predict = self.Q[state, action]

        if not done:
            best_next_action = np.argmax(self.Q[next_state, :])
            
            expected_q = 0

            for a in range(self.n_actions):
                if a == best_next_action:
                    expected_q += (1 - self.epsilon) * self.Q[next_state, a]
                
                else:
                    expected_q += (self.epsilon / (self.n_actions - 1)) * self.Q[next_state, a]
            
            target = reward + self.gamma * expected_q  
        
        else:
            target = reward
        
        self.Q[state, action] += self.alpha * (target - predict)

    def train(self, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        
        episode_returns = []
        env = SCE.ShortcutEnvironment()
        
        for _ in range(n_episodes):
            env.reset()
            state = env.y * env.c + env.x
            cumulative_reward = 0

            while not env.isdone:
                action = self.select_action(state)
                
                reward = env.step(action)
                next_state = env.y * env.c + env.x
                done = env.isdone
                
                self.update(state, action, reward, done, next_state)
                
                state = next_state
                cumulative_reward += reward

            episode_returns.append(cumulative_reward)
            
        return episode_returns   
    
class nStepSARSAAgent(object):

    def __init__(self, n_actions, n_states, n, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.n = n
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # TO DO: Initialize variables if necessary
        
    def select_action(self, state):
        # TO DO: Implement policy
        action = None
        return action
        
    def update(self, states, actions, rewards, done): # Augment arguments if necessary
        # TO DO: Implement n-step SARSA update
        pass
    
    def train(self, n_episodes):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        return episode_returns  
    
    
    