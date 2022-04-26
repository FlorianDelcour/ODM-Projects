"""
    University of Liege
    INFO8003-1 - Optimal decision making for complex problems
    Assignment 2 - Reinforcement Learning in a Continuous Domain
    Authors : 
        DELCOUR Florian
        MAKEDONSKY Aliocha
"""

from section1 import *
import matplotlib.pyplot as plt
import random
import numpy as np
import tqdm
from sklearn.neural_network import MLPRegressor
import imageio
from car_on_the_hill_images import save_caronthehill_image
import math
import torch
import torch.nn as nn
import torch.optim as optim
    
class PQL_online():
    def __init__(self,alpha,K=20,N=100):
        self.alpha = alpha
        self.target = None
        self.eval = None
        self.buffer = []
        self.K = K
        self.N = N
        self.model = nn.Sequential(nn.Linear(3, 20), nn.Tanh(),
                            nn.Linear(20, 20), nn.Tanh(),
                            nn.Linear(20, 20), nn.Tanh(),
                            nn.Linear(20, 10), nn.Tanh(),
                            nn.Linear(10, 1), nn.Sigmoid())

    def random_trans(self):
        """
        This function take a random transition from the replay buffer

        Returns
        --------
        self.buffer[rand] : a tuple (p,s),r,u
        """
        if len(self.buffer) == 1:
            rand = 0
        else :
            rand = np.random.randint(0,len(self.buffer)-1)
        return self.buffer[rand]
    
    def set_transitions(self):
        """
        This functions creates a trajectory of 100 transitions
        """
        p= np.random.uniform(-1, 1)
        s = 0
        steps_max = 100
        set_transition = []
        for j in range(steps_max):
            if np.abs(p) > 1 or np.abs(s) > 3: # Don't want to get stuck in terminal state
                break
            u = action_space[random.randint(0, 1)]
            next_p, next_s = dynamics(p,s,u)
            r = reward_signal(p,s,u)
            self.buffer.append((p,s,u,next_p,next_s))
            sample = ((p,s), u, r, (next_p, next_s))
            set_transition.append(sample)
            p = next_p; s = next_s
    
    def PQL(self, transitions, norm):
        """
        Parametric Q learning algorithm
        
        Parameters
        ----------
        transitions : list of tuples (p, s), r, u
        norm : True to compute the normalized parametric Q-learning
        """
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        for i in range(len(transitions)-1):
            x_t = transitions[i][0:2]
            u_t = transitions[i][1]
            r_t = torch.FloatTensor([transitions[i][2]])
            new_x_t = transitions[i][3:5]
            new_u_t = transitions[i+1][1]
            q =  self.model(torch.FloatTensor([x_t[0], x_t[1], u_t]))
            
            with torch.no_grad():
                q_max = self.model(torch.FloatTensor([new_x_t[0], new_x_t[1], new_u_t]))
                target = torch.where(r_t!=0, r_t, discount_factor*q_max)
                delta = q - target
            loss = (delta * q).mean()
            optimizer.zero_grad()
            loss.backward()
            
            if norm:
              norm = torch.norm(torch.stack([torch.norm(p.grad, 2.) for p in self.model.parameters()]), 2.)
              for p in self.model.parameters():
                  p.grad /= norm + 1e-6
            
            optimizer.step()
        
    def train(self, nb_trans):
        """
        This function trains our neural network
        Parameters
        -----------
        nb_trans : number of transitions to make
        """
        counter = 0
        for i in range(self.N):
            if(counter>nb_trans):
                return
            self.set_transitions()
            set_trans = []
            for j in range(self.K):
                sample = self.random_trans()
                set_trans.append(sample)
            counter+=len(set_trans)
            self.PQL(set_trans, True)
                
def monte_carlo_simulations(nb_init_states, N, model, algorithm):
    """
        This function returns the average over nb_init_states simulations of the cumulative expected return of the
        always accelerate policy over N steps.

        Parameters
        ----------
        nb_init_states : int
            Number of simulations to do
        N : int
            Number of steps

    """
    total_J_N = np.zeros(N)
    for t in range(nb_init_states):
        total_J_N = total_J_N + expected_return(N, model, algorithm)
    total_J_N = total_J_N / nb_init_states
    return total_J_N[-1]

def expected_return(N, model, algorithm):
    """
        This function returns the cumulative expected return of a random policy over N steps.

        Parameters
        ----------
        N : int
            Number of steps

        Returns
        -------
        total_J_N : float
            The cumulative expected return

    """
    p = random.uniform(-0.1, 0.1)
    s = 0
    J_N = 0
    total_J_N = np.zeros(N)
    total_J_N[0] = J_N
    for t in range(N-1):
        if algorithm == "PQL":
            v1 = model(torch.FloatTensor([p, s, action_space[0]]))
            v2 = model(torch.FloatTensor([p, s, action_space[1]]))
        else: 
            v1 = model.predict(np.array([[p, s, action_space[0]]]))[0]
            v2 = model.predict(np.array([[p, s, action_space[1]]]))[0]
        if v1 > v2:
            u = action_space[0]
        else:
            u = action_space[1]
        reward = reward_signal(p, s, u)
        next_p, next_s = dynamics(p, s, u)
        if next_p == p and next_s == s:  # terminal state
            total_J_N[t+1] = total_J_N[t]
            continue
        J_N += (discount_factor ** t) * reward
        p = next_p
        s = next_s
        total_J_N[t + 1] = J_N
    return total_J_N

def generate_set(nb_transitions):
    """
        This function generates several episodes of one-step transitions
        It implements the second strategy based on the whole initial space

        Parameters
        ----------
        nb_episodes : int
            The number of transitions to generate

        Returns
        ----------
            A list of four-tuples
    """
    set_transition = []
    while len(set_transition) <= nb_transitions:
        p = np.random.uniform(-1, 1)
        s = 0
        max_length = 100
        for j in range(max_length):
            if np.abs(p) > 1 or np.abs(s) > 3: # Don't want to get stuck in terminal state
                break
            u = random_policy()
            next_p, next_s = dynamics(p,s,u)
            r = reward_signal(p,s,u)
            sample = ((p,s), u, r, (next_p, next_s))
            set_transition.append(sample)
            p = next_p; s = next_s
            
    return set_transition
      
def FQI(set_trajectory, model):
    """
        This function executes the fitted Q iteration algorithm

        Parameters
        ----------
        set_trajectory : list of four-tuples 
            Episodes of one-step transitions (x_t, u_t, r_t, x_t+1)
        model : sklearn model
            Supervised learning algorithm used to learn the Q_N function
        stop_rule : function
            returns True if we must stop the fitted Q iteration algorithm
        bound_error : float
            treshold used by the stop_rule function

        Returns
        ----------
            The estimated Q_N function 
    """
    N = 0; Q = 0; Q_prev = 0
    N_max = math.ceil(math.log((0.01 / (2 * 1)) * (1. - discount_factor)**2)/math.log(discount_factor))
    for _ in range(N_max):
        N += 1; i = []; o = []
        nb_samples = len(set_trajectory)
        for k in range(nb_samples):
            x_t = set_trajectory[k][0]
            u_t = set_trajectory[k][1]
            r_t = set_trajectory[k][2]
            new_x_t = set_trajectory[k][3]
            i.append([x_t[0], x_t[1], u_t]) # Inputs are always the same
            
            if N == 1: # First iteration
                o.append(r_t)
            else : 
                Q_max = max(Q.predict(np.array([[new_x_t[0], new_x_t[1], action_space[0]]]))[0],
                            Q.predict(np.array([[new_x_t[0], new_x_t[1], action_space[1]]]))[0])
                o.append(r_t+discount_factor*Q_max)
        
        Q_prev = Q
        Q = model(i,o)
    return Q

def neural_network(i,o):
    """
        This function creates a neural network model

        Parameters
        ----------
        i : list of lists
            Inputs given to the neural network model
        o : list
            Labels of the inputs

        Returns
        ----------
            A trained model
    """
    model = MLPRegressor(hidden_layer_sizes=(20, 20, 20, 10), activation='tanh', random_state=42, max_iter=800)
    model.fit(i,o)
    return model
      
def experimental_protocol():
    """
    Experimental protocol of the section 5
    """
    nb_transitions = [1000, 1500, 2000, 2500]
    Q_FQI = np.zeros(len(nb_transitions))
    Q_PQL = np.zeros(len(nb_transitions))
    for i in tqdm.tqdm(range(len(nb_transitions))):
        
        set_trans = generate_set(nb_transitions[i])
        model_FQI = FQI(set_trans, neural_network)
        Q_FQI[i] = monte_carlo_simulations(50, 150, model_FQI, "FQI")
        model_PQL = PQL_online(alpha=0.02, K=20, N=10000)
        model_PQL.train(nb_transitions[i])
        Q_PQL[i] = monte_carlo_simulations(50, 150, model_PQL.model, "PQL")
    
    plt.figure()
    plt.plot(nb_transitions,Q_FQI, 'r') 
    plt.plot(nb_transitions,Q_PQL, 'b')
    plt.title('FQI (red) - PQL (blue)')
    plt.xlabel('Number of one-step system transitions')
    plt.ylabel('Expected return')
    plt.savefig("FQI_PQL")
    plt.close()

if __name__ == "__main__":

    experimental_protocol()