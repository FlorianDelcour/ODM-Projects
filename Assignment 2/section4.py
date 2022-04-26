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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
import imageio
from car_on_the_hill_images import save_caronthehill_image

   
def generate_set1(nb_episodes):
    """
        This function generates several episodes of one-step transitions
        It implements the first strategy based on the reduce initial space

        Parameters
        ----------
        nb_episodes : int
            The number of episodes to generate

        Returns
        ----------
            A list of four-tuples of maximum size = 100*60
    """
    set_transition = []
    for i in range(nb_episodes):
        p = np.random.uniform(-0.1, 0.1)
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

def generate_set2(nb_episodes):
    """
        This function generates several episodes of one-step transitions
        It implements the second strategy based on the whole initial space

        Parameters
        ----------
        nb_episodes : int
            The number of episodes to generate

        Returns
        ----------
            A list of four-tuples of maximum size = 100*60
    """
    set_transition = []
    for i in range(nb_episodes):
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

def stop_criterion1(bound_error, N, Q, Q_prev, set_transition):
    """
        This function returns True if we must stop the fitted Q iteration algorithm
        It implements the first stopping rule based on the infinite norm of 
        the difference between Q_N and Q_N-1

        Parameters
        ----------
        bound_error : float
            treshold used by the stop_rule function
        N : int
            number of the ith iteration of the fitted Q iteration algorithm
        Q : sklearn model
            The actual estimated Q_N function
        Q_prev : sklearn model
            The previous estimated Q_N function
        set_transition : list of four-tuples 
            Episodes of one-step transitions (x_t, u_t, r_t, x_t+1)

        Returns
        ----------
            True if we must stop the fitted Q iteration algorithm
            False otherwise
    """
    if N==0 or N==1:
        return False
    
    pred1 = []
    pred2 = []
    for i in range(len(set_transition)):
        x_t = set_transition[i][0]
        u_t = set_transition[i][1]
        pred1.append(Q_prev.predict(np.array([[x_t[0], x_t[1], u_t]]))[0])
        pred2.append(Q.predict(np.array([[x_t[0], x_t[1], u_t]]))[0])
    substract = [pred2-pred1 for (pred2, pred1) in zip(pred2, pred1)]
    threshold = max(abs(number) for number in substract)
    print(threshold)
    if threshold > bound_error:
        return False
    return True
    
def stop_criterion2(bound_error, N, Q, Q_prev, set_transition):
    """
        This function returns True if we must stop the fitted Q iteration algorithm
        It implements the second stopping rule based on the theoretical bound
        
        Parameters
        ----------
        bound_error : float
            treshold used by the stop_rule function
        N : int
            number of the ith iteration of the fitted Q iteration algorithm
        Q : sklearn model
            The actual estimated Q_N function
        Q_prev : sklearn model
            The previous estimated Q_N function
        set_transition : list of four-tuples 
            Episodes of one-step transitions (x_t, u_t, r_t, x_t+1)

        Returns
        ----------
            True if we must stop the fitted Q iteration algorithm
            False otherwise
    """
    Br = 1 
    upper_bound = 2 * discount_factor**N * Br / (1-discount_factor)**2
    if upper_bound < bound_error:
        return True
    else:
        return False
    
def fitted_Q_iteration(set_trajectory, model, stop_rule, bound_error):
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
    
    while stop_rule(bound_error, N, Q, Q_prev, set_trajectory) == 0: # Stop when N satisfy the stopping rule
        print(N)
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

def random_policy():
    """
        This functions returns a random action defined over the action space.
    """
    return action_space[random.randint(0, 1)]

def lin_regression(i,o):
    """
        This function creates a linear regression model

        Parameters
        ----------
        i : list of lists
            Inputs given to the linear regression model
        o : list
            Labels of the inputs

        Returns
        ----------
            A trained model
    """
    model = LinearRegression(n_jobs=-1)
    model.fit(i,o)
    return model

def extremely_random_trees(i,o):
    """
        This function creates an Extremely Randomized Trees model

        Parameters
        ----------
        i : list of lists
            Inputs given to the Extremely Randomized Trees model
        o : list
            Labels of the inputs

        Returns
        ----------
            A trained model
    """
    model = ExtraTreesRegressor(n_estimators=10, random_state=42)
    model.fit(i,o)
    return model

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

def heatmaps(Q_hat):
    """
        This function generates Q_N plots for each action and the derived
        optimal policy plot.

        Parameters
        ----------
        Q_hat : sklearn model
            The estimated Q_N function 
    """
    
    p_vec = np.arange(-1, 1, 0.01)
    s_vec = np.arange(-3, 3, 0.01)
    Q_map = np.zeros([len(p_vec),len(s_vec),len(action_space)])

    for i in range(len(p_vec)):
        for j in range(len(s_vec)):
            for k in range(len(action_space)):
                p = p_vec[i]
                s = s_vec[j]
                u = action_space[k]
                Q_map[i,j,k] = Q_hat.predict(np.array([[p,s,u]]))[0]
    
    color_map1 = Q_map[:,:,0].T
    color_map2 = Q_map[:,:,1].T
    min_v = np.min(color_map1)
    max_v = np.max(color_map1)
    fig1, ax1 = plt.subplots()
    X, Y = np.meshgrid(p_vec, s_vec)
    c = ax1.contourf(X,Y,color_map1,10,cmap='RdBu',vmax=max_v,vmin=min_v)
    fig1.colorbar(c)
    ax1.set_title('$\hat{Q}_N$ for U = 4')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Speed')
    ax1.set_xlim([-1., 1.])
    ax1.set_ylim([-3, 3])
    plt.savefig("Q_hat_estimated_right_action.png")
    plt.close()
    
    min_v = np.min(color_map2)
    max_v = np.max(color_map2)
    fig2, ax2 = plt.subplots()
    c = ax2.contourf(X,Y,color_map2,10,cmap='RdBu',vmax=max_v,vmin=min_v)
    fig2.colorbar(c)
    ax2.set_title('$\hat{Q}_N$ for U = -4')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Speed')
    ax2.set_xlim([-1., 1.])
    ax2.set_ylim([-3, 3])
    plt.savefig("Q_hat_estimated_left_action.png")
    plt.close()
    
    opt_policy = np.zeros([len(p_vec),len(s_vec)])

    for p in range(len(p_vec)):
        for s in range(len(s_vec)):
            ind = np.argmax(Q_map[p,s])
            opt_policy[p,s] = action_space[ind]
    
    fig3, ax3 = plt.subplots()
    c = ax3.contourf(X,Y,opt_policy.T,cmap='RdBu',vmax=4,vmin=-4)
    ax3.set_title('Policy (blue = 4) | (red = -4)')
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Speed')
    ax3.set_xlim([-1., 1.])
    ax3.set_ylim([-3, 3])
    plt.savefig("Estimated_policy.png")
    plt.close()

def monte_carlo_simulations(nb_init_states, N, model):
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
        total_J_N = total_J_N + expected_return(N, model)
    total_J_N = total_J_N / nb_init_states
    return total_J_N[-1]

def expected_return(N, model):
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

def gen_traj(N, model):
    """
        This function generates a trajectory with N moves

        Parameters
        ----------
        N : int
            Number of actions the car has to take
        model : sklearn model
            Estimated Q_N function
    """
    prev_p = random.uniform(-0.1, 0.1)
    prev_s = 0
    for t in range(N):
        v1 = model.predict(np.array([[prev_p, prev_s, action_space[0]]]))[0]
        v2 = model.predict(np.array([[prev_p, prev_s, action_space[1]]]))[0]
        if v1 > v2:
            u = action_space[0]
        else:
            u = action_space[1]
        p, s = dynamics(prev_p, prev_s, u)
        reward = reward_signal(prev_p, prev_s, u)
        if reward==1 or reward == -1:
            x_t = "x_" + str(t) + " = (" + str("{:.3f}".format(prev_p)) + "," + str("{:.3f}".format(prev_s)) + ")"
            u_t = "u_" + str(t) + " = " + str(u)
            r_t = "r_" + str(t) + " = " + str(reward)
            x_t_next = "x_" + str(t + 1) + " = (" + str("{:.3f}".format(p)) + "," + str("{:.3f}".format(s)) + ")"
            print("Step " + str(t) + " : (" + x_t + ", " + u_t + ", " + r_t + ", " + x_t_next + ")")
        prev_p = p
        prev_s = s
        
def images_car(N, model):
    """
        This function creates N images of the car on the hill problem, so the position of the car and its speed. Then
        it fuse those images into a GIF. The car follow a specific policy derived from Q.

        Parameters
        ----------
        N : int
            Number of images (steps of the car on the hill problem)
        model : sklearn model
            Estimated Q_N function  

    """
    p = np.random.uniform(-0.1, 0.1)
    s = 0
    images = []
    for i in range(N):
        file_name = 'GIF_car/section3_{}.jpg'.format(i)
        # We need to add the two following lines otherwise there will be a problem to display the graph
        # because if s is bigger than the max speed which is 3, then there is an error for the color
        if p > 1:
            nice_p = 1
        elif p < -1:
            nice_p = -1
        else:
            nice_p = p

        if s > 3:
            nice_s = 3
        elif s < -3:
            nice_s = -3
        else:
            nice_s = s

        save_caronthehill_image(nice_p, nice_s, file_name)
        images.append(imageio.imread(file_name))

        v1 = model.predict(np.array([[p, s, action_space[0]]]))[0]
        v2 = model.predict(np.array([[p, s, action_space[1]]]))[0]
        if v1 > v2:
            u = action_space[0]
        else:
            u = action_space[1]
        p, s = dynamics(p, s, u)

    imageio.mimsave('simulation.gif', images)

if __name__ == "__main__":
    #Q_lr = fitted_Q_iteration(generate_set2(60), lin_regression, stop_criterion2, 0.01)
    #heatmaps(Q_lr)
    #J_Q_lr = monte_carlo_simulations(50, 400, Q_lr)
    #print("Expected reward of mu* for linear regression : " + str(J_Q_lr))
    #images_car(300, Q_lr)
    
    #Q_trees = fitted_Q_iteration(generate_set2(60), extremely_random_trees, stop_criterion2, 60)
    #heatmaps(Q_trees)
    #J_Q_trees = monte_carlo_simulations(50, 400, Q_trees)
    #print("Expected reward of mu* for trees : " + str(J_Q_trees))
    #gen_traj(100, Q_trees)
    #images_car(300, Q_trees)

    #Q_nn = fitted_Q_iteration(generate_set2(60), neural_network, stop_criterion2, 0.01)
    #heatmaps(Q_nn)
    #J_Q_nn = monte_carlo_simulations(50, 400, Q_nn)
    #print("Expected reward of mu* for neural network : " + str(J_Q_nn))
    #gen_traj(100, Q_nn)
    #images_car(300, Q_nn)
    