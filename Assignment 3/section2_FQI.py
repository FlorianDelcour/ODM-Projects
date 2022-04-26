"""
    University of Liege
    INFO8003-1 - Optimal decision making for complex problems
    Assignment 3 - Deep Reinforcement Learning with Images for the Car on the Hill Problem
    Authors :
        DELCOUR Florian
        MAKEDONSKY Aliocha
"""

from sklearn.ensemble import ExtraTreesRegressor
from dql import *
from domain import *


def extremely_random_trees(i, o):
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
    model = ExtraTreesRegressor()
    model.fit(i, o)
    return model


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
    N = 0
    Q = 0
    Q_prev = 0

    while stop_rule(bound_error, N, Q, Q_prev, set_trajectory) == 0:  # Stop when N satisfy the stopping rule
        print(N)
        N += 1
        i = []
        o = []
        nb_samples = len(set_trajectory)
        for k in range(nb_samples):
            x_t = set_trajectory[k][0]
            u_t = set_trajectory[k][1]
            r_t = set_trajectory[k][2]
            new_x_t = set_trajectory[k][3]
            i.append(np.append(x_t, u_t))  # Inputs are always the same

            if N == 1:  # First iteration
                o.append(r_t)
            else:
                Q_max = max(Q.predict(np.array([np.append(new_x_t, action_space[0])]))[0],
                            Q.predict(np.array([np.append(new_x_t, action_space[1])])))
                o.append(r_t + gamma * Q_max)

        Q_prev = Q
        Q = model(i, o)
    return Q


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
    upper_bound = 2 * gamma ** N * Br / (1 - gamma) ** 2
    if upper_bound < bound_error:
        return True
    else:
        return False


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
    while len(set_transition)<10000:
    #for i in range(nb_episodes):
        p = np.random.uniform(-0.1, 0.1)
        s = 0
        max_length = 150
        while np.abs(p) < 1 and np.abs(s) < 3:
        #for j in range(max_length):
            #if np.abs(p) >= 1 or np.abs(s) >= 3:  # Don't want to get stuck in terminal state
            #    break
            u = random_policy()
            next_p, next_s = dynamics(p, s, u)
            r = reward_signal(p, s, u)
            image = x_to_image((p, s)).view(30000).numpy()
            next_image = x_to_image((next_p, next_s)).view(30000).numpy()
            sample = (image, u, r, next_image)
            set_transition.append(sample)
            p = next_p
            s = next_s
    print(len(set_transition))
    return set_transition


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
    Q_map = np.zeros([len(p_vec), len(s_vec), len(action_space)])

    for i in range(len(p_vec)):
        for j in range(len(s_vec)):
            for k in range(len(action_space)):
                p = p_vec[i]
                s = s_vec[j]
                u = action_space[k]
                Q_map[i, j, k] = Q_hat.predict(np.array([np.append(x_to_image((p, s)).view(30000).numpy(), u)]))[0]

    color_map1 = Q_map[:, :, 0].T
    color_map2 = Q_map[:, :, 1].T
    min_v = np.min(color_map1)
    max_v = np.max(color_map1)
    fig1, ax1 = plt.subplots()
    X, Y = np.meshgrid(p_vec, s_vec)
    c = ax1.contourf(X, Y, color_map1, 10, cmap='RdBu', vmax=max_v, vmin=min_v)
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
    c = ax2.contourf(X, Y, color_map2, 10, cmap='RdBu', vmax=max_v, vmin=min_v)
    fig2.colorbar(c)
    ax2.set_title('$\hat{Q}_N$ for U = -4')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Speed')
    ax2.set_xlim([-1., 1.])
    ax2.set_ylim([-3, 3])
    plt.savefig("Q_hat_estimated_left_action.png")
    plt.close()

    opt_policy = np.zeros([len(p_vec), len(s_vec)])

    for p in range(len(p_vec)):
        for s in range(len(s_vec)):
            ind = np.argmax(Q_map[p, s])
            opt_policy[p, s] = action_space[ind]

    fig3, ax3 = plt.subplots()
    c = ax3.contourf(X, Y, opt_policy.T, cmap='RdBu', vmax=4, vmin=-4)
    ax3.set_title('Policy (blue = 4) | (red = -4)')
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Speed')
    ax3.set_xlim([-1., 1.])
    ax3.set_ylim([-3, 3])
    plt.savefig("Estimated_policy_FQI_image.png")
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
    for t in range(N - 1):
        v1 = model.predict(np.array([np.append(x_to_image((p,s)).view(30000).numpy(), action_space[0])]))[0]
        v2 = model.predict(np.array([np.append(x_to_image((p,s)).view(30000).numpy(), action_space[1])]))[0]
        if v1 > v2:
            u = action_space[0]
        else:
            u = action_space[1]
        reward = reward_signal(p, s, u)
        next_p, next_s = dynamics(p, s, u)
        if next_p == p and next_s == s:  # terminal state
            total_J_N[t + 1] = total_J_N[t]
            continue
        J_N += (gamma ** t) * reward
        p = next_p
        s = next_s
        total_J_N[t + 1] = J_N
    return total_J_N


if __name__ == "__main__":
    Q_trees = fitted_Q_iteration(generate_set2(10000), extremely_random_trees, stop_criterion2, 60)
    heatmaps(Q_trees)
    J_Q_trees = monte_carlo_simulations(50, 400, Q_trees)
    print("Expected reward of mu* for trees : " + str(J_Q_trees))
    pass
