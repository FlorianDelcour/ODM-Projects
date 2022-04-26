"""
    University of Liege
    INFO8003-1 - Optimal decision making for complex problems
    Assignment 1 - Reinforcement Learning in a Discrete Domain
    Authors :
        DELCOUR Florian
        MAKEDONSKY Aliocha
"""
import numpy as np
from section3 import MDP, do_action
import matplotlib.pyplot as plt
import random

grid = [[-3, 1, -5, 0, 19], [6, 3, 8, 9, 10], [5, -8, 4, 1, -8], [6, -9, 4, 19, -5], [-20, -17, -4, -3, 9]]
actions_to_take = [(1, 0), (-1, 0), (0, 1), (0, -1)]
discount_factor = 0.99


class Agent:
    def __init__(self, stochastic, N):
        """
        Initiates an instance of the class Agent.

        Parameters
        ----------
        stochastic : boolean
            Boolean indicating if the domain is stochastic (True) or deterministic (False)
        N : int
            Number of moves in the initial trajectory
        """
        self.stochastic = stochastic
        self.N = N
        self.discount_factor = 0.99
        self.traj = None
        self.rewards, self.next_states = None, None
        self.est_p = None
        self.est_r = None
        self.est_Q = None
        self.generate_traj(N)
        self.est_policy_grid = None

    def estimated_p(self):
        """
            This function computes the estimated probability of reaching all the states from a given state with a
            given action.
            It's computed from a trajectory.

            Returns
            ----------
            probas : matrices of float
                100 matrices, each one corresponding to an action taken in a given state, and representing
                the probability of transition from the state with the action to every other states
        """
        probas = np.zeros([len(actions_to_take), len(grid), len(grid[0]), len(grid), len(grid[0])])
        for actual_state in self.next_states:
            x, u = actual_state
            actual_x, actual_y = x
            action_index = actions_to_take.index(u)
            for next_state in self.next_states[actual_state]:
                next_x, next_y = next_state
                probas[action_index, actual_x, actual_y, next_x, next_y] += 1
            probas[action_index, actual_x, actual_y] = np.divide(probas[action_index, actual_x, actual_y],
                                                                 sum(sum(probas[action_index, actual_x, actual_y])))
        self.est_p = probas
        return probas

    def est_p_newx_x_u(self, new_pos, pos, action):
        """
            This function computes the estimated probability of reaching the state new_pos from the state pos with
            an action.

            Parameters
            ----------
            new_pos : list of 2 int
                The state to reach.
            pos : list of 2 int
                The state in which the agent is.
            action : list of 2 int
                The action that the agent takes.

            Returns
            ----------
            self.est_p[indx, actual_x, actual_y, new_x, new_y] : float
                The estimated probability to reach new_pos from pos with action.
        """
        indx = actions_to_take.index(action)
        actual_x, actual_y = pos
        new_x, new_y = new_pos
        return self.est_p[indx, actual_x, actual_y, new_x, new_y]

    def estimated_r(self):
        """
            This function computes the estimated reward of all the states for every action.
            It's computed from a trajectory.

            Returns
            ----------
            final_rewards : matrices of float
                4 matrices, each one corresponding to an action, and representing the reward obtained for taking
                a particular action in every state.
        """
        all_rewards = np.zeros([len(actions_to_take), len(grid), len(grid[0]), len(grid), len(grid[0])])
        final_rewards = np.zeros([len(actions_to_take), len(grid), len(grid[0])])
        for actual_state in self.rewards:
            x, u = actual_state
            actual_x, actual_y = x
            action_index = actions_to_take.index(u)
            for reward in self.rewards[actual_state]:
                next_state = self.next_states[actual_state][self.rewards[actual_state].index(reward)]
                next_x, next_y = next_state
                all_rewards[action_index, actual_x, actual_y, next_x, next_y] = reward
            all_rewards[action_index, actual_x, actual_y] = np.multiply(all_rewards[action_index, actual_x, actual_y],
                                                                        self.est_p[action_index, actual_x, actual_y])
            final_rewards[action_index, actual_x, actual_y] = sum(sum(all_rewards[action_index, actual_x, actual_y]))
        self.est_r = final_rewards
        return final_rewards

    def est_r_x_u(self, pos, action):
        """
                This function computes the estimated reward obtained when making an action from the state pos

                Parameters
                ----------
                pos : list of 2 int
                    The state in which the agent is.
                action : list of 2 int
                    The action that the agent takes.

                Returns
                ----------
                self.est_r[indx, actual_x, actual_y] : float
                    The estimated reward obtained by the agent when making action in the state pos
        """
        indx = actions_to_take.index(action)
        actual_x, actual_y = pos
        return self.est_r[indx, actual_x, actual_y]

    def estimated_Q(self, N):
        """
            This function computes the estimated Q_N functions of all the states for every action.
            It's computed from a trajectory.

            Parameters
            ----------
            N : int
                Number of steps computed.

            Returns
            ----------
            Q_n : matrices of float
                4 matrices, each one corresponding to an action, and representing the Q_N functions for an action
                in every state.
        """
        Q_prev = np.zeros([len(actions_to_take), len(grid), len(grid[0])])
        for n in range(1, N + 1):
            Q_n = np.zeros([len(actions_to_take), len(grid), len(grid[0])])
            for x in range(len(grid)):
                for y in range(len(grid[0])):
                    for i in range(len(actions_to_take)):
                        Q_n[i] = self.computation(Q_n[i], Q_prev, actions_to_take[i], (x, y))
            Q_prev = Q_n
        self.est_Q = Q_n
        return Q_n

    def computation(self, Q_n, Q_prev, action, pos):
        """
            This function computes the reward function, which gives the reward obtained when taking an action
            while the agent is in the state pos.

            Parameters
            ----------
            Q_n : matrix of float
                Matrix of zeros, corresponding to the action action (e.g. if action is
                going up, then Q_n is Q_n_up, see the call in the function Q_N_function).
            Q_prev : matrices of float
                Matrices representing the Q_N functions at the previous step (n-1) for each state, one matrix
                per action (going down, up, right, left respectively).
            pos : list of 2 int
                A state, described by its x and y coordinates.
            action : list of 2 int
                An action that the agent takes, described by a move either on the x or the y axis.

            Returns
            -------
            Q_n : matrix of float
                The new (at step n) Q_n functions for the action action
        """
        new_x, new_y = do_action(pos, action)
        reward = self.est_r_x_u(pos, action)
        proba = self.est_p_newx_x_u((new_x, new_y), pos, action)
        computation = proba * max(Q_prev[0][new_x][new_y], Q_prev[1][new_x][new_y], Q_prev[2][new_x][new_y],
                                  Q_prev[3][new_x][new_y])
        if proba != 1:
            proba_stocha = self.est_p_newx_x_u((0, 0), pos, action)
            computation += proba_stocha * max(Q_prev[0][0][0], Q_prev[1][0][0], Q_prev[2][0][0], Q_prev[3][0][0])
        x, y = pos
        Q_n[x][y] = reward + self.discount_factor * computation
        return Q_n

    def optimal_policy(self, pos):
        """
            This function returns the action to take when the agent is in state pos, to follow the optimal policy
            computed.

            Parameters
            ----------
            pos : list of 2 int
                The state in which the agent is, described by its x and y coordinates.

            Returns
            -------
            self.policy_grid[x][y] : a list of 2 int
                The action to take, described by the x an y coordinates of the action (either 1,0,-1)
        """
        if isinstance(self.est_policy_grid, np.ndarray):
            x, y = pos
            return self.est_policy_grid[x][y]
        else:
            self.compute_policy_grid()
            x, y = pos
            return self.est_policy_grid[x][y]

    def compute_policy_grid(self):
        """
            This function computes the policy_grid, which is a matrix indicating which action to take in any
            state.
        """
        self.est_policy_grid = np.zeros((len(grid), len(grid[0])), dtype=object)
        for x in range(len(grid)):
            for y in range(len(grid[0])):
                best_action = np.argmax([self.est_Q[0][x][y], self.est_Q[1][x][y], self.est_Q[2][x][y],
                                         self.est_Q[3][x][y]])
                self.est_policy_grid[x][y] = actions_to_take[best_action]

    def generate_traj(self, N):
        """
            This function generates a trajectory with N moves from a random policy.

            Parameters
            ----------
            N : int
                Number of moves computed.

            Returns
            ----------
            self.traj : list of lists of int, and of int
                The trajectory h_N = (x0, u0, r0, x1, u1, r1, ..., x_(N-1), u_(N-1), r_(N-1), x_N)
        """
        trajectory = list()
        x, y = (3, 0)
        trajectory.append((x, y))
        for step in range(N):
            action_taken_x, action_taken_y = actions_to_take[random.randint(0, 3)]
            new_x, new_y = do_action((x, y), (action_taken_x, action_taken_y))
            if self.stochastic:
                if random.uniform(0, 1) > 1 / 2:
                    new_x, new_y = (0, 0)
            reward = grid[new_x][new_y]
            trajectory.append((action_taken_x, action_taken_y))
            trajectory.append(reward)
            trajectory.append((new_x, new_y))
            x, y = new_x, new_y
        self.traj = trajectory
        self.info_from_trajectory()
        self.estimated_p()
        self.estimated_r()
        self.estimated_Q(10)
        return self.traj

    def info_from_trajectory(self):
        """
            This function generates two dictionaries, one to associate a reward to a state and an action, one to
            associate a next state to a state and an action.
        """
        self.rewards = {}
        self.next_states = {}
        for i in range(int((len(self.traj) - 1) / 3)):
            x = self.traj[3 * i]
            u = self.traj[3 * i + 1]
            r = self.traj[3 * i + 2]
            next_x = self.traj[3 * i + 3]
            if (x, u) not in self.rewards:
                self.rewards.setdefault((x, u), [])
                self.next_states.setdefault((x, u), [])
            self.rewards[(x, u)].append(r)
            self.next_states[(x, u)].append(next_x)

    def expected_return(self, N):
        """
        This function returns a matrix where each element contains the expected
        return of the random policy starting from this initial state.

        Parameters
        ----------
        N : int
            Number of size steps.

        Returns
        -------
        J_N_pre : matrix of same size as reward matrix (grid)
            Each element of the matrix contains the expected return of the
            random policy starting from this initial state.

        """
        J_N_pre = np.zeros((len(grid), len(grid[0])))
        for t in range(N):
            J_N = np.zeros((len(grid), len(grid[0])))

            for x in range(len(grid)):
                for y in range(len(grid[0])):
                    action_taken_x, action_taken_y = self.optimal_policy((x, y))
                    new_x, new_y = do_action((x, y), (action_taken_x, action_taken_y))
                    det_reward = grid[new_x][new_y]
                    if self.stochastic:
                        J_N[x][y] = 0.5 * (det_reward + grid[0][0]) + discount_factor * \
                                    0.5 * (J_N_pre[new_x][new_y] + J_N_pre[0][0])
                    else:
                        J_N[x][y] = det_reward + discount_factor * J_N_pre[new_x][new_y]
            J_N_pre = J_N
        return J_N_pre


def convergence_speed_r(true_r, stochastic):
    """
        This function computes the speed of convergence of the estimated rewards to the true rewards, for different
        lengths of trajectory.

        Parameters
        ----------
        true_r : matrices of float
            4 matrices, one for each action, indicating the rewards in each state.
        stochastic : boolean
            Boolean indicating if the domain is determinstic (False) or stochastic (True).

        Returns
        -------
        inf_norm : list of float
            List of the infinite norms between the estimated and the true rewards, one infinite norm for each length
            of trajectory.
    """
    agent = Agent(stochastic, 1)
    inf_norm = np.zeros(7)
    nb_move_traj = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
    for N in nb_move_traj:
        agent.generate_traj(N)
        estimated_r = agent.estimated_r()
        inf_norm[nb_move_traj.index(N)] = np.max(abs(estimated_r - true_r))
    plt.figure()
    plt.plot(nb_move_traj, inf_norm)
    if stochastic:
        plt.title('Speed of convergence of r in stochastic domain')
    else:
        plt.title('Speed of convergence of r in deterministic domain')
    plt.xscale('log')
    plt.xlabel('Number of moves')
    plt.ylabel('Infinite norm')
    plt.show()
    plt.close()
    return inf_norm


def convergence_speed_p(true_p, stochastic):
    """
        This function computes the speed of convergence of the estimated probabilities to the true probabilities, for
        different lengths of trajectory.

        Parameters
        ----------
        true_p : matrices of float
            100 matrices, one for each action and each state, indicating the probability of reaching every other state
            from the corresponding state by making the corresponding action.
        stochastic : boolean
            Boolean indicating if the domain is determinstic (False) or stochastic (True).

        Returns
        -------
        inf_norm : list of float
            List of the infinite norms between the estimated and the true probabilities, one infinite norm for each
            length of trajectory.
    """
    agent = Agent(stochastic, 1)
    inf_norm = np.zeros(7)
    nb_move_traj = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
    for N in nb_move_traj:
        agent.generate_traj(N)
        estimated_p = agent.estimated_p()
        inf_norm[nb_move_traj.index(N)] = np.max(abs(estimated_p - true_p))
    plt.figure()
    plt.plot(nb_move_traj, inf_norm)
    if stochastic:
        plt.title('Speed of convergence of p in stochastic domain')
    else:
        plt.title('Speed of convergence of p in deterministic domain')
    plt.xscale('log')
    plt.xlabel('Number of moves')
    plt.ylabel('Infinite norm')
    plt.show()
    plt.close()
    return inf_norm


def convergence_speed_Q(true_Q, stochastic):
    """
        This function computes the speed of convergence of the estimated Q_N functions to the true Q_N functions, for
        different lengths of trajectory.

        Parameters
        ----------
        true_Q : matrices of float
            4 matrices, one for each action, indicating the value of the Q_N functions for each state.
        stochastic : boolean
            Boolean indicating if the domain is determinstic (False) or stochastic (True).

        Returns
        -------
        inf_norm : list of float
            List of the infinite norms between the estimated and the true Q_N functions, one infinite norm for each
            length of trajectory.
    """
    agent = Agent(stochastic, 1)
    inf_norm = np.zeros(7)
    nb_move_traj = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
    for N in nb_move_traj:
        agent.generate_traj(N)
        estimated_Q = agent.estimated_Q(10)
        inf_norm[nb_move_traj.index(N)] = np.max(abs(estimated_Q - true_Q))
    return inf_norm


def compute_true_r(mdp):
    """
        This function computes the true rewards defining the MDP.

        Parameters
        ----------
        mdp : MDP
            Instance of the class MDP.

        Returns
        -------
        true_rewards : matrices of float
            4 matrices, one for each action, indicating the rewards in each state.
    """
    true_r = np.zeros([len(actions_to_take), len(grid), len(grid[0])])
    for u in actions_to_take:
        index_action = actions_to_take.index(u)
        for x in range(len(grid)):
            for y in range(len(grid[0])):
                true_r[index_action, x, y] = mdp.r_x_u((x, y), u)
    return true_r


def compute_true_p(mdp):
    """
        This function computes the true probabilities defining the MDP.

        Parameters
        ----------
        mdp : MDP
            Instance of the class MDP.

        Returns
        -------
        true_probas : matrices of float
            100 matrices, one for each action and each state, indicating the probability of reaching every other state
            from the corresponding state by making the corresponding action.
    """
    true_p = np.zeros([len(actions_to_take), len(grid), len(grid[0]), len(grid), len(grid[0])])
    for u in actions_to_take:
        index_action = actions_to_take.index(u)
        for x in range(len(grid)):
            for y in range(len(grid[0])):
                for new_x in range(len(grid)):
                    for new_y in range(len(grid[0])):
                        true_p[index_action, x, y, new_x, new_y] = mdp.p_newx_x_u((new_x, new_y), (x, y), u)
    return true_p


if __name__ == "__main__":
    deter_agent = Agent(False, 1)
    stocha_agent = Agent(True, 1)
    deter_MDP = MDP(False)
    stocha_MDP = MDP(True)

    true_r_deter = compute_true_r(deter_MDP)
    true_r_stocha = compute_true_r(stocha_MDP)
    true_p_deter = compute_true_p(deter_MDP)
    true_p_stocha = compute_true_p(stocha_MDP)
    list_true_Q_deter = (deter_MDP.Q_N_down, deter_MDP.Q_N_up, deter_MDP.Q_N_right, deter_MDP.Q_N_left)
    list_true_Q_stocha = (stocha_MDP.Q_N_down, stocha_MDP.Q_N_up, stocha_MDP.Q_N_right, stocha_MDP.Q_N_left)
    true_Q_deter = np.zeros([len(actions_to_take), len(grid), len(grid[0])])
    true_Q_stocha = np.zeros([len(actions_to_take), len(grid), len(grid[0])])
    for i in range(len(list_true_Q_deter)):
        true_Q_deter[i] = list_true_Q_deter[i]
        true_Q_stocha[i] = list_true_Q_stocha[i]

    inf_norm_r_deter = convergence_speed_r(true_r_deter, False)
    inf_norm_r_stocha = convergence_speed_r(true_r_stocha, True)
    inf_norm_p_deter = convergence_speed_p(true_p_deter, False)
    inf_norm_p_stocha = convergence_speed_p(true_p_stocha, True)
    inf_norm_Q_deter = convergence_speed_Q(true_Q_deter, False)
    inf_norm_Q_stocha = convergence_speed_Q(true_Q_stocha, True)
    print("inf norm Q_deter : ", inf_norm_Q_deter)
    print("inf norm Q_stocha : ", inf_norm_Q_stocha)

    deter_agent.generate_traj(10000000)
    stocha_agent.generate_traj(10000000)
    deter_agent.compute_policy_grid()
    stocha_agent.compute_policy_grid()
    deter_opt_policy = deter_agent.est_policy_grid
    stocha_opt_policy = stocha_agent.est_policy_grid
    print("deter_opt_policy : \n", deter_opt_policy)
    print("\n stocha_opt_policy : \n", stocha_opt_policy)

    est_deter_JN = deter_agent.expected_return(5000)
    est_stocha_JN = stocha_agent.expected_return(5000)
    print("est_deter_JN : \n", est_deter_JN)
    print("est_stocha_JN : \n", est_stocha_JN)
