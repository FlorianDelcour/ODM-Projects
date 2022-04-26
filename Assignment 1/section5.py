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


class Q_learning:
    def __init__(self, stochastic):
        self.discount_factor = 0.99
        self.learning_rate = 0.05
        self.stochastic = stochastic
        self.epsilon = 0.5
        self.traj = self.generate_traj(10 ** 7)
        self.est_Q = self.Q_offline()
        self.est_policy_grid = self.compute_policy_grid()

    def Q_offline(self):
        """
        This function computes the estimation of Q with offline Q-learning

        Returns
        -------
        Q_est : matrices of float
            4 matrices, one for each action, with the estimation of Q thanks to offline Q-learning.
        """
        Q_est = np.zeros([len(actions_to_take), len(grid), len(grid[0])])
        for k in range(int((len(self.traj) - 1) / 3)):
            x_k, y_k = self.traj[3 * k]
            u_k = self.traj[3 * k + 1]
            r_k = self.traj[3 * k + 2]
            next_x, next_y = self.traj[3 * k + 3]
            index_action = actions_to_take.index(u_k)
            Q_est[index_action, x_k, y_k] = (1 - self.learning_rate) * Q_est[index_action, x_k, y_k] + \
                                            self.learning_rate * (
                                                    r_k + self.discount_factor * max(Q_est[0, next_x, next_y],
                                                                                     Q_est[1, next_x, next_y],
                                                                                     Q_est[2, next_x, next_y],
                                                                                     Q_est[3, next_x, next_y]))
        return Q_est

    def Q_online_1(self, bonus, T=0):
        """
            This function computes the estimation of Q with online Q-learning with the first protocol so no replay
            and the learning rate is constant.

            Parameters
            -------
            bonus : boolean
                True if we want to use the softmax policy instead of the epsilon-greedy policy, False otherwise
            T : int
                Parameter for the softmax policy influencing the exploration/exploitation tradeoff

            Returns
            -------
            exp_ret_bonus : vector of matrices of float
                1 matrix for each episode, representing the expected cumulative reward
        """
        nb_episodes = 100
        nb_transitions = 1000
        inf_norm = np.zeros(nb_episodes)
        MDP_099 = MDP(self.stochastic)
        MDP_099.compute_policy_grid()
        J_N = MDP_099.expected_return(5000)
        x, y = 3, 0
        Q_est = np.zeros([len(actions_to_take), len(grid), len(grid[0])])
        self.est_Q = Q_est
        self.est_policy_grid = self.compute_policy_grid()
        if bonus:
            exp_ret_bonus = np.zeros([nb_episodes, len(grid), len(grid[0])])
        for k in range(nb_episodes):
            for t in range(nb_transitions):
                if bonus:
                    action_taken_x, action_taken_y = self.softmax((x, y), Q_est, T)
                else:
                    if random.uniform(0, 1) < self.epsilon:
                        action_taken_x, action_taken_y = actions_to_take[random.randint(0, 3)]
                    else:
                        action_taken_x, action_taken_y = self.optimal_policy((x, y))
                new_x, new_y = do_action((x, y), (action_taken_x, action_taken_y))
                action_index = actions_to_take.index((action_taken_x, action_taken_y))
                Q_est[action_index, x, y] = (1 - self.learning_rate) * Q_est[action_index, x, y] + \
                                            self.learning_rate * (grid[new_x][new_y] + self.discount_factor *
                                                                  max(Q_est[0, new_x, new_y],
                                                                      Q_est[1, new_x, new_y],
                                                                      Q_est[2, new_x, new_y],
                                                                      Q_est[3, new_x, new_y]))
                x, y = new_x, new_y
                self.est_Q = Q_est
                self.est_policy_grid = self.compute_policy_grid()
            if self.discount_factor == 0.99:
                est_J_N = self.expected_return(5000)
                inf_norm[k] = np.max(abs(est_J_N - J_N))
            else:
                MDP_04 = MDP(self.stochastic)
                MDP_04.discount_factor = 0.4
                MDP_04.Q_N_down, MDP_04.Q_N_up, MDP_04.Q_N_right, MDP_04.Q_N_left = MDP_04.Q_N_function(7)
                true_Q = np.zeros([len(actions_to_take), len(grid), len(grid[0])])
                true_Q[0] = MDP_04.Q_N_down
                true_Q[1] = MDP_04.Q_N_up
                true_Q[2] = MDP_04.Q_N_right
                true_Q[3] = MDP_04.Q_N_left
                inf_norm[k] = np.max(abs(Q_est - true_Q))
            if bonus:
                est_J_N = self.expected_return(5000)
                exp_ret_bonus[k] = est_J_N
        if self.discount_factor == 0.99:
            plt.figure()
            plt.plot(range(nb_episodes), inf_norm)
            if self.stochastic:
                plt.title(r'||$J^{N}_{\mu_{\hat{Q}}} - J^N_{\mu^*}$||$_\infty$, first protocol, stochastic domain, '
                          r'$\gamma = 0.99$')
            else:
                plt.title(r'||$J^{N}_{\mu_{\hat{Q}}} - J^N_{\mu^*}$||$_\infty$, first protocol, deterministic domain, '
                          r'$\gamma = 0.99$')
            plt.xlabel('Episode number')
            plt.ylabel('Infinite norm')
            plt.savefig("1st_gamma_099_stocha_" + str(self.stochastic) + "_bonus_" + str(bonus) + ".png")
            plt.close()
        else:
            plt.figure()
            plt.plot(range(nb_episodes), inf_norm)
            if self.stochastic:
                plt.title(r'||$\hat{Q} - Q_N$||$_\infty$, first protocol, stochastic domain, $\gamma = 0.4$')
            else:
                plt.title(r'||$\hat{Q} - Q_N$||$_\infty$, first protocol, deterministic domain, $\gamma = 0.4$')
            plt.xlabel('Episode number')
            plt.ylabel('Infinite norm')
            plt.savefig("1st_gamma_04_stocha_" + str(self.stochastic) + "_bonus_" + str(bonus) + ".png")
            plt.close()
        if bonus:
            return exp_ret_bonus

    def Q_online_2(self, bonus):
        """
            This function computes the estimation of Q with online Q-learning with the second protocol so no replay
            and the learning rate is decreases with the time.

            Parameters
            -------
            bonus : boolean
                True if we want to use the softmax policy instead of the epsilon-greedy policy, False otherwise
        """
        nb_episodes = 100
        nb_transitions = 1000
        inf_norm = np.zeros(nb_episodes)
        MDP_099 = MDP(self.stochastic)
        MDP_099.compute_policy_grid()
        J_N = MDP_099.expected_return(5000)
        x, y = 3, 0
        Q_est = np.zeros([len(actions_to_take), len(grid), len(grid[0])])
        self.est_Q = Q_est
        self.est_policy_grid = self.compute_policy_grid()
        for k in range(nb_episodes):
            alpha = self.learning_rate
            for t in range(nb_transitions):
                if random.uniform(0, 1) < self.epsilon:
                    action_taken_x, action_taken_y = actions_to_take[random.randint(0, 3)]
                else:
                    action_taken_x, action_taken_y = self.optimal_policy((x, y))
                new_x, new_y = do_action((x, y), (action_taken_x, action_taken_y))
                action_index = actions_to_take.index((action_taken_x, action_taken_y))
                Q_est[action_index, x, y] = (1 - alpha) * Q_est[action_index, x, y] + \
                                            alpha * (grid[new_x][new_y] + self.discount_factor *
                                                     max(Q_est[0, new_x, new_y],
                                                         Q_est[1, new_x, new_y],
                                                         Q_est[2, new_x, new_y],
                                                         Q_est[3, new_x, new_y]))
                x, y = new_x, new_y
                alpha *= 0.8
                self.est_Q = Q_est
                self.est_policy_grid = self.compute_policy_grid()
            if self.discount_factor == 0.99:
                est_J_N = self.expected_return(5000)
                inf_norm[k] = np.max(abs(est_J_N - J_N))
            else:
                MDP_04 = MDP(self.stochastic)
                MDP_04.discount_factor = 0.4
                MDP_04.Q_N_down, MDP_04.Q_N_up, MDP_04.Q_N_right, MDP_04.Q_N_left = MDP_04.Q_N_function(7)
                true_Q = np.zeros([len(actions_to_take), len(grid), len(grid[0])])
                true_Q[0] = MDP_04.Q_N_down
                true_Q[1] = MDP_04.Q_N_up
                true_Q[2] = MDP_04.Q_N_right
                true_Q[3] = MDP_04.Q_N_left
                inf_norm[k] = np.max(abs(Q_est - true_Q))
        if self.discount_factor == 0.99:
            plt.figure()
            plt.plot(range(nb_episodes), inf_norm)
            if self.stochastic:
                plt.title(r'||$J^{N}_{\mu_{\hat{Q}}} - J^N_{\mu^*}$||$_\infty$, second protocol, stochastic domain, '
                          r'$\gamma = 0.99$')
            else:
                plt.title(r'||$J^{N}_{\mu_{\hat{Q}}} - J^N_{\mu^*}$||$_\infty$, second protocol, deterministic '
                          r'domain, $\gamma = 0.99$')
            plt.xlabel('Episode number')
            plt.ylabel('Infinite norm')
            plt.savefig("2nd_gamma_099_stocha_" + str(self.stochastic) + "_bonus_" + str(bonus) + ".png")
            plt.close()
        else:
            plt.figure()
            plt.plot(range(nb_episodes), inf_norm)
            if self.stochastic:
                plt.title(r'||$\hat{Q} - Q_N$||$_\infty$, second protocol, stochastic domain, $\gamma = 0.4$')
            else:
                plt.title(r'||$\hat{Q} - Q_N$||$_\infty$, second protocol, deterministic domain, $\gamma = 0.4$')
            plt.xlabel('Episode number')
            plt.ylabel('Infinite norm')
            plt.savefig("2nd_gamma_04_stocha_" + str(self.stochastic) + ".png")
            plt.close()

    def Q_online_3(self, bonus):
        """
            This function computes the estimation of Q with online Q-learning with the third protocol so there is replay
            and the learning rate is constant.

            Parameters
            -------
            bonus : boolean
                True if we want to use the softmax policy instead of the epsilon-greedy policy, False otherwise
        """
        nb_episodes = 100
        nb_transitions = 1000
        inf_norm = np.zeros(nb_episodes)
        MDP_099 = MDP(self.stochastic)
        MDP_099.compute_policy_grid()
        J_N = MDP_099.expected_return(5000)
        x, y = 3, 0
        replay_buffer = []
        Q_est = np.zeros([len(actions_to_take), len(grid), len(grid[0])])
        self.est_Q = Q_est
        self.est_policy_grid = self.compute_policy_grid()
        for k in range(nb_episodes):
            for t in range(nb_transitions):
                if random.uniform(0, 1) < self.epsilon:
                    action_taken_x, action_taken_y = actions_to_take[random.randint(0, 3)]
                else:
                    action_taken_x, action_taken_y = self.optimal_policy((x, y))
                replay_buffer.append((action_taken_x, action_taken_y))
                for i in range(10):
                    action_taken_x, action_taken_y = replay_buffer[random.randint(0, len(replay_buffer) - 1)]
                    new_x, new_y = do_action((x, y), (action_taken_x, action_taken_y))
                    action_index = actions_to_take.index((action_taken_x, action_taken_y))
                    Q_est[action_index, x, y] = (1 - self.learning_rate) * Q_est[action_index, x, y] + \
                                                self.learning_rate * (grid[new_x][new_y] + self.discount_factor *
                                                                      max(Q_est[0, new_x, new_y],
                                                                          Q_est[1, new_x, new_y],
                                                                          Q_est[2, new_x, new_y],
                                                                          Q_est[3, new_x, new_y]))
                    x, y = new_x, new_y
                self.est_Q = Q_est
                self.est_policy_grid = self.compute_policy_grid()
            if self.discount_factor == 0.99:
                est_J_N = self.expected_return(5000)
                inf_norm[k] = np.max(abs(est_J_N - J_N))
            else:
                MDP_04 = MDP(self.stochastic)
                MDP_04.discount_factor = 0.4
                MDP_04.Q_N_down, MDP_04.Q_N_up, MDP_04.Q_N_right, MDP_04.Q_N_left = MDP_04.Q_N_function(7)
                true_Q = np.zeros([len(actions_to_take), len(grid), len(grid[0])])
                true_Q[0] = MDP_04.Q_N_down
                true_Q[1] = MDP_04.Q_N_up
                true_Q[2] = MDP_04.Q_N_right
                true_Q[3] = MDP_04.Q_N_left
                inf_norm[k] = np.max(abs(Q_est - true_Q))
        if self.discount_factor == 0.99:
            plt.figure()
            plt.plot(range(nb_episodes), inf_norm)
            if self.stochastic:
                plt.title(r'||$J^{N}_{\mu_{\hat{Q}}} - J^N_{\mu^*}$||$_\infty$, third protocol, stochastic domain, '
                          r'$\gamma = 0.99$')
            else:
                plt.title(r'||$J^{N}_{\mu_{\hat{Q}}} - J^N_{\mu^*}$||$_\infty$, third protocol, deterministic domain, '
                          r'$\gamma = 0.99$')
            plt.xlabel('Episode number')
            plt.ylabel('Infinite norm')
            plt.savefig("3rd_gamma_099_stocha_" + str(self.stochastic) + "_bonus_" + str(bonus) + ".png")
            plt.close()
        else:
            plt.figure()
            plt.plot(range(nb_episodes), inf_norm)
            if self.stochastic:
                plt.title(r'||$\hat{Q} - Q_N$||$_\infty$, third protocol, stochastic domain, $\gamma = 0.4$')
            else:
                plt.title(r'||$\hat{Q} - Q_N$||$_\infty$, third protocol, deterministic domain, $\gamma = 0.4$')
            plt.xlabel('Episode number')
            plt.ylabel('Infinite norm')
            plt.savefig("3rd_gamma_04_stocha_" + str(self.stochastic) + ".png")
            plt.close()

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
        return self.est_policy_grid

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
                        J_N[x][y] = 0.5 * (det_reward + grid[0][0]) + self.discount_factor * \
                                    0.5 * (J_N_pre[new_x][new_y] + J_N_pre[0][0])
                    else:
                        J_N[x][y] = det_reward + self.discount_factor * J_N_pre[new_x][new_y]
            J_N_pre = J_N
        return J_N_pre

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
        return trajectory

    def softmax(self, pos, Q, T):
        """
            This function implements the softmax policy

            Parameters
            ----------
            pos : list of 2 int
                Position of the agent.
            Q : matrices of float
                4 matrices, one for each action, representing the Q functions
            T : int
                Influence the exploration/exploitation tradeoff. Higher it is, more the agent will explore.

            Returns
            ----------
            actions_to_take[index_action_taken] : list of 2 int
                The actions taken by the agent, determined by the softmax policy
        """
        probas = np.zeros(len(actions_to_take))
        x, y = pos
        for u in range(len(actions_to_take)):
            sum_normalize = 0
            for all_u in range(len(actions_to_take)):
                sum_normalize += np.exp(Q[all_u, x, y] / T)
            probas[u] = np.exp(Q[u, x, y] / T) / sum_normalize
        index_action_taken = np.random.choice([0, 1, 2, 3], size=1, p=probas)[0]
        return actions_to_take[int(index_action_taken)]


if __name__ == "__main__":
    # section 5.1 deter
    agent_deter = Q_learning(False)
    print("policy table _deter: \n", agent_deter.est_policy_grid)
    J_N_mu_est = agent_deter.expected_return(5000)
    print("J_N mu estimated _deter: \n", J_N_mu_est)
    # section 5.2 deter
    
    agent_deter.Q_online_1(False)

    agent_deter.Q_online_2(False)
    
    agent_deter.Q_online_3(False)

    # section 5.3 deter
    agent_deter.discount_factor = 0.4
    agent_deter.est_Q = agent_deter.Q_offline()
    agent_deter.est_policy_grid = agent_deter.compute_policy_grid()

    agent_deter.Q_online_1(False)
    
    agent_deter.Q_online_2(False)
    
    agent_deter.Q_online_3(False)


    # section 5.1 stocha
    agent_stocha = Q_learning(True)
    print("policy table _stocha: \n", agent_stocha.est_policy_grid)
    J_N_mu_est = agent_stocha.expected_return(5000)
    print("J_N mu estimated _stocha: \n", J_N_mu_est)
    # section 5.2 stocha

    agent_stocha.Q_online_1(False)
    
    agent_stocha.Q_online_2(False)
    
    agent_stocha.Q_online_3(False)

    # section 5.3 stocha
    agent_stocha.discount_factor = 0.4
    agent_stocha.est_Q = agent_stocha.Q_offline()
    agent_stocha.est_policy_grid = agent_stocha.compute_policy_grid()

    agent_stocha.Q_online_1(False)
    
    agent_stocha.Q_online_2(False)

    agent_stocha.Q_online_3(False)

    # --------------------------------------
    # section 5.4
    agent_deter = Q_learning(False)
    # section 5.2 deter
    diff_T = [100, 1000, 10000, 100000]
    plt.figure()
    for T in diff_T:
        diff_J = agent_deter.Q_online_1(True, T)
        diff_J_summed = [sum(sum(diff_J[i])) for i in range(len(diff_J))]
        plt.plot(range(len(diff_J_summed)), diff_J_summed)
    plt.legend(['T = 100', 'T = 1000', 'T = 10000', 'T = 100000'])
    plt.title('Expected cumulative reward according to the number of episodes')
    plt.xlabel('Episode number')
    plt.ylabel('Expected cumulative reward')
    plt.savefig("bonus_deter.png")
    plt.close()
    agent_deter.Q_online_1(True, 100)

    # section 5.1 stocha
    agent_stocha = Q_learning(True)
    # section 5.2 stocha
    diff_T = [100, 1000, 10000, 100000]
    plt.figure()
    for T in diff_T:
        diff_J = agent_stocha.Q_online_1(True, T)
        diff_J_summed = [sum(sum(diff_J[i])) for i in range(len(diff_J))]
        plt.plot(range(len(diff_J_summed)), diff_J_summed)
    plt.legend(['T = 100', 'T = 1000', 'T = 10000', 'T = 100000'])
    plt.title('Expected cumulative reward according to the number of episodes')
    plt.xlabel('Episode number')
    plt.ylabel('Expected cumulative reward')
    plt.savefig("bonus_stocha.png")
    plt.close()

