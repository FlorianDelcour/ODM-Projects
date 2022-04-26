"""
    University of Liege
    INFO8003-1 - Optimal decision making for complex problems
    Assignment 1 - Reinforcement Learning in a Discrete Domain
    Authors :
        DELCOUR Florian
        MAKEDONSKY Aliocha
"""
import numpy
import numpy as np

grid = [[-3, 1, -5, 0, 19], [6, 3, 8, 9, 10], [5, -8, 4, 1, -8], [6, -9, 4, 19, -5], [-20, -17, -4, -3, 9]]
actions_to_take = [(1, 0), (-1, 0), (0, 1), (0, -1)]
discount_factor = 0.99


class MDP:

    def __init__(self, stochastic):
        """
        Initiates an instance of the class MDP.

        Parameters
        ----------
        stochastic : boolean
            Boolean indicating if the domain is stochastic (True) or deterministic (False)
        """
        self.stochastic = stochastic
        self.discount_factor = 0.99
        self.Q_N_down, self.Q_N_up, self.Q_N_right, self.Q_N_left = self.Q_N_function(7)
        self.policy_grid = None

    def r_x_u(self, pos, action):
        """
            This function computes the reward function, which gives the reward obtained when taking an action
            while the agent is in the state pos.

            Parameters
            ----------
            pos : list of 2 int
                A state, described by its x and y coordinates.
            action : list of 2 int
                An action that the agent takes, described by a move either on the x or the y axis.

            Returns
            -------
            grid[new_x][new_y] : int
                The reward obtained by the agent in its new state.
        """
        new_x, new_y = do_action(pos, action)
        if self.stochastic:
            return 0.5 * grid[0][0] + 0.5 * grid[new_x][new_y]
        else:
            return grid[new_x][new_y]

    def p_newx_x_u(self, new_pos, pos, action):
        """
            This function computes the probability for the agent to arrive in state new_pos when making action
            in the state pos.

            Parameters
            ----------
            new_pos : list of 2 int
                The state in which the agent may arrive, described by its x and y coordinates.
            pos : list of 2 int
                The state in which the agent is, described by its x and y coordinates.
            action : list of 2 int
                An action that the agent takes, described by a move either on the x or the y axis.

            Returns
            -------
            The probability (a float) for the agent to be in state new_pos after taking action in state pos.
        """
        pos_obtained = do_action(pos, action)
        if self.stochastic:
            return 0.5 * equality_pos(pos_obtained, new_pos) + 0.5 * equality_pos((0, 0), new_pos)
        else:
            return equality_pos(new_pos, pos_obtained)

    def Q_N_function(self, N):
        """
            This function computes the Q_N functions for each of the possible actions, and put the final result
            for each state in a matrix corresponding to each of the actions.

            Parameters
            ----------
            N : int
                The number of iteration taken to compute the Q_N functions

            Returns
            -------
            Q_n_down, Q_n_up, Q_n_right, Q_n_left : matrices of float
                Matrices representing the Q_N functions for each state, one matrix per action (going down, up, right,
                left respectively).

        """
        Q_prev_down = np.zeros((len(grid), len(grid[0])))
        Q_prev_up = np.zeros((len(grid), len(grid[0])))
        Q_prev_right = np.zeros((len(grid), len(grid[0])))
        Q_prev_left = np.zeros((len(grid), len(grid[0])))
        for n in range(1, N + 1):
            Q_n_down = np.zeros((len(grid), len(grid[0])))
            Q_n_up = np.zeros((len(grid), len(grid[0])))
            Q_n_right = np.zeros((len(grid), len(grid[0])))
            Q_n_left = np.zeros((len(grid), len(grid[0])))
            for x in range(len(grid)):
                for y in range(len(grid[0])):
                    Q_n_down = self.computation(Q_n_down, Q_prev_down, Q_prev_up, Q_prev_right, Q_prev_left,
                                                actions_to_take[0], (x, y))
                    Q_n_up = self.computation(Q_n_up, Q_prev_down, Q_prev_up, Q_prev_right, Q_prev_left,
                                              actions_to_take[1], (x, y))
                    Q_n_right = self.computation(Q_n_right, Q_prev_down, Q_prev_up, Q_prev_right, Q_prev_left,
                                                 actions_to_take[2], (x, y))
                    Q_n_left = self.computation(Q_n_left, Q_prev_down, Q_prev_up, Q_prev_right, Q_prev_left,
                                                actions_to_take[3], (x, y))
            Q_prev_down = Q_n_down
            Q_prev_up = Q_n_up
            Q_prev_right = Q_n_right
            Q_prev_left = Q_n_left
        return Q_n_down, Q_n_up, Q_n_right, Q_n_left

    def computation(self, Q_n, Q_prev_down, Q_prev_up, Q_prev_right, Q_prev_left, action, pos):
        """
            This function computes the reward function, which gives the reward obtained when taking an action
            while the agent is in the state pos.

            Parameters
            ----------
            Q_n : matrix of float
                Matrix of zeros, corresponding to the action action (e.g. if action is
                going up, then Q_n is Q_n_up, see the call in the function Q_N_function).
            Q_prev_down, Q_prev_up, Q_prev_right, Q_prev_left : matrices of float
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
        reward = self.r_x_u(pos, action)
        proba = self.p_newx_x_u((new_x, new_y), pos, action)
        computation = proba * max(Q_prev_down[new_x][new_y], Q_prev_up[new_x][new_y],
                                           Q_prev_right[new_x][new_y], Q_prev_left[new_x][new_y])
        if proba != 1:
            proba_stocha = self.p_newx_x_u((0, 0), pos, action)
            computation += proba_stocha * max(Q_prev_down[0][0], Q_prev_up[0][0], Q_prev_right[0][0], Q_prev_left[0][0])
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
        if isinstance(self.policy_grid, np.ndarray):
            x, y = pos
            return self.policy_grid[x][y]
        else:
            self.compute_policy_grid()
            x, y = pos
            return self.policy_grid[x][y]

    def compute_policy_grid(self):
        """
            This function computes the policy_grid, which is a matrix indicating which action to take in any
            state.
        """
        self.policy_grid = np.zeros((len(grid), len(grid[0])), dtype=object)
        for x in range(len(grid)):
            for y in range(len(grid[0])):
                best_action = np.argmax([self.Q_N_down[x][y], self.Q_N_up[x][y], self.Q_N_right[x][y],
                                         self.Q_N_left[x][y]])
                self.policy_grid[x][y] = actions_to_take[best_action]

    def expected_return(self, N):
        """
        This function returns a matrix where each element contains the expected
        return of the random policy starting from this initial state.

        Parameters
        ----------
        N : int
            Number of size steps

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
                    # print("x : ", x, "; y : ", y, "; action_taken_x : ", action_taken_x, "; action_taken_y : ",
                    #     action_taken_y)
                    new_x, new_y = do_action((x, y), (action_taken_x, action_taken_y))
                    det_reward = grid[new_x][new_y]
                    if self.stochastic:
                        J_N[x][y] = 0.5 * (det_reward + grid[0][0]) + discount_factor * \
                                    0.5 * (J_N_pre[new_x][new_y] + J_N_pre[0][0])
                    else:
                        J_N[x][y] = det_reward + discount_factor * J_N_pre[new_x][new_y]
            J_N_pre = J_N
        return J_N_pre

    def determine_best_N(self):
        """
            This function computes the optimal policy grid associated to the Q_N functions 
            for N=1 to 15 and prints TRUE if the previous optimal policy grid is equal to the actual one 
            depending on the iteration N.
        """
        N = 1
        self.Q_N_down, self.Q_N_up, self.Q_N_right, self.Q_N_left = self.Q_N_function(N)
        self.compute_policy_grid()
        prev_policy_grid = None
        actual_policy_grid = self.policy_grid
        if self.stochastic:
            print("Stochastic domain :")
        else:
            print("Deterministic domain :")
        while N != 15:
            print("N=", N)
            print(np.array_equal(actual_policy_grid, prev_policy_grid), "\n")
            N += 1
            self.Q_N_down, self.Q_N_up, self.Q_N_right, self.Q_N_left = self.Q_N_function(N)
            prev_policy_grid = actual_policy_grid
            self.compute_policy_grid()
            actual_policy_grid = self.policy_grid
        print()


def equality_pos(pos1, pos2):
    """
        This function returns a boolean indicating if pos1 and pos2 are the same.

        Parameters
        ----------
        pos1, pos2 : list of 2 int
            A state, described by its x and y coordinates.

        Returns
        -------
        a boolean, indicating if pos1 and pos2 represent the same state.

        """
    x_pos1, y_pos1 = pos1
    x_pos2, y_pos2 = pos2
    if x_pos1 == x_pos2 and y_pos1 == y_pos2:
        return 1
    return 0


def do_action(pos, action):
    """
        This function returns the state obtained when taking the action given as parameter when the agent
        is in the state described by pos.

        Parameters
        ----------
        pos : list of 2 int
            A state, described by its x and y coordinates.
        action : list of 2 int
            An action that the agent takes, described by a move either on the x or the y axis.

        Returns
        -------
        x, y : 2 ints
            The coordinates representing the state of the agent, obtained after taking the action.

    """
    action_taken_x, action_taken_y = action
    x, y = pos
    new_x = x + action_taken_x
    new_y = y + action_taken_y
    # We must check that the agent doesn't go out of the bounds
    if len(grid[0]) > new_x >= 0 and len(grid) > new_y >= 0:
        return new_x, new_y
    return x, y


def print_policy_smoothly(policy_grid):
    """
        This function replaces every action contained in policy_grid by a letter corresponding to the action, where
        'u' stands for 'up', 'd' for 'down', 'r' for 'right', 'l' for 'left'.

        Parameters
        ----------
        policy_grid : matrix of lists of 2 int
            Matrix indicating which action to take in each state to follow the optimal policy.

        Returns
        -------
        smooth_grid : matrix of char
            Same matrix than policy_grid but where the lists indicating the action to take are replaced by
            a letter indicating the action to take.

    """
    smooth_grid = np.zeros((len(policy_grid), len(policy_grid[0])), dtype=str)
    for x in range(len(policy_grid)):
        for y in range(len(policy_grid[0])):
            if policy_grid[x][y] == (1, 0):
                smooth_grid[x][y] = "d"
            elif policy_grid[x][y] == (-1, 0):
                smooth_grid[x][y] = "u"
            elif policy_grid[x][y] == (0, 1):
                smooth_grid[x][y] = "r"
            else:
                smooth_grid[x][y] = "l"
    return smooth_grid


if __name__ == "__main__":
    deter_agent = MDP(stochastic=False)
    stocha_agent = MDP(stochastic=True)
    deter_agent.determine_best_N()
    stocha_agent.determine_best_N()
    J_deter = deter_agent.expected_return(5000)
    J_stocha = stocha_agent.expected_return(5000)
    deter_agent.compute_policy_grid()
    stocha_agent.compute_policy_grid()
    deter_policy = deter_agent.policy_grid
    deter_policy = print_policy_smoothly(deter_policy)
    stocha_policy = stocha_agent.policy_grid
    stocha_policy = print_policy_smoothly(stocha_policy)

    print("\n J_deter : \n", J_deter)
    print("\n J_stocha : \n", J_stocha)
    print("\n deter_policy : \n", deter_policy)
    print("\n stocha_policy : \n", stocha_policy)
