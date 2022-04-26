"""
    University of Liege
    INFO8003-1 - Optimal decision making for complex problems
    Assignment 1 - Reinforcement Learning in a Discrete Domain
    Authors : 
        DELCOUR Florian
        MAKEDONSKY Aliocha
"""

import random
import numpy as np

grid = [[-3, 1, -5, 0, 19], [6, 3, 8, 9, 10], [5, -8, 4, 1, -8], [6, -9, 4, 19, -5], [-20, -17, -4, -3, 9]]
actions_to_take = [(1, 0), (-1, 0), (0, 1), (0, -1)]
discount_factor = 0.99


class Agent:

    def __init__(self):
        self.actual_position_x = 3
        self.actual_position_y = 0
        self.total_reward = 0

    def naive_policy(self):
        """
        This function simulates an action of a random policy. It selects at random 
        an action from {go down, go up, go left, go right}, and moves the agent 
        from its actual position to the next one given the action.
        """
        action_taken_x, action_taken_y = actions_to_take[random.randint(0, 3)]
        x = self.actual_position_x + action_taken_x
        y = self.actual_position_y + action_taken_y
        # We must check that the agent doesn't go out of the bounds
        if len(grid[0]) > x >= 0 and len(grid) > y >= 0:
            self.actual_position_x = x
            self.actual_position_y = y

    def expected_return(self, N, stochastic):
        """
        This function returns a matrix where each element contains the expected
        return of the random policy starting from this initial state.

        Parameters
        ----------
        N : int
            Number of size steps  
        stochastic : boolean
            If true, the computation of the matrix will be performed in a 
            stochastic domain. If false, it will be in deterministic one.

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
                    self.actual_position_x = x
                    self.actual_position_y = y
                    self.naive_policy()
                    new_x = self.actual_position_x
                    new_y = self.actual_position_y
                    det_reward = grid[new_x][new_y]
                    if stochastic: 
                        J_N[x][y] = 0.5*(det_reward+grid[0][0]) + discount_factor * \
                                    0.5*(J_N_pre[new_x][new_y]+J_N_pre[0][0])
                    else:
                        J_N[x][y] = det_reward + discount_factor*J_N_pre[new_x][new_y]
            J_N_pre = J_N
        return J_N_pre


if __name__ == "__main__":
    agent = Agent()
    J_N_deter = agent.expected_return(5000, False)
    J_N_stocha = agent.expected_return(5000, True)
    print("J_N_deter : \n")
    print(J_N_deter)
    print("\n \n J_N_stocha : \n")
    print(J_N_stocha)
