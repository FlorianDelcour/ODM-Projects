"""
    University of Liege
    INFO8003-1 - Optimal decision making for complex problems
    Assignment 1 - Reinforcement Learning in a Discrete Domain
    Authors : 
        DELCOUR Florian
        MAKEDONSKY Aliocha
"""

import random

grid = [[-3, 1, -5, 0, 19], [6, 3, 8, 9, 10], [5, -8, 4, 1, -8], 
        [6, -9, 4, 19, -5], [-20, -17, -4, -3, 9]]
actions_to_take = [(1, 0), (-1, 0), (0, 1), (0, -1)]
discount_factor = 0.99


class Agent:

    def __init__(self):
        """
        This function initializes the agent at position x0 = (3,0) on the map
        previously defined by "grid".
        """
        self.actual_position_x = 3
        self.actual_position_y = 0

    def naive_policy(self, nb_steps, stochastic):
        """
        This function simulates a naive policy where actions are selected at random
        using a deterministic or stochastic domain. 

        Parameters
        ----------
        nb_steps : int 
                number of steps that the agent performs
        stochastic : boolean
                True if stochastic domain, False if deterministic domain
        """
        for t in range(nb_steps):
            previous_state_x = self.actual_position_x
            previous_state_y = self.actual_position_y
            if stochastic:
                if random.uniform(0, 1) <= 1/2:
                    action_taken_x, action_taken_y = actions_to_take[random.randint(0, 3)]
                else:
                    # If w > 1/2, then the agent moves to state (0,0). Below in the code,
                    # the value of x is actual_position_x + action_taken_x so indeed it will make 0.
                    action_taken_x, action_taken_y = (-self.actual_position_x, -self.actual_position_y)
            else:
                action_taken_x, action_taken_y = actions_to_take[random.randint(0, 3)]
            x = self.actual_position_x + action_taken_x
            y = self.actual_position_y + action_taken_y
            # We must check that the agent doesn't go out of the bounds
            if len(grid) > x >= 0 and len(grid[0]) > y >= 0:
                self.actual_position_x = x
                self.actual_position_y = y
            # If it goes out of the bounds, the agent doesn't move
            x = self.actual_position_x
            y = self.actual_position_y
            reward = grid[x][y]
            x_t = "x_" + str(t) + " = (" + str(previous_state_x) + "," + str(previous_state_y) + ")"
            u_t = "u_" + str(t) + " = (" + str(action_taken_x) + "," + str(action_taken_y) + ")"
            r_t = "r_" + str(t) + " = " + str(reward)
            x_t_next = "x_"+ str(t+1) + " = (" + str(self.actual_position_x) + "," + str(self.actual_position_y) + ")"
            print("Step : (" + x_t + ", " + u_t + ", " + r_t + ", " + x_t_next + ")\n")
        # Reset agent position
        self.actual_position_x = 3
        self.actual_position_y = 0

if __name__ == "__main__":
    agent = Agent()
    print("Deterministic : \n")
    agent.naive_policy(nb_steps=10, stochastic=False)
    print("Stochastic : \n")
    agent.naive_policy(nb_steps=10, stochastic=True)
