"""
    University of Liege
    INFO8003-1 - Optimal decision making for complex problems
    Assignment 3 - Deep Reinforcement Learning with Images for the Car on the Hill Problem
    Authors : 
        DELCOUR Florian
        MAKEDONSKY Aliocha
"""

import numpy as np
import random

# Globals
action_space = [-4, 4]
m = 1.
g = 9.81
time_step = 1e-1
integration_step = 1e-3
gamma = 0.95
eps = 1e-2  # precision
B_r = 1  # sup |r|


def random_policy():
    """
        This functions returns a random action defined over the action space.
    """
    return action_space[random.randint(0, 1)]


def hill(p):
    """
        This function computes the hill function evaluated in the position p.

        Parameters
        ----------
        p : float
            Position of the car

        Returns
        ----------
            The value (float) of the hill function evaluated in the position p
    """
    if p < 0:
        return p + p ** 2
    else:
        return p / np.sqrt(1 + 5 * (p ** 2))


def hill_prime(p):
    """
        This function computes the derivative w.r.t. p of the hill function, evaluated in the position p.

        Parameters
        ----------
        p : float
            Position of the car

        Returns
        ----------
            The value (float) of the derivative w.r.t. p of the hill function, evaluated in the position p
    """
    if p < 0:
        return 1 + 2 * p
    else:
        return 1 / np.power(1 + 5 * (p ** 2), 1.5)


def hill_prime2(p):
    """
        This function computes the second derivative w.r.t. p of the hill function, evaluated in the position p.

        Parameters
        ----------
        p : float
            Position of the car

        Returns
        ----------
            The value (float) of the second derivative w.r.t. p of the hill function, evaluated in the position p
    """
    if p < 0:
        return 2
    else:
        return (-15 * p) / np.power(1 + 5 * (p ** 2), 2.5)


def dynamics(p, s, u):
    """
        This function computes the new state of the car, so its new position p and new speed s, according to the
        dynamics of the system, when taking the action u

        Parameters
        ----------
        p : float
            Position of the car
        s : float
            Speed of the car
        u : int
            Action to take (either +4, or -4)

        Returns
        ----------
            next_p, next_s : 2 floats
                Next position of the car (next_p) and next speed of the car (next_s) according to the dynamics
    """
    if np.abs(p) >= 1 or np.abs(s) >= 3:  # the agent is in a terminal state, it is stuck in this terminal state
        return p, s

    next_p = p
    next_s = s
    for t in range(int(time_step / integration_step)):
        p_dot = next_s
        s_dot = (u / (m * (1 + hill_prime(next_p) ** 2))) - ((g * hill_prime(next_p)) / (1 + hill_prime(next_p) ** 2)) \
                - (((next_s ** 2) * hill_prime(next_p) * hill_prime2(next_p)) / (1 + hill_prime(next_p) ** 2))
        next_p += integration_step * p_dot
        next_s += integration_step * s_dot
    if np.abs(next_p) > 1:
        next_p = -1 if p < 0 else 1
    if np.abs(next_s) > 3:
        next_s = -3 if s < 0 else 3
    return next_p, next_s


def reward_signal(p, s, u):
    """
        This function computes the reward signal obtained by the car when taking the action u in the state (p,s)

        Parameters
        ----------
        p : float
            Position of the car
        s : float
            Speed of the car
        u : int
            Action to take (either +4, or -4)

        Returns
        ----------
        Reward : int
    """
    next_p, next_s = dynamics(p, s, u)
    if next_p == p and next_s == s:  # the agent is in a terminal state, all the rewards are 0
        return 0
    elif next_p < -1 or np.abs(next_s) > 3:
        return -1
    elif next_p > 1 and np.abs(next_s) <= 3:
        return 1
    else:
        return 0
