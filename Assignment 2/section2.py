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

random.seed(42)


def expected_return(N):
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
        u = random_policy()
        reward = reward_signal(p, s, u)
        next_p, next_s = dynamics(p, s, u)
        if next_p == p and next_s == s:  # terminal state
            total_J_N[t + 1] = total_J_N[t]
            continue
        J_N += (discount_factor ** t) * reward
        p = next_p
        s = next_s
        total_J_N[t + 1] = J_N
    return total_J_N


def monte_carlo_simulations(nb_init_states, N):
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
        total_J_N = total_J_N + expected_return(N)
    total_J_N = total_J_N / nb_init_states
    # total_J_N is constant after a certain number of iterations because we reached a terminal state in each of the
    # simulations, so we cut the graph to make it more readable
    """
    counter_same = 0
    final_J_N = []
    for t in range(len(total_J_N) - 1):
        if total_J_N[t + 1] == total_J_N[t]:
            counter_same += 1
            if counter_same == 50 or t == len(total_J_N) - 2:
                for i in range(t):
                    final_J_N.append(total_J_N[i])
                break
        else:
            counter_same = 0
    

    print("Value of J_mu_400 : " + str(final_J_N[-1]))
    plt.figure()
    plt.plot(range(len(final_J_N)), final_J_N)
    plt.title(r'Evolution of $J^{\mu}_{400}$ over ' + str(nb_init_states) + ' simulations')
    plt.xlabel('N')
    plt.ylabel(r'$J^{\mu}_{400}$')
    plt.savefig("Expected_return_" + str(nb_init_states) + "_simulations.png")
    plt.close()
    """
    print("Value of J_mu_400 : " + str(total_J_N[-1]))
    plt.figure()
    plt.plot(range(len(total_J_N)), total_J_N)
    plt.title(r'Evolution of $J^{\mu}_{400}$ over ' + str(nb_init_states) + ' simulations')
    plt.xlabel('N')
    plt.ylabel(r'$J^{\mu}_{400}$')
    plt.savefig("Expected_return_" + str(nb_init_states) + "_simulations.png")
    plt.close()


if __name__ == "__main__":
    monte_carlo_simulations(50, 400)
