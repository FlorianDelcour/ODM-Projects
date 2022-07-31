import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from gym_anm.envs import ANM6Easy
import warnings

warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device : ", device)

# Global variables
env = ANM6Easy()
NB_ACTION_VAR = env.action_space.shape[0]
NB_STATE_VAR = env.state_N
DISCOUNT_FACTOR = env.gamma
NB_STEPS_TRAJ = 96 * 50  # Number of timesteps for generating trajectories : 96 = 1 day, 96*2 = 2 days
LOW_BOUNDS = env.action_space.low
UP_BOUNDS = env.action_space.high
NB_DISCRETE_STEPS = 500

T = 3000  # Number of maximum timestep per episode
N_EVAL = 5  # Number of episodes


class RF_discrete(nn.Module):
    """REINFORCE network with discretized action space"""

    def __init__(self):
        super(RF_discrete, self).__init__()
        self.lin1 = nn.Linear(NB_STATE_VAR, 512)
        self.lin2 = nn.Linear(512, NB_ACTION_VAR * NB_DISCRETE_STEPS)

    def forward(self, input):
        x = input
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        x = x.reshape(NB_ACTION_VAR, NB_DISCRETE_STEPS)
        output = F.softmax(x, dim=-1)
        return output


class RF_continuous(nn.Module):
    """REINFORCE network with continuous action space"""

    def __init__(self):
        super(RF_continuous, self).__init__()
        self.lin1 = nn.Linear(NB_STATE_VAR, 512)
        self.lin2 = nn.Linear(512, 512)
        self.lin3 = nn.Linear(512, NB_ACTION_VAR)

    def forward(self, input):
        x = input
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x


class Actor_Critic_NN(nn.Module):
    """PPO network"""

    def __init__(self, in_features, out_features):
        super(Actor_Critic_NN, self).__init__()

        self.in_layer = nn.Linear(in_features, 64)
        self.hid_layer = nn.Linear(64, 64)
        self.out_layer = nn.Linear(64, out_features)

    def forward(self, x):
        x = F.tanh(self.in_layer(x))
        x = F.tanh(self.hid_layer(x))
        x = self.out_layer(x)

        return x.to('cpu')


if __name__ == "__main__":

    # REINFORCE continuous -- Plot expected return over 3000 timesteps
    # Data generation
    RF_cont = RF_continuous().to(device)
    RF_cont.load_state_dict(torch.load('./weights/RF_cont/RF_cont_base64_final.pt', map_location=torch.device('cpu')))
    episode_rewards, episode_discounted_rewards = [], []
    returns = []

    for i in range(N_EVAL):
        s = torch.tensor(env.reset(), dtype=torch.float).to(device)
        done = False
        episode_reward = 0.0
        episode_discounted_reward = 0.0
        episode_length = 0
        episode_discounted_rewards = np.full(T, np.nan)

        while not done and episode_length < T:
            a = RF_cont(s)

            for k in range(NB_ACTION_VAR):
                a[k] = a[k].clamp(LOW_BOUNDS[k], UP_BOUNDS[k])

            next_s, reward, done, _ = env.step(a.tolist())
            episode_reward += reward
            episode_discounted_reward += reward * (env.gamma ** episode_length)
            episode_discounted_rewards[episode_length] = episode_discounted_reward
            s = torch.tensor(next_s, dtype=torch.float).to(device)
            episode_length += 1

        returns.append(episode_discounted_rewards)
        episode_rewards.append(episode_reward)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_discounted_reward = np.mean(episode_discounted_rewards)
    std_discounted_reward = np.std(episode_discounted_rewards)

    # Preparation of the plots

    mean = np.nanmean(returns, axis=0)
    std = np.nanstd(returns, axis=0)

    plt.figure(figsize=(9, 6))
    plt.plot(mean)
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('$J^{\mu}_{3000}$', fontsize=12)
    plt.fill_between(
        range(len(mean)),
        mean - std,
        mean + std,
        alpha=0.5
    )

    # --------------------------------------------------------------------------------
    # REINFORCE discretized - Plot expected return (T=3000) of final model

    # Discretize action space
    # Data generation
    action_possibles = np.zeros((NB_ACTION_VAR, NB_DISCRETE_STEPS))
    for i in range(NB_ACTION_VAR):
        bound_inf, bound_sup = env.action_space.low[i], env.action_space.high[i]
        action_possibles[i] = np.linspace(start=bound_inf, stop=bound_sup, num=NB_DISCRETE_STEPS)

    RF_dis = RF_discrete().to(device)
    RF_dis.load_state_dict(torch.load('./weights/RF_discrete/RF_discrete_final.pt', map_location=torch.device('cpu')))
    episode_rewards, episode_discounted_rewards = [], []
    returns = []

    for i in range(N_EVAL):
        s = torch.tensor(env.reset(), dtype=torch.float).to(device)
        done = False
        episode_reward = 0.0
        episode_discounted_reward = 0.0
        episode_length = 0
        episode_discounted_rewards = np.full(T, np.nan)

        while not done and episode_length < T:
            proba = RF_dis(s)
            a = []

            for j in range(NB_ACTION_VAR):
                ind_a = np.argmax(proba[j].detach().numpy())
                a.append(action_possibles[j][ind_a])

            next_s, reward, done, _ = env.step(a)
            episode_reward += reward
            episode_discounted_reward += reward * (env.gamma ** episode_length)
            episode_discounted_rewards[episode_length] = episode_discounted_reward
            s = torch.tensor(next_s, dtype=torch.float).to(device)
            episode_length += 1

        returns.append(episode_discounted_rewards)
        episode_rewards.append(episode_reward)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_discounted_reward = np.mean(episode_discounted_rewards)
    std_discounted_reward = np.std(episode_discounted_rewards)

    # Preparation of the plots
    mean = np.nanmean(returns, axis=0)
    std = np.nanstd(returns, axis=0)

    plt.figure(figsize=(9, 6))
    plt.plot(mean)
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('$J^{\mu}_{3000}$', fontsize=12)
    plt.fill_between(
        range(len(mean)),
        mean - std,
        mean + std,
        alpha=0.5
    )

    # --------------------------------------------------------------------------------------
    # PPO - Plot expect return (T = 3000) of final model
    # Data generation
    PPO = Actor_Critic_NN(NB_STATE_VAR, NB_ACTION_VAR).to(device)
    PPO.load_state_dict(torch.load('./weights/PPO/tanh_weights_actor_64.pt', map_location=torch.device('cpu')))
    episode_rewards, episode_discounted_rewards = [], []
    returns = []

    for i in range(N_EVAL):
        s = torch.tensor(env.reset(), dtype=torch.float).to(device)
        done = False
        episode_reward = 0.0
        episode_discounted_reward = 0.0
        episode_length = 0
        episode_discounted_rewards = np.full(T, np.nan)

        while not done and episode_length < T:
            a = PPO(s)

            for k in range(NB_ACTION_VAR):
                a[k] = a[k].clamp(LOW_BOUNDS[k], UP_BOUNDS[k])

            next_s, reward, done, _ = env.step(a.tolist())
            episode_reward += reward
            episode_discounted_reward += reward * (env.gamma ** episode_length)
            episode_discounted_rewards[episode_length] = episode_discounted_reward
            s = torch.tensor(next_s, dtype=torch.float).to(device)
            episode_length += 1

        returns.append(episode_discounted_rewards)
        episode_rewards.append(episode_reward)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_discounted_reward = np.mean(episode_discounted_rewards)
    std_discounted_reward = np.std(episode_discounted_rewards)

    # Preparation of the plot
    mean = np.nanmean(returns, axis=0)
    std = np.nanstd(returns, axis=0)

    plt.figure(figsize=(9, 6))
    plt.plot(mean)
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('$J^{\mu}_{3000}$', fontsize=12)
    plt.fill_between(
        range(len(mean)),
        mean - std,
        mean + std,
        alpha=0.5
    )

    # -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------
    # STATISTICS OVER THE TRAINING
    # -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------

    # REINFORCE continuous
    # Data generation
    weights = sorted(os.listdir('./weights/RF_cont'))
    mean_J_RF_cont = []
    std_J_RF_cont = []

    for i in range(len(weights)):
        RF_cont = RF_continuous().to(device)
        path = './weights/RF_cont/' + weights[i]
        RF_cont.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

        J = []
        for j in range(N_EVAL):

            s = torch.tensor(env.reset(), dtype=torch.float).to(device)
            done = False
            episode_discounted_reward = 0.0
            episode_length = 0

            while not done and episode_length < T:
                a = RF_cont(s)

                for k in range(NB_ACTION_VAR):
                    a[k] = a[k].clamp(LOW_BOUNDS[k], UP_BOUNDS[k])

                next_s, reward, done, _ = env.step(a.tolist())
                episode_discounted_reward += reward * (env.gamma ** episode_length)
                s = torch.tensor(next_s, dtype=torch.float).to(device)
                episode_length += 1

            J.append(episode_discounted_reward)

        mean_J_RF_cont.append(np.mean(J))
        std_J_RF_cont.append(np.std(J))

    # Save results
    sample_list = [mean_J_RF_cont, std_J_RF_cont]
    file_name = "train_J_cont.pkl"

    open_file = open(file_name, "wb")
    pickle.dump(sample_list, open_file)
    open_file.close()

    # Preparation of the plot
    # After will make only one plot with RF discrete, RF cont and PPO
    mean_J_RF_cont = np.array(mean_J_RF_cont)
    std_J_RF_cont = np.array(std_J_RF_cont)

    plt.figure(figsize=(9, 6))
    N_eval_RF_cont = 96 * 50  # evaluate training each 4800 timestep
    plt.plot(np.arange(0, N_eval_RF_cont * 101, N_eval_RF_cont * 10), mean_J_RF_cont)
    plt.xlabel('Timestep of training', fontsize=12)
    plt.ylabel('Discounted Reward (T=3000)', fontsize=12)
    plt.fill_between(
        np.arange(0, N_eval_RF_cont * 101, N_eval_RF_cont * 10),
        mean_J_RF_cont - std_J_RF_cont,
        mean_J_RF_cont + std_J_RF_cont,
        alpha=0.3
    )

    # -----------------------------------------------------------------------------------
    # REINFORCE discrete
    # Data generation
    action_possibles = np.zeros((NB_ACTION_VAR, NB_DISCRETE_STEPS))
    for i in range(NB_ACTION_VAR):
        bound_inf, bound_sup = env.action_space.low[i], env.action_space.high[i]
        action_possibles[i] = np.linspace(start=bound_inf, stop=bound_sup, num=NB_DISCRETE_STEPS)

    weights = sorted(os.listdir('./weights/RF_discrete'))
    mean_J_RF_dis = []
    std_J_RF_dis = []

    for i in range(len(weights)):
        RF_dis = RF_discrete().to(device)
        path = './weights/RF_discrete/' + weights[i]
        RF_dis.load_state_dict(torch.load(path))

        J = []
        for j in range(N_EVAL):

            s = torch.tensor(env.reset(), dtype=torch.float).to(device)
            done = False
            episode_discounted_reward = 0.0
            episode_length = 0

            while not done and episode_length < T:
                proba = RF_dis(s)
                a = []

                for k in range(NB_ACTION_VAR):
                    ind_a = np.argmax(proba[k].cpu().detach().numpy())
                    a.append(action_possibles[k][ind_a])

                next_s, reward, done, _ = env.step(a)
                episode_discounted_reward += reward * (env.gamma ** episode_length)
                s = torch.tensor(next_s, dtype=torch.float).to(device)
                episode_length += 1

            J.append(episode_discounted_reward)

        mean_J_RF_dis.append(np.mean(J))
        std_J_RF_dis.append(np.std(J))

    # Save results
    sample_list = [mean_J_RF_dis, std_J_RF_dis]
    file_name = "train_J_dis.pkl"

    open_file = open(file_name, "wb")
    pickle.dump(sample_list, open_file)
    open_file.close()

    # Preparation of the plot
    mean_J_RF_dis = np.array(mean_J_RF_dis)
    std_J_RF_dis = np.array(std_J_RF_dis)

    plt.figure(figsize=(9, 6))
    N_eval_RF_dis = 96 * 50  # evaluate training each 4800 timestep
    plt.plot(np.arange(0, N_eval_RF_dis * 100, N_eval_RF_dis * 10), mean_J_RF_dis)
    plt.xlabel('Timestep of training', fontsize=12)
    plt.ylabel('Discounted Reward (T=3000)', fontsize=12)
    plt.fill_between(
        np.arange(0, N_eval_RF_dis * 100, N_eval_RF_dis * 10),
        mean_J_RF_dis - std_J_RF_dis,
        mean_J_RF_dis + std_J_RF_dis,
        alpha=0.3
    )

    # -------------------------------------------------------------------------------------------
    # PPO
    # -------------------------------------------------------------------------------------------
    # Generation of the data
    weights = sorted(os.listdir('./weights/PPO'))
    mean_J_PPO = []
    std_J_PPO = []

    for i in range(len(weights)):
        PPO = Actor_Critic_NN(NB_STATE_VAR, NB_ACTION_VAR).to(device)
        path = './weights/PPO/' + weights[i]
        PPO.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

        J = []
        for j in range(N_EVAL):

            s = torch.tensor(env.reset(), dtype=torch.float).to(device)
            done = False
            episode_discounted_reward = 0.0
            episode_length = 0

            while not done and episode_length < T:
                a = PPO(s)

                for k in range(NB_ACTION_VAR):
                    a[k] = a[k].clamp(LOW_BOUNDS[k], UP_BOUNDS[k])

                next_s, reward, done, _ = env.step(a.tolist())
                episode_discounted_reward += reward * (env.gamma ** episode_length)
                s = torch.tensor(next_s, dtype=torch.float).to(device)
                episode_length += 1

            J.append(episode_discounted_reward)

        mean_J_PPO.append(np.mean(J))
        std_J_PPO.append(np.std(J))

    # Save results
    sample_list = [mean_J_PPO, std_J_PPO]
    file_name = "train_J_PPO.pkl"

    open_file = open(file_name, "wb")
    pickle.dump(sample_list, open_file)
    open_file.close()

    # Preparation of the plot
    mean_J_PPO = np.array(mean_J_PPO)
    std_J_PPO = np.array(std_J_PPO)
    plt.figure(figsize=(9, 6))
    N_eval_PPO = 96 * 50 * 3  # evaluate training each 4800*3 timestep
    plt.plot(np.arange(0, N_eval_PPO * 35, N_eval_PPO * 3.5), mean_J_PPO)
    plt.xlabel('Timestep of training', fontsize=12)
    plt.ylabel('Discounted Reward (T=3000)', fontsize=12)
    plt.fill_between(
        np.arange(0, N_eval_PPO * 35, N_eval_PPO * 3.5),
        mean_J_PPO - std_J_PPO,
        mean_J_PPO + std_J_PPO,
        alpha=0.3
    )

    # -----------------------------------------------------------------------------------------
    # Plot discount reward (T=3000) for REINFORCE continuous, discretized and PPO over training
    open_file = open("./train_J_cont.pkl", "rb")
    stat_RF_cont = pickle.load(open_file)
    open_file.close()

    open_file = open("./train_J_dis.pkl", "rb")
    stat_RF_dis = pickle.load(open_file)
    open_file.close()

    open_file = open("./train_J_PPO.pkl", "rb")
    stat_PPO = pickle.load(open_file)
    open_file.close()
    stat_PPO[0] = stat_PPO[0][:9]
    stat_PPO[1] = stat_PPO[1][:9]

    for stat in [stat_RF_cont, stat_RF_dis, stat_PPO]:
        stat[0] = np.array(stat[0])
        stat[1] = np.array(stat[1])

    plt.figure(figsize=(9, 6))

    N_eval_PPO = 96 * 50 * 3  # evaluate training each 4800*3 timestep
    plt.plot(np.arange(0, N_eval_PPO * 35, N_eval_PPO * 4), stat_PPO[0], label='PPO')
    plt.xlabel('Timestep of training', fontsize=12)
    plt.ylabel('Discounted Return (T=3000)', fontsize=12)
    plt.fill_between(
        np.arange(0, N_eval_PPO * 35, N_eval_PPO * 4),
        stat_PPO[0] - stat_PPO[1],
        stat_PPO[0] + stat_PPO[1],
        alpha=0.3
    )

    N_eval_RF = 96 * 50  # evaluate training each 4800 timestep
    plt.plot(np.arange(0, N_eval_RF * 101, N_eval_RF * 10), stat_RF_cont[0], label='RF continuous')
    plt.xlabel('Timestep of training', fontsize=12)
    plt.ylabel('Discounted Return (T=3000)', fontsize=12)
    plt.fill_between(
        np.arange(0, N_eval_RF * 101, N_eval_RF * 10),
        stat_RF_cont[0] - stat_RF_cont[1],
        stat_RF_cont[0] + stat_RF_cont[1],
        alpha=0.3
    )

    plt.plot(np.arange(0, N_eval_RF * 101, N_eval_RF * 10), stat_RF_dis[0], label='RF discrete')
    plt.xlabel('Timestep of training', fontsize=12)
    plt.ylabel('Discounted Return (T=3000)', fontsize=12)
    plt.fill_between(
        np.arange(0, N_eval_RF * 101, N_eval_RF * 10),
        stat_RF_dis[0] - stat_RF_dis[1],
        stat_RF_dis[0] + stat_RF_dis[1],
        alpha=0.3
    )
    plt.legend(fontsize=12, loc='right')

    # Plot only for PPO and RF continuous
    plt.figure(figsize=(9, 6))
    N_eval_PPO = 96 * 50 * 3  # evaluate training each 4800*3 timestep
    plt.plot(np.arange(0, N_eval_PPO * 35, N_eval_PPO * 4), stat_PPO[0], label='PPO')
    plt.xlabel('Timestep of training', fontsize=12)
    plt.ylabel('Discounted Return (T=3000)', fontsize=12)
    plt.fill_between(
        np.arange(0, N_eval_PPO * 35, N_eval_PPO * 4),
        stat_PPO[0] - stat_PPO[1],
        stat_PPO[0] + stat_PPO[1],
        alpha=0.3
    )

    N_eval_RF = 96 * 50  # evaluate training each 4800 timestep
    plt.plot(np.arange(0, N_eval_RF * 101, N_eval_RF * 10), stat_RF_cont[0], label='RF continuous')
    plt.xlabel('Timestep of training', fontsize=12)
    plt.ylabel('Discounted Return (T=3000)', fontsize=12)
    plt.fill_between(
        np.arange(0, N_eval_RF * 101, N_eval_RF * 10),
        stat_RF_cont[0] - stat_RF_cont[1],
        stat_RF_cont[0] + stat_RF_cont[1],
        alpha=0.3
    )
    plt.legend(fontsize=12, loc='right')


