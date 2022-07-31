import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from gym_anm.envs import ANM6Easy
import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device : ", device)

env = ANM6Easy()
NB_ACTION_VAR = env.action_space.shape[0]
NB_STATE_VAR = env.state_N
DISCOUNT_FACTOR = env.gamma
NB_STEPS_TRAJ = 96 * 50 # Number of timesteps for generating trajectories : 96 = 1 day, 96*2 = 2 days
LOW_BOUNDS = env.action_space.low
UP_BOUNDS = env.action_space.high

# Number of steps to discretize the action space
NB_DISCRETE_STEPS = 500

action_possibles = np.zeros((NB_ACTION_VAR,NB_DISCRETE_STEPS))
for i in range(NB_ACTION_VAR):
    bound_inf, bound_sup = env.action_space.low[i], env.action_space.high[i]
    action_possibles[i] = np.linspace(start=bound_inf, stop=bound_sup, num=NB_DISCRETE_STEPS)

class RF_cont(nn.Module):
    """REINFORCE algorithm with continuous action space"""

    def __init__(self):
        super(RF_cont, self).__init__()
        self.lin1 = nn.Linear(NB_STATE_VAR, 512)
        self.lin2 = nn.Linear(512, 512)
        self.lin3 = nn.Linear(512, NB_ACTION_VAR)

    def forward(self, input):
        x = input
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x


def generate_traj(initial_state, model, env, sigma):
    """Collect trajectory : state, action, reward"""

    states, actions, rewards = [], [], []
    s = torch.tensor(initial_state, dtype=torch.float).to(device)
    done = False
    nb_step = 0

    while not done and nb_step < NB_STEPS_TRAJ:
        mu_a = model(s)
        a = torch.distributions.MultivariateNormal(mu_a.type(torch.DoubleTensor).to(device), torch.diag(sigma)).sample()

        penalized = 0
        for i in range(NB_ACTION_VAR):
            if a[i] < LOW_BOUNDS[i] or a[i] > UP_BOUNDS[i]:
                penalized += 1
            a[i] = a[i].clamp(LOW_BOUNDS[i], UP_BOUNDS[i])

        next_s, r, done, _ = env.step(a.tolist())

        # Penalized reward when action variables are out of boundaries
        if penalized != 0:
            r = -100. * penalized
        actions.append(a.tolist())
        rewards.append(r)
        states.append(s.tolist())

        s = torch.tensor(next_s, dtype=torch.float).to(device)
        nb_step += 1

    return states, actions, rewards


def train_cont(env, model, n_epochs):
    """Training REINFORCE continuous action space"""

    model = model.to(device)
    log_sigma = torch.ones(NB_ACTION_VAR, dtype=torch.double).to(device).requires_grad_()
    optimizer = optim.Adam(list(model.parameters()) + [log_sigma], lr=0.001)
    losses = []

    for ep in tqdm(range(n_epochs)):
        states, actions, rewards = generate_traj(initial_state=env.reset(), model=model, env=env,
                                                 sigma=torch.exp(log_sigma))

        cum = 0  # cumulative reward
        discounted_rewards = np.zeros(len(rewards))

        for t in reversed(range(len(rewards))):
            cum = cum * DISCOUNT_FACTOR + rewards[t]
            discounted_rewards[t] = cum

        states = torch.Tensor(states).to(device)
        actions = torch.Tensor(actions).to(device)
        returns = torch.Tensor(discounted_rewards).to(device)

        mu_a = model(states)
        pi = torch.distributions.MultivariateNormal(mu_a, torch.diag(torch.exp(log_sigma)))
        log_prob = pi.log_prob(actions)

        loss = torch.mean(-returns * log_prob)
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()

        if ep % 10 == 0:
            torch.save(model.state_dict(), f"RF_cont_{ep}.pt")

    torch.save(model.state_dict(), f"RF_cont_final.pt")

    return losses


def generate_traj_dis(initial_state, model, env):
    """Collect trajectory : state, action, reward"""

    states, actions, rewards = [], [], []
    s = torch.tensor(initial_state, dtype=torch.float).to(device)
    done = False
    nb_step = 0

    while not done and nb_step < NB_STEPS_TRAJ:
        a = []
        proba = model(s)

        for i in range(NB_ACTION_VAR):
            proba_dist = torch.distributions.Categorical(probs=proba[i])
            index_action = proba_dist.sample().item()
            a.append(action_possibles[i][index_action])

        next_s, r, done, _ = env.step(a)
        actions.append(a)
        rewards.append(r)
        states.append(s)

        s = torch.tensor(next_s, dtype=torch.float).to(device)
        nb_step += 1

    return states, actions, rewards


def train_dis(env, model, n_epochs):
    """Training REINFORCE discretized action space"""

    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    model = model.to(device)
    losses = []

    for ep in tqdm(range(n_epochs)):
        states, actions, rewards = generate_traj_dis(initial_state=env.reset(), model=model, env=env)

        cum = 0  # cumulative reward
        discounted_rewards = np.zeros(len(rewards))

        for t in reversed(range(len(rewards))):  # get discounted rewards
            cum = cum * DISCOUNT_FACTOR + rewards[t]
            discounted_rewards[t] = cum

        for i in range(len(actions)):
            s = states[i].to(device)
            proba = model(s)

            optimizer.zero_grad()
            cum_loss = 0

            for j in range(NB_ACTION_VAR):
                proba_dist = torch.distributions.Categorical(probs=proba[j])
                a = np.where(action_possibles[j] == actions[i][j])
                log_proba = proba_dist.log_prob(torch.tensor(a).to(device))
                cum_loss += -discounted_rewards[i] * log_proba

            loss = cum_loss / NB_ACTION_VAR

            if abs(loss.item()) > 1e-3:
                loss.backward()
                losses.append(loss.item())

            optimizer.step()

        if ep % 10 == 0:
            torch.save(model.state_dict(), f"RF_discrete_{ep}.pt")

    torch.save(model.state_dict(), f"RF_discrete_final.pt")

    return losses


class RF_discrete(nn.Module):
    """REINFORCE algorithm with discretized action space"""

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


if __name__ == "__main__":
    """
    RF_cont_NN = RF_cont()
    losses = train_cont(env=env, model=RF_cont_NN, n_epochs=100)

    plt.figure(figsize=(9, 6))
    plt.plot(losses)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss RF continuous', fontsize=12)
    """

    RF_dis = RF_discrete()
    losses = train_dis(env=env, model=RF_dis, n_epochs=100)

    plt.figure(figsize=(9, 6))
    plt.plot(losses)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss RF discrete', fontsize=12)

