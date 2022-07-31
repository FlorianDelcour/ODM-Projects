import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gym_anm.envs import ANM6Easy
from torch.distributions import MultivariateNormal
import torch.optim as optim
import warnings

warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device : ", device)

class Actor_Critic_NN(nn.Module):
    def __init__(self, in_features, out_features):
        """
        This function instantiate a neural network
        Parameters
        -----------
        in_features : int
            Number of features for the input layers

        out_features : int
            Number of features of the output layer
        """
        super(Actor_Critic_NN, self).__init__()

        self.in_layer = nn.Linear(in_features, 64)
        self.hid_layer = nn.Linear(64, 64)
        self.out_layer = nn.Linear(64, out_features)

    def forward(self, obs):
        obs = F.tanh(self.in_layer(obs))
        obs = F.tanh(self.hid_layer(obs))
        obs = self.out_layer(obs)

        return obs.to('cpu')


class PPO:
    def __init__(self):
        self.env = ANM6Easy()

        self.nb_actions = self.env.action_space.shape[0]
        self.nb_states = self.env.observation_space.shape[0]

        self.actor = Actor_Critic_NN(in_features=self.nb_states, out_features=self.nb_actions).to(device)
        self.critic = Actor_Critic_NN(in_features=self.nb_states, out_features=1).to(device)

        self.timesteps = 1000000
        self.actor_losses = []
        self.critic_losses = []

        self.cov_var = torch.full(size=(self.nb_actions,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def train(self):
        """This function trains our actor and critic networks from our PPO class."""
        nb_epoch = 5
        epsilon = 0.2
        learning_rate = 0.002
        actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        save = 0

        while self.timesteps > 0:
            print(self.timesteps)
            if save % 4 == 0:
                torch.save(self.actor.state_dict(), f"tanh_weights_actor_{save}.pt")
                torch.save(self.critic.state_dict(), f"tanh_weights_critic_{save}.pt")
            save += 1
            batch_obs, batch_actions, batch_log_probs, batch_r_to_go, batch_lens = self.gen_traj()

            self.timesteps -= np.sum(batch_lens)

            V, _ = self.v_function(batch_obs, batch_actions)

            A_k = batch_r_to_go - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)  # 1e-10 to avoid dividing by 0

            for _ in range(nb_epoch):
                V, curr_log_prbs = self.v_function(batch_obs, batch_actions)

                ratios = torch.exp(curr_log_prbs - batch_log_probs)

                loss1 = ratios * A_k
                loss2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * A_k

                actor_loss = (-torch.min(loss1, loss2)).mean()
                self.actor_losses.append(actor_loss)

                actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                actor_optimizer.step()

                critic_criterion = nn.MSELoss()
                critic_loss = critic_criterion(V, batch_r_to_go)
                self.critic_losses.append(critic_loss)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
            print('critic loss : ', critic_loss)
            print("actor loss : ", actor_loss)
        torch.save(self.actor.state_dict(), f"tanh_weights_actor_final.pt")
        torch.save(self.critic.state_dict(), f"tanh_weights_critic_final.pt")

    def gen_traj(self):
        """
        This function generate a trajectory with actions taken from the current actor network.

        Returns
        ----------
        batch_obs : list of all the observations collected from each episode
        batch_actions : list of all the actions taken during each episode
        batch_log_probs : list of all the log probabilities of the actions computed during each episode
        batch_r_to_go : list of all the rewards to go computed during each episode
        batch_lens : list of the length of each episode
        """
        timesteps_per_episode = 50 * 96
        timesteps_per_batch = 3 * timesteps_per_episode

        batch_obs = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_r_to_go = []
        batch_lens = []

        t = 0
        lower_bounds = self.env.action_space.low
        upper_bounds = self.env.action_space.high

        while t < timesteps_per_batch:
            obs = self.env.reset()

            rewards_of_episode = []
            for ep_t in range(timesteps_per_episode):
                t += 1
                batch_obs.append(obs)

                a, log_prob = self.policy(torch.tensor(obs, dtype=torch.float))

                for i in range(len(a)):
                    if a[i] < lower_bounds[i]:
                        a[i] = lower_bounds[i]
                    elif a[i] > upper_bounds[i]:
                        a[i] = upper_bounds[i]
                obs, reward, done, _ = self.env.step(a.tolist())
                rewards_of_episode.append(reward)
                batch_actions.append(a)
                batch_log_probs.append(log_prob)

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rewards.append(rewards_of_episode)

        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_actions = torch.tensor(batch_actions, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        batch_r_to_go = self.rewards_to_go(batch_rewards)

        return batch_obs, batch_actions, batch_log_probs, batch_r_to_go, batch_lens

    def policy(self, obs):
        """
        This functions computes the action to take given the observation in input

        Parameter
        ---------------
        obs : observation of the environment

        Returns
        ------------
        action : actions to take
        log_prob : log probabilities of the actions
        """
        mean = self.actor(obs.to(device))

        dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample()

        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()

    def rewards_to_go(self, batch_rewards):
        """
        Compute the rewards to go given the rewards given.

        Parameter
        -----------
        batch_rewards : list of rewards collected during the episodes

        Return
        ---------
        batch_r_to_go : list of the rewards to go computed
        """
        discount_factor = 0.995
        batch_r_to_go = []

        for rewards_of_episode in reversed(batch_rewards):
            discounted_reward = 0

            for reward in reversed(rewards_of_episode):
                discounted_reward = reward + discounted_reward * discount_factor
                batch_r_to_go.insert(0, discounted_reward)

        batch_r_to_go = torch.tensor(batch_r_to_go, dtype=torch.float)

        return batch_r_to_go

    def v_function(self, batch_obs, batch_actions):
        """
        Calls our critic network to compute the v function, and the actor network to compute the log probabilities of our actions

        Parameters
        ------------
        batch_obs : list of observations collected from each episode
        batch_actions : list of actions collected from each episode

        Returns
        ---------
        V : V function from the critic network
        log_probs : log probabilities of our actions computed by the actor network
        """
        V = self.critic(batch_obs.to(device)).squeeze()
        mean = self.actor(batch_obs.to(device))
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_actions)
        return V, log_probs


if __name__ == "__main__":
    model = PPO()
    model.train()
    act_loss = model.actor_losses
    crit_loss = model.critic_losses
