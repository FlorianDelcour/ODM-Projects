"""
    University of Liege
    INFO8003-1 - Optimal decision making for complex problems
    Assignment 3 - Deep Reinforcement Learning with Images for the Car on the Hill Problem
    Authors : 
        DELCOUR Florian
        MAKEDONSKY Aliocha
"""

from torch.autograd import Variable
from car_on_the_hill_images import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor
from domain import *
from tqdm import tqdm
import time

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def opt_policy_est(p, s, model):
    """
        This function builds a sample of self.batch_size length from the replay memory.

        Parameters
        ----------
        p : tuple of floats
            Position of our agent
        s : float
            Velocity of our agent
        model : neural network
            Trained neural network

        Returns
        -------
        action_space[i] : int
            Optimal action to take
    """
    q = model(x_to_image((p, s)).to(device).unsqueeze(0))
    q_0 = q.detach()[0][0].item()
    q_1 = q.detach()[0][1].item()
    return action_space[0] if q_0 > q_1 else action_space[1]


def expected_return(N, model):
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
        u = opt_policy_est(p, s, model)
        reward = reward_signal(p, s, u)
        next_p, next_s = dynamics(p, s, u)
        if next_p == p and next_s == s:  # terminal state
            total_J_N[t + 1] = total_J_N[t]
            continue
        J_N += (gamma ** t) * reward
        p = next_p
        s = next_s
        total_J_N[t + 1] = J_N
    return total_J_N


def x_to_image(x):
    """
        This function transforms a state into an image then into a tensor.

        Parameters
        ----------
        x : tuple
            State of the game, so position and velocity of the agent.

        Returns
        -------
        img : tensor
            Tensor representing the RGB image.

    """
    p, s = x
    img = save_caronthehill_image(position=p, speed=s, out_file=None).copy()
    img = to_tensor(img)
    img = F.avg_pool2d(img, 4)

    return img


class ReplayMemory:

    def __init__(self, capacity=1024, batch_size=64):
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = [None]*capacity
        self.index = 0
        self.memory_len = 0

    def long_enough(self):
        """
        This function indicates if the replay memory has enough transitions to constitute a sample.

        Returns
        -------
        self.memory_len >= self.batch_size : boolean
            True if the length of the replay memory is long enough.
        """
        return self.memory_len >= self.batch_size

    def add_transition(self, transition):
        """
            This function adds a transition in the replay memory.

            Parameters
            -------
            transition : 4-tuple
                A 4-tuple composed of (state, u, r, state') with state the state of the game, represented by an image
                itself represented by a tensor, u the action taken, r the reward obtained, and state' the next state,
                as an image, and so by a tensor too.
        """
        self.memory[self.index] = transition
        self.index = (self.index + 1) % self.capacity
        if self.memory_len < self.capacity:
            self.memory_len += 1

    def sample(self):
        """
            This function builds a sample of self.batch_size length from the replay memory.

            Returns
            -------
            x, u, r, y : 4-tuple
                Sample of size self.batch_size, containing self.batch_size transitions
        """
        if self.memory_len < self.capacity:
            choices = random.sample(self.memory[:self.memory_len], self.batch_size)
        else:
            choices = random.sample(self.memory, self.batch_size)
        x, u, r, y = tuple(zip(*choices))

        x = torch.stack(x)
        y = torch.stack(y)
        u = torch.tensor(u)
        r = torch.tensor(r)

        return x, u, r, y


class DQN(nn.Module):
    def __init__(self):
        super().__init__()

        self.CNN = CNN(3, 16)
        self.MLP = MLP(1520768, 2)

    def forward(self, x):
        x = self.CNN(x)
        x = self.MLP(x)

        return x


class CNN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=1, padding=2):
        layers = []
        out = out_channels
        for M in range(3):
            out *= 2
            for N in range(2):
                layers.append(nn.Conv2d(in_channels, out, kernel_size, stride, padding))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out
            layers.append(nn.AvgPool2d(kernel_size, stride, padding))
            print(out)
        layers.append(nn.Flatten())

        super().__init__(*layers)


class MLP(nn.Sequential):
    def __init__(self, input_size, output_size, hidden_layer_size=10, hidden_layer_number=4):
        layers = []
        layers.append(nn.Linear(input_size, hidden_layer_size))
        layers.append(nn.ReLU(inplace=True))

        for i in range(hidden_layer_number):
            layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(hidden_layer_size, output_size))

        super().__init__(*layers)


def train_network():
    """
            This function trains our neural network.
    """
    x, action, reward, next_x = replay_memory.sample()
    x = x.to(device)
    action = action.to(device)
    reward = reward.to(device)
    next_x = next_x.to(device)

    with torch.no_grad():
        Q = target_net(next_x)
        max_Q = Q.max(1)[0].to(device)
        reward = reward.type(torch.FloatTensor).to(device)
        y = torch.where(reward != 0, reward, gamma * max_Q)

    action_adapted = []
    for act in action.view(-1, 1):
        action_adapted.append(0 if act == -4 else 1)
    action_adapted = torch.LongTensor(action_adapted)
    output = neural_net(x)
    Q = []
    for i in range(len(action)):
        Q.append(output[i][action_adapted[i]])
    Q = torch.FloatTensor(Q).to(device)

    loss = criterion(Q, y).to(device)
    loss = Variable(loss, requires_grad=True)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(neural_net.parameters(), 1.)
    optimizer.step()


def heatmaps(Q_map):
    """
        This function generates Q_N plots for each action and the derived
        optimal policy plot.

        Parameters
        ----------
        Q_map : 2 matrices of Q
            The estimated Q_N function
    """

    color_map1 = Q_map[:, :, 0].T
    color_map2 = Q_map[:, :, 1].T
    min_v = np.min(color_map1)
    max_v = np.max(color_map1)
    fig1, ax1 = plt.subplots()
    X, Y = np.meshgrid(p_vec, s_vec)
    c = ax1.contourf(X, Y, color_map1, 10, cmap='RdBu', vmax=max_v, vmin=min_v)
    fig1.colorbar(c)
    ax1.set_title('$\hat{Q}$ for u = 4')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Speed')
    ax1.set_xlim([-1., 1.])
    ax1.set_ylim([-3, 3])
    plt.savefig("Q_roight.png")
    plt.close()

    min_v = np.min(color_map2)
    max_v = np.max(color_map2)
    fig2, ax2 = plt.subplots()
    c = ax2.contourf(X, Y, color_map2, 10, cmap='RdBu', vmax=max_v, vmin=min_v)
    fig2.colorbar(c)
    ax2.set_title('$\hat{Q}$ for u = -4')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Speed')
    ax2.set_xlim([-1., 1.])
    ax2.set_ylim([-3, 3])
    plt.savefig("Q_Left.png")
    plt.close()

    opt_policy = np.zeros([len(p_vec), len(s_vec)])

    for p in range(len(p_vec)):
        for s in range(len(s_vec)):
            ind = np.argmax(Q_map[p, s])
            opt_policy[p, s] = action_space[ind]

    fig3, ax3 = plt.subplots()
    c = ax3.contourf(X, Y, opt_policy.T, cmap='RdBu', vmax=4, vmin=-4)
    ax3.set_title('Policy (blue = 4) | (red = -4)')
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Speed')
    ax3.set_xlim([-1., 1.])
    ax3.set_ylim([-3, 3])
    plt.savefig("Estimated_policy.png")
    plt.close()


if __name__ == "__main__":
    print(torch.cuda.is_available())
    neural_net = DQN().to(device)
    target_net = DQN().to(device)
    criterion = nn.MSELoss()
    replay_memory = ReplayMemory()
    optimizer = torch.optim.SGD(neural_net.parameters(), lr=0.001)

    nb_episode = 100
    nb_epoch = 25
    epsilon = np.linspace(0.05, 0.9, num=nb_episode)
    epsilon = np.flip(epsilon)

    with tqdm(total=nb_episode * nb_epoch) as tq:
        for i in range(nb_episode):
            targetnet = neural_net
            for j in range(nb_epoch):
                terminal_state = False
                p = random.uniform(-1., 1.)
                s = 0
                state = (p, s)
                image = x_to_image(state)

                while not terminal_state:
                    if random.uniform(0, 1) < epsilon[i]:
                        u = action_space[random.randint(0, 1)]
                    else:
                        u = opt_policy_est(p, s, neural_net)
                    next_p, next_s = dynamics(p, s, u)

                    if p == next_p and s == next_s:
                        terminal_state = True

                    next_state = (next_p, next_s)
                    next_image = x_to_image(next_state)
                    if j==0 and terminal_state == True:
                        r = 1 if p >= 1 else -1
                    else:
                        r = reward_signal(p, s, u)

                    replay_memory.add_transition((image, u, r, next_image))

                    p, s = next_p, next_s
                    image = next_image

                    if replay_memory.long_enough():
                        tic_train = time.perf_counter()
                        train_network()
                        toc_train = time.perf_counter()

                tq.update(1)

    p_vec = np.arange(-1, 1, 0.01)
    s_vec = np.arange(-3, 3, 0.01)

    with torch.no_grad():
        Q = []

        for p in p_vec:
            for s in s_vec:
                image = x_to_image((p,s)).to(device)
                Q.append(neural_net(image.unsqueeze(0)).cpu())

        Q = torch.cat(Q)
        Q = Q.view(len(p_vec), len(s_vec), len(action_space)).numpy()

    heatmaps(Q)

    J_est = expected_return(150, neural_net)
    print("J_est : ", J_est)
