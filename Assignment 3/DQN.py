"""
    University of Liege
    INFO8003-1 - Optimal decision making for complex problems
    Assignment 3 - Deep Reinforcement Learning with Images for the Car on the Hill Problem
    Authors :
        DELCOUR Florian
        MAKEDONSKY Aliocha
"""

from dql import *


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
    plt.savefig("Q_roight_DQN.png")
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
    plt.savefig("Q_Left_DQN.png")
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
    plt.savefig("Estimated_policy_DQN.png")
    plt.close()


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    neural_net = DQN().to(device)
    target_net = DQN().to(device)
    criterion = nn.MSELoss()
    replay_memory = ReplayMemory()
    optimizer = torch.optim.Adam(neural_net.parameters(), lr=0.001)

    nb_episode = 100
    nb_epoch = 25
    epsilon = np.linspace(0.05, 0.9, num=nb_episode)
    epsilon = np.flip(epsilon)
    update_period = 20
    epoch_th = 0

    with tqdm(total=nb_episode * nb_epoch) as tq:
        for i in range(nb_episode):
            for j in range(nb_epoch):
                epoch_th += 1
                if epoch_th == update_period:
                    target_net.load_state_dict(neural_net.state_dict())
                    epoch_th = 0
                terminal_state = False
                p = random.uniform(-0.1, 0.1)
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
                    if j == 0 and terminal_state == True:
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
                image = x_to_image((p, s)).to(device)
                Q.append(neural_net(image.unsqueeze(0)).cpu())

        Q = torch.cat(Q)
        Q = Q.view(len(p_vec), len(s_vec), len(action_space)).numpy()

    heatmaps(Q)

    J_est = expected_return(150, neural_net)
    print("J_est : ", J_est)
