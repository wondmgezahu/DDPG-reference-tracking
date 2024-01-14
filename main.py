import torch.nn as nn
import torch
import torch.nn.functional as F
import random
from collections import deque
import numpy as np
import gym
import torch.autograd
from torch.autograd import Variable
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import sys

seed = 1100

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

print('s')

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        # print('state before cat')
        # print(state, action)
        # print(state.shape)
        xxx=torch.reshape(state,(-1,2))
        # print(xxx)

        x=torch.cat([xxx,action],1)
        # print('xx',xx)
        # x = torch.cat([state, action], 1)
        # print(x)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=3e-4):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        """
        Param state is a torch tensor
        """
        # print(state.shape)
        x = F.relu(self.linear1(state.reshape(-1,2)))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x


class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)


class OUNoise(object):
    def __init__(self,  num_actions,mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = num_actions
        self.low = -1
        self.high = 4
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def _action(self, action):
        act_k = (self.action_space.high - self.action_space.low) / 2.
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k * action + act_b

    def _reverse_action(self, action):
        act_k_inv = 2. / (self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k_inv * (action - act_b)


class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)


class DDPGagent:
    def __init__(self, hidden_size=256, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2,
                 max_memory_size=50000):

        # def __init__(self, hidden_size=256, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2,
        #              max_memory_size=50000):
        #
        # Params
        self.num_states = 2
        self.num_actions = 1
        self.gamma = gamma
        self.tau = tau

        # Networks
        self.actor = Actor(self.num_states, hidden_size, self.num_actions)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Training
        self.memory = Memory(max_memory_size)
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    # torch.from_numpy(np.asarray(x))
    def get_action(self, state):
        # print(state)
        state = Variable(torch.from_numpy(np.asarray(state)).float().unsqueeze(0))
        action = self.actor.forward(state)
        # print("action")
        action = action.detach().numpy()[0]  # [0,0]
        return action

    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        # temp=np.array(states)
        temp=[]
        for i in states:
            j=i.reshape(-1,1)
            temp.append(j)

        states = torch.FloatTensor(temp)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # Critic loss
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()

        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def eval_func(self, x_0, eps_len, r=None,noise=None):

        state = np.array([x_0])
        state_seq = []
        action_seq = []
        for step in range(eps_len):
            observ = [float(state - r), float(state)]
            action = self.get_action(observ)
            if noise == 1:
                action += np.random.normal(0, 0.001, 1)
            elif noise == 2:
                action += np.random.normal(0, 0.01, 1)
            state_seq.append(state)
            action_seq.append(action)
            new_state, reward, done = step_func(state, action,r)

            state = new_state

        return state_seq, action_seq


def step_func(state, action,r=None):
    Done = False
    norm = float(np.linalg.norm(state-r))
    # reward =
    # reward=np.exp(reward)

    next_state = 1.2* state + action

    if float(norm)> 10 or float(state)<0:#10
        Done = True
        reward = -50#-50
    elif float(norm)<0.1 and float(state)>0:
        reward=100
    else:
        reward=-norm ** 2


    return next_state, reward, Done


if __name__ == '__main__':

    # env = gym.make('Pendulum-v1')
    agent = DDPGagent()
    noise=OUNoise(1)
    batch_size = 128  #
    rewards = []
    avg_rewards = []
    eps_len = 2000#2000
    eps_num =100#100

    for episode in range(eps_num):
        state = 0
        r = np.random.uniform(0,10,1)#10
        r=float(r)
        episode_reward = 0
        for step in range(eps_len):
            observ=[float(state-r),float(state)]
            observ=np.array(observ)
            action = agent.get_action(observ)
            action = noise.get_action(action, step)
            new_state, reward, done = step_func(state, action,r)
            agent.memory.push(observ, action, reward, [float(new_state-r),float(new_state)], done)

            if len(agent.memory) > batch_size:
                agent.update(batch_size)

            state = new_state
            episode_reward += reward

            if done or step == eps_len - 1:
                sys.stdout.write("episode: {}, reward: {}, avg_reward: {} \n".format(episode,
                                                                                     np.round(episode_reward,
                                                                                              decimals=2),
                                                                                     np.round(
                                                                                         np.mean(rewards[-10:]),
                                                                                         decimals=2)
                                                                                     ))
                break

        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-10:]))

    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()
