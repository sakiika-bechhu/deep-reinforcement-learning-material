from typing import List
import gym
import numpy as np
from torch import nn, optim, FloatTensor, LongTensor


class Episode:
    total_reward: float
    actions: List
    observations: List

    def __init__(self):
        self.total_reward = 0.0
        self.actions = []
        self.observations = []

    def append(self, reward, action, observation):
        self.total_reward += reward
        self.actions.append(action)
        self.observations.append(observation)


class Net(nn.Module):
    def __init__(self, observation_size, hidden_size, action_size):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, action_size),
        )
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        return self.net(x)

    def action_prob(self, x):
        return self.sm(self.net(x)).detach().numpy()[0]


def generate_dataset(env, net, batch_size):
    batch = []
    obs = env.reset()
    episode = Episode()
    while True:
        net_input = FloatTensor([obs])
        # action_probability = net(net_input)
        action = np.random.choice(2, p=net.action_prob(net_input))
        next_obs, reward, endgame, _ = env.step(action)
        episode.append(reward, action=action, observation=obs)
        if endgame:
            batch.append(episode)
            episode = Episode()
            next_obs = env.reset()
            if len(batch) == batch_size:
                return batch
        obs = next_obs


def select_elite(batch, top_percent):
    rewards = [x.total_reward for x in batch]
    reward_threshold = np.percentile(rewards, top_percent)
    reward_mean = np.mean(rewards)
    train_observation = []
    train_action = []
    for episode in batch:
        if episode.total_reward < reward_threshold:
            continue
        for each_act, each_obs in zip(episode.actions, episode.observations):
            train_observation.append(each_obs)
            train_action.append(each_act)
    train_observation = FloatTensor(train_observation)
    train_action = LongTensor(train_action)
    return train_observation, train_action, reward_threshold, reward_mean
BATCH_SIZE = 20


def main():
    env = gym.make("CartPole-v0")
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    net = Net(obs_size, 128, n_actions)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    batch_idx = 0
    while True:
        batch = generate_dataset(env, net, BATCH_SIZE)
        train_observation, train_action, reward_bound, reward_mean = select_elite(batch, 70)
        optimizer.zero_grad()
        action_scores = net(train_observation)
        _loss = loss(action_scores, train_action)
        _loss.backward()
        optimizer.step()
        print(f"{batch_idx}: loss={_loss.item()}, reward_mean={reward_mean}, reward_bound={reward_bound}")
        batch_idx += 1


main()
