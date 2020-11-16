import gym
from gym.utils.play import play

env = gym.make('CartPole-v0')
keys_to_mapping = {
    (ord('a'), ): 0,
    (ord('s'), ): 1,
}
play(env, keys_to_action=keys_to_mapping)

# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample())
