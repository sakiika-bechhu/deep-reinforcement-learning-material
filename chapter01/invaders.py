import gym
import time
env = gym.make('SpaceInvaders-v0')
env.reset()
for _ in range(1000):
    env.render()
    time.sleep(0.1)
    env.step(env.action_space.sample())
