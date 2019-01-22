import gym
import gym_gazebo
import numpy as np
import tensorflow as tf
from ddpg import *

EPISODES = 100000
MAX_EP_STEPS = 300
state_dim = 12
action_dim = 1

def main():
    env = gym.make('train_world-v2')
    agent = DDPG(state_dim, action_dim, env)
    print('Training mode')
    for episode in range(EPISODES):
        state = env.reset()
        one_round_step, total_reward = 0, 0

        for step in range(MAX_EP_STEPS):
            a = agent.noise_action(state)[0]
            state_, r, done, arrive = env.step(a)
            time_step = agent.perceive(state, a, r, state_, done)
            state = state_
            one_round_step += 1
            total_reward += r

            if arrive:
                result = 'Success'
            else:
                result = 'Fail'

            if done or step == MAX_EP_STEPS - 1:
                print('Ep:', episode, '| Step: %3i' % one_round_step, '| Time step: %i' % time_step, '| Avg reward: % 0.2f' % (total_reward/one_round_step), '| ', result)
                break

def test():
    env = gym.make('train_world-v2')
    agent = DDPG(state_dim, action_dim, env)
    print('Testing mode')
    for episode in range(EPISODES):
        state = env.reset()
        one_round_step = 0

        for step in range(150):
            a = agent.action(state)[0]
            state_, _, done, arrive = env.step(a)
            state = state_
            one_round_step += 1

            if done:
                break

if __name__ == '__main__':
    Train = False
    if Train:
        main()
    else:
        test()
