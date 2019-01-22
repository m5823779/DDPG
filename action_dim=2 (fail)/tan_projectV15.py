import gym
import gym_gazebo
import numpy as np
import tensorflow as tf
from ddpg import *

EPISODES = 100000
MAX_EP_STEPS = 200
state_dim = 12
action_dim = 2

def main():
    env = gym.make('train_world-v3')
    agent = DDPG(env, state_dim, action_dim)
    print('Training mode')
    total_reward = 0
    epsilon = 2

    for episode in range(EPISODES):
        state = env.reset()
        one_round_step = 0

        for step in range(MAX_EP_STEPS):
            print(state)
            a = agent.noise_action(state, epsilon)
            state_, r, done, arrive = env.step(a[0], a[1])
            time_step = agent.perceive(state, a, r, state_, done)
            state = state_
            one_round_step += 1

            if arrive:
                result = 'Success'
            else:
                result = 'Fail'

            if time_step > 0:
                total_reward += r

            if time_step % 10 == 0 and time_step > 0:
                epsilon = max([epsilon*.9998, 0.1])

            if time_step % 10000 == 0 and time_step > 30000:
                print('---------------------------------------------------')
                print('Average_reward = ', total_reward / 10000)
                total_reward = 0

            if done or step == MAX_EP_STEPS - 1:
                print('Ep:', episode, '| Step: %3i' % one_round_step, '| Epsilon: %.2f' % epsilon,  '| Time step: %i' % time_step, '| ', result)
                break
def test():
    env = gym.make('train_world-v3')
    agent = DDPG(env, state_dim, action_dim)
    print('Testing mode')
    for episode in range(EPISODES):
        state = env.reset()
        one_round_step = 0

        for step in range(200):
            a = agent.action(state)
            state_, _, done, arrive = env.step(a[0], a[1])
            state = state_
            one_round_step += 1

            if done:
                break

if __name__ == '__main__':
    Train = True
    if Train:
        main()
    else:
        test()
