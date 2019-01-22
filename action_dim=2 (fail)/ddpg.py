import gym
import tensorflow as tf
import numpy as np
from ou_noise import OUNoise
from critic_network import CriticNetwork
from actor_network import ActorNetwork
from replay_buffer import ReplayBuffer

REPLAY_BUFFER_SIZE = 1000000
REPLAY_START_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99


class DDPG:
    def __init__(self, env, state_dim, action_dim):
        self.name = 'DDPG'
        self.environment = env
        self.time_step = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sess = tf.InteractiveSession()

        self.actor_network = ActorNetwork(self.sess, self.state_dim, self.action_dim)
        self.critic_network = CriticNetwork(self.sess, self.state_dim, self.action_dim)

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.linear_noise = OUNoise(1, 0.5, 0.3, 0.6)
        self.angular_noise = OUNoise(1, 0, 0.6, 0.8)

    def train(self):
        minibatch = self.replay_buffer.get_batch(BATCH_SIZE)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])
        # for action_dim = 1
        action_batch = np.resize(action_batch, [BATCH_SIZE, self.action_dim])

        next_action_batch = self.actor_network.target_actions(next_state_batch)
        q_value_batch = self.critic_network.target_q(next_state_batch, next_action_batch)
        y_batch = []
        for i in range(len(minibatch)):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])
        y_batch = np.resize(y_batch, [BATCH_SIZE, 1])
        # Update critic by minimizing the loss L
        self.critic_network.train(y_batch, state_batch, action_batch)

        # Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_network.actions(state_batch)
        q_gradient_batch = self.critic_network.gradients(state_batch, action_batch_for_gradients)

        self.actor_network.train(q_gradient_batch, state_batch)

        # Update the target networks
        self.actor_network.update_target()
        self.critic_network.update_target()

    def noise_action(self, state, epsilon):
        action = self.actor_network.action(state)
        noise_t = np.zeros(self.action_dim)
        noise_t[0] = epsilon * self.linear_noise.noise()
        noise_t[1] = epsilon * self.angular_noise.noise()
        action = action + noise_t
        a_linear = np.clip(action[0], 0, 1)
        a_linear = round(a_linear, 1)
        a_angular = np.clip(action[1], -1, 1)
        a_angular = round(a_angular, 1)
        #print(a_linear, a_angular)

        return [a_linear, a_angular]

    def action(self, state):
        action = self.actor_network.action(state)
        a_linear = np.clip(action[0], 0, 1)
        a_linear = round(a_linear, 1)
        a_angular = np.clip(action[1], -1, 1)
        a_angular = round(a_angular, 1)

        return [a_linear, a_angular]

    def perceive(self,state,action,reward,next_state,done):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.replay_buffer.add(state,action,reward,next_state,done)
        if self.replay_buffer.count() == REPLAY_START_SIZE:
            print('\n---------------Start training---------------')
        # Store transitions to replay start size then start training
        if self.replay_buffer.count() > REPLAY_START_SIZE:
            self.time_step += 1
            self.train()

        if self.time_step % 10000 == 0 and self.time_step > 0:
            self.actor_network.save_network(self.time_step)
            self.critic_network.save_network(self.time_step)

        if done:
            self.linear_noise.reset()
            self.angular_noise.reset()


        return self.time_step










