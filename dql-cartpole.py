"""
Deep Q Learning Networks
==== = ======== ========

Environment States:
    - Cart Position
    - Cart Velocity
    - Pole Angle
    - Pole Angular Velocity

Actions:
    - Move Cart Left
    - Move Cart Right
"""

# %%
# imports
import random
import gym
import numpy as np
import keras
import os
import tensorflow
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

print('Keras:', keras.__version__)

# %%
# set parameters
env = gym.make('CartPole-v0')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 32
n_episodes = 1001
output_dir = 'model_output/cartpole'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"State size: {state_size}, Action_size: {action_size}, Batch Size: {batch_size}, Episodes: {n_episodes}")

# %%
env.reset()
for _ in range(1000):
    env.render()
    next_state, reward, done, _ = env.step(env.action_space.sample())
    print(next_state, reward, done, _)
    if done:
        break
env.close()


# %%
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # neural net to approximate Q-value function:
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu')) # 1st hidden layer; states as input
        model.add(Dense(24, activation='relu')) # 2nd hidden layer
        model.add(Dense(self.action_size, activation='linear')) # 2 actions, so 2 output neurons: 0 and 1 (L/R)
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        print('Model:', model)
        return model

    # def _build_model(self):
    #     # neural net to approximate Q-value function:
    #     model = Sequential()
    #     model.add(Dense(24, input_dim=self.state_size, activation='relu'))
    #     model.add(Dense(24, activation='relu'))
    #     model.add(Dense(self.action_size, activation='linear'))
    #     model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
    #     return model

    def remember(self, state, action, reward, next_state, done):
        # list of previous experiences, enabling re-training later
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # if not acting randomly, predict reward value bases on current state
        act_values = self.model.predict(state)

        # pick the action that will give the highest reward (i.e. go left or right)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        """
        Trains the NN with experiences samples from memory
        :param batch_size:
        :return:
        """

        # sample a minibatch from memory
        minibatch = random.sample(self.memory, batch_size)

        # extract data for each minibatch sample
        for state, action, reward, next_state, done in minibatch:
            # if done (boolean whether game ended or not, i.e. whether final state or not
            target = reward

            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


# %%
# initialize agent
agent = DQNAgent(state_size, action_size)


#%%
done = False
# iterate over new episodes of the game
for e in range(n_episodes):
    # reset state at start of each new episode of the game
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    for time in range(5000):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f'episode: {e}/{n_episodes}, score: {time}, e: {agent.epsilon}')
            break

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    if e % 50 == 0:
        agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")

