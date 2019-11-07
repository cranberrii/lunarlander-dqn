import numpy as np
import random
import gym
from collections import deque
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DQN:
    def __init__(self):
        self.gamma = 0.98
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.05 # 0.023
        self.learning_rate_decay = 0.01
        self.episodes = 1000
        self.batch_size = 64
        self.memory = deque(maxlen=5000)

        self.env = gym.make('CartPole-v0')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        # TODO: 1 model per action?
        self.model = self._nn_model_build()

    def _nn_model_build(self):
        # build the deep nn Q model
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate, decay=self.learning_rate_decay))
        return model

    def choose_action(self, state):
        if np.random.random() < self.epsilon:  # e-greedy selects a random action with probability e (very small)
            action = np.random.randint(self.env.action_space.n)  # dont use env.action_space.sample()
        else:
            # action = np.argmax(Q[state, :])
            action = np.argmax(self.model.predict(state))  # greedy
        return action

    def remember(self, state, action, reward, state2, done):
        self.memory.append((state, action, reward, state2, done))

    def run(self):
        scores = deque(maxlen=100)
        rewards = []

        for e in range(self.episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, 4])
            done = False
            i = 0

            while not done:
                # self.env.render()

                action = self.choose_action(state)
                state2, reward, done, _ = self.env.step(action)
                state2 = np.reshape(state2, [1, 4])
                self.remember(state, action, reward, state2, done)
                state = state2
                i += 1

                # print(self.epsilon)

            scores.append(i)
            mean_score = np.mean(scores)
            rewards.append(mean_score)
            # print('Rewards per episode: {}'.format(mean_score))
            # TODO
            # The reward for each training episode while training your agent?
            # The reward per trial for 100 trials using your trained agent?

            if mean_score >= 195 and e >= 100:  # TODO n_win_ticks ?
                print('Ran {} episodes. Solved after {} trials'.format(e, e - 100))
                e = e - 100
            if e % 100 == 0:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))

            # TODO why?
            self.replay(self.batch_size)

        print('Did not solve after {} episodes'.format(e))
        return e, rewards

    def replay(self, batch_size):
        # train the agent with the experience of the episode
        x_batch = []
        y_batch = []

        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))

        for state, action, reward, state2, done in minibatch:
            y_target = self.model.predict(state)
            # print(y_target[0])
            y_target[0][action] = reward if done \
                else reward + self.gamma * np.max(self.model.predict(state2)[0])

            x_batch.append(state[0])
            y_batch.append(y_target[0])

        # keras: Trains the model for a fixed number of epochs
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0, epochs=1)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # TODO
    def plot_running_avg(self, tot_rewards):
        N = len(tot_rewards)
        running_avg = np.empty(N)
        for t in range(N):
            running_avg[t] = np.mean(tot_rewards[max(0, t-100):(t+1)])
        plt.plot(running_avg)
        plt.title("Running Average")
        plt.show()


if __name__ == '__main__':
    agent = DQN()
    e, scores = agent.run()
    print(scores)
    agent.plot_running_avg(scores)
