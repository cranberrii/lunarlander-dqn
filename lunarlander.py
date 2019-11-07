import numpy as np
import random
import gym
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD

seed = 5
np.random.seed(seed)


class DQN:
    def __init__(self):
        self.env = gym.make('LunarLander-v2')
        # self.env = gym.make('CartPole-v0')

        self.state_size = self.env.observation_space.shape[0]  # 8
        self.action_size = self.env.action_space.n  # 4

        self.gamma = 0.95  # TODO larger?
        self.epsilon = 0.90
        self.epsilon_min = 0.01  # TODO smaller?
        self.epsilon_decay = 0.999  # TODO 0.9995
        self.learning_rate = 0.001  ### 0.001
        self.learning_rate_decay = 0.01
        self.episodes = 1000
        self.batch_size = 16
        self.memory = deque(maxlen=10000)

        self.model = self.nn_model_build()
        self.goal_model = self.nn_model_build()  # updates slower, enables convergence
        print(self.model.summary())

    def random_agent(self):
        observation_samples = []
        # play a bunch of games randomly and collect observations
        for n in range(5):
            state = self.env.reset()
            observation_samples.append(state)
            done = False
            while not done:
                self.env.render()
                action = np.random.randint(self.env.action_space.n)
                state, reward, done, _ = self.env.step(action)
                observation_samples.append(state)
        observation_samples = np.array(observation_samples)
        self.env.close()

    def nn_model_build(self):
        # build the deep nn Q model
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        # model.add(BatchNormalization())
        model.add(Dense(128, activation='relu'))
        # model.add(BatchNormalization())
        # model.add(Dense(32, activation='relu'))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.1))  # TODO: makes it learn slower
        model.add(Dense(self.action_size, activation='linear'))  # TODO softmax?
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(lr=self.learning_rate))  # momentum=0.99
        return model

    def choose_action(self, state):
        # print(state.reshape((1, self.state_size)).shape)
        # print(np.array([state]).shape)
        if np.random.random() < self.epsilon:  # e-greedy selects a random action with probability e (very small)
            return np.random.randint(self.env.action_space.n)  # dont use env.action_space.sample()
        else:
            return np.argmax(self.model.predict_on_batch(state.reshape((1, self.state_size))))  # greedy

    def remember(self, state, action, reward, state2, done):
        self.memory.append((state, action, reward, state2, done))

    def replay(self, batch_size):
        # train the agent with the experience of the episode
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)

        # TODO train on full batch?
        for state, action, reward, state2, done in minibatch:
            y_target = self.goal_model.predict_on_batch(state.reshape((1, self.state_size)))
            # print(state)
            # print(y_target)

            # terminal state
            if done:
                y_target[0][action] = reward
            # non-term state, Q update
            else:
                Q_future = max(self.goal_model.predict_on_batch(state2.reshape((1, self.state_size)))[0])
                # print('QF',Q_future)
                y_target[0][action] = reward + self.gamma * Q_future

            # keras: Trains the model for a fixed number of epochs
            # self.model.fit(state.reshape((1, self.state_size)), y_target, epochs=1, verbose=0)
            self.model.train_on_batch(state.reshape((1, self.state_size)), y_target)

    def goal_train(self):
        # copy weights from main to goal model
        weights = self.model.get_weights()
        goal_weights = self.goal_model.get_weights()

        # print('GW:', len(goal_weights), len(weights))
        for i in range(len(goal_weights)):
            goal_weights[i] = weights[i]
        self.goal_model.set_weights(goal_weights)

    def run(self):
        # scores = deque(maxlen=100)
        total_episodes = []
        total_steps = []
        total_rewards = np.array([])

        for e in range(self.episodes):
            state = self.env.reset()
            # state = np.reshape(state, [1, 4])
            done = False
            totalreward = 0
            i = 0

            # decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            while not done:
                # self.env.render()

                action = self.choose_action(state)
                state2, reward, done, _ = self.env.step(action)
                # state2 = np.reshape(state2, [1, 4])
                self.remember(state, action, reward, state2, done)

                self.replay(self.batch_size)  # internally iterates default (prediction) model
                self.goal_train()  # iterates target model

                state = state2
                totalreward += reward  # total reward for ONE episode
                i += 1

            # TODO:
            # The reward for each training episode while training your agent?
            # The reward per trial for 100 trials using your trained agent?

            print("episode:", e, "iters:", i, "epsilon", self.epsilon, "total reward:", totalreward)
            total_episodes.append(e)
            total_steps.append(i)
            total_rewards = np.append(total_rewards, totalreward)
            if e % 10 == 0:
                print("======", "episode:", e, "total reward:", totalreward,
                      "avg reward (last 10):", total_rewards[max(0, e - 10):(e + 1)].mean(), "======")
            # if totalrewards[max(0, n - 100):(n + 1)].mean() >= 200:
            #     break

        print("avg reward for last 100 episodes:", total_rewards[-100:].mean())
        print("total steps:", total_rewards.sum())

        log_df = pd.DataFrame(data={'episodes': total_episodes, 'steps': total_steps, 'rewards': total_rewards})
        log_df.to_csv('logs/data.{}.csv'.format(datetime.now()), index=False)
        self.model.save("saved_models/dqn.{}.h5".format(datetime.now()))

        return total_rewards

    def test_dqn_agent(self, model_path, episodes=5, epsilon=0.13, render=True):
        total_episodes = []
        total_steps = []
        total_rewards = np.array([])

        trained_model = load_model(model_path)

        def choose_action(s):
            if np.random.random() < epsilon:  # e-greedy selects a random action with probability e (very small)
                return np.random.randint(self.env.action_space.n)
            else:
                return np.argmax(trained_model.predict_on_batch(s.reshape((1, self.state_size))))  # greedy

        for e in range(episodes):
            state = self.env.reset()
            done = False
            totalreward = 0
            i = 0

            while not done:
                if render:
                    self.env.render()

                action = choose_action(state)
                state2, reward, done, _ = self.env.step(action)

                state = state2
                totalreward += reward  # total reward for ONE episode
                i += 1

            # TODO:
            # The reward per trial for 100 trials using your trained agent?
            print("episode:", e, "iters:", i, "epsilon", epsilon, "total reward:", totalreward)
            total_episodes.append(e)
            total_steps.append(i)
            total_rewards = np.append(total_rewards, totalreward)
            if e % 10 == 0:
                print("======", "episode:", e, "total reward:", totalreward,
                      "avg reward (last 10):", total_rewards[max(0, e - 10):(e + 1)].mean(), "======")

        print("avg reward for last 100 episodes:", total_rewards[-100:].mean())
        print("total rewards:", total_rewards.sum())

        return total_rewards


    # TODO
    # The effect of hyperparameters (alpha, lambda , epsilon) on your agent
    # You pick the ranges
    # Be prepared to explain why you chose them

    def plot_running_avg(self, tot_rewards):
        N = len(tot_rewards)
        running_avg = np.empty(N)
        for t in range(N):
            running_avg[t] = np.mean(tot_rewards[max(0, t-100):(t+1)])

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(running_avg)
        plt.title("running average of rewards", size=16)
        plt.xlabel('episode', size=14)
        plt.ylabel('reward', size=14)
        plt.rc('xtick', labelsize=14)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=14)  # fontsize of the tick labels
        fig.text(0.9, 0.1, 'zbek3@gatech.edu',
                 fontsize=35, color='gray',
                 ha='right', va='bottom', alpha=0.33)
        plt.show()
        # plt.savefig()

    def plot_training_reward(self, tot_rewards):
        # The reward for each training episode while training your agent?
        # The reward per trial for 100 trials using your trained agent?
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(tot_rewards)
        plt.title("reward for each training episode", size=16)
        plt.xlabel('episode', size=14)
        plt.ylabel('reward', size=14)
        plt.rc('xtick', labelsize=14)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=14)  # fontsize of the tick labels
        fig.text(0.9, 0.1, 'zbek3@gatech.edu',
                 fontsize=35, color='gray',
                 ha='right', va='bottom', alpha=0.33)
        plt.show()
        # plt.savefig()


if __name__ == '__main__':
    start = datetime.now()
    print(start)

    agent = DQN()
    # agent.random_agent()

    # TR = agent.run()
    # agent.plot_training_reward(TR)
    # agent.plot_running_avg(TR)

    # mp = 'saved_models/lunarlander.dqn.1000.2019-10-27_19:59:52.h5'
    mp = 'saved_models/lunarlander.dqn.700.2019-10-27-17:26:35.h5'
    test_tr = agent.test_dqn_agent(model_path=mp, episodes=100, epsilon=0.13, render=False)
    agent.plot_training_reward(test_tr)
    agent.plot_running_avg(test_tr)

    print('total elapsed time:', datetime.now() - start)
