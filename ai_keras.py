# AI for Doom



# Importing the libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from keras.optimizers import Adam
import random

# Importing the packages for OpenAI and Doom
import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

# Importing the other Python files
import experience_replay, image_preprocessing

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def choose_multinomial(probs):
    r = random.random()
    index = 0
    while(r >= 0 and index < len(probs)):
        r -= probs[index]
        index += 1
    return index - 1

# Part 1 - Building the AI

# Making the brain

class CNN():
    
    def __init__(self, number_actions):
        self.number_actions = number_actions
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, input_shape=(80,80,1), activation='relu', kernel_size=5))
        model.add(MaxPooling2D(pool_size=3,strides=2))
        model.add(Conv2D(32, activation='relu', kernel_size=3))
        model.add(MaxPooling2D(pool_size=3,strides=2))
        model.add(Conv2D(64, activation='relu', kernel_size=2))
        model.add(MaxPooling2D(pool_size=3,strides=2))
        model.add(Flatten())
        model.add(Dense(40, activation='relu'))
        model.add(Dense(self.number_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam())
        return model

# Making the body

class SoftmaxBody():
    def __init__(self, T):
        self.T = T

    def choose_actions(self, outputs):
        probs = softmax((outputs * self.T)[0])
        actions = [[choose_multinomial(probs)]]
        return actions

# Making the AI

class AI:

    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, inputs):
        input = np.array(inputs, dtype = np.float32)
        output = self.brain.model.predict(input)
        actions = self.body.choose_actions(output)
        return actions



# Part 2 - Training the AI with Deep Convolutional Q-Learning

# Getting the Doom environment
doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width = 80, height = 80, grayscale = True)
doom_env = gym.wrappers.Monitor(doom_env, "videos", force = True)
number_actions = doom_env.action_space.n

# Building an AI
cnn = CNN(number_actions)
softmax_body = SoftmaxBody(T = 1.0)
ai = AI(brain = cnn, body = softmax_body)

# Setting up Experience Replay
n_steps = experience_replay.NStepProgress(env = doom_env, ai = ai, n_step = 10)
memory = experience_replay.ReplayMemory(n_steps = n_steps, capacity = 10000)
    
# Implementing Eligibility Trace
def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:
        input = (np.array([series[0].state, series[-1].state], dtype = np.float32))
        output = cnn.model.predict(input)
        cumul_reward = 0.0 if series[-1].done else output[1].max()
        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward
        state = series[0].state
        target = output[0].data
        target[series[0].action] = cumul_reward
        inputs.append(state)
        targets.append(target)
    return np.array(inputs, dtype = np.float32), np.array(targets)

# Making the moving average on 100 steps
class MA:
    def __init__(self, size):
        self.list_of_rewards = []
        self.size = size
    def add(self, rewards):
        if isinstance(rewards, list):
            self.list_of_rewards += rewards
        else:
            self.list_of_rewards.append(rewards)
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]
    def average(self):
        return np.mean(self.list_of_rewards)
ma = MA(100)

# Training the AI
nb_epochs = 100
for epoch in range(1, nb_epochs + 1):
    memory.run_steps(200)
    for batch in memory.sample_batch(128):
        inputs, targets = eligibility_trace(batch)
        #predictions = cnn.model.predict(inputs)
        cnn.model.fit(inputs, targets, batch_size= len(batch), epochs=1, verbose=0)

    rewards_steps = n_steps.rewards_steps()
    ma.add(rewards_steps)
    avg_reward = ma.average()
    print("Epoch: %s, Average Reward: %s" % (str(epoch), str(avg_reward)))
    if avg_reward >= 1500:
        print("Congratulations, your AI wins")
        break

# Closing the Doom environment
doom_env.close()
