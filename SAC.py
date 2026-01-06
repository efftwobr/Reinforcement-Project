import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import __init__
import matplotlib.pyplot as plt
import pickle

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=1)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)

def soft_update(target, source, rho):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1-rho) * sp.data + rho * tp.data)


#env_name = "CartPole-v1"
#env_name = "WordleEnv10-v0"
env_name = "WordleEnv100-v0"
#env_name = "WordleEnv1000-v0"

episodes = 20000


env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy = Policy(state_dim, action_dim)
q1 = Critic(state_dim, action_dim)
q2 = Critic(state_dim, action_dim)


###### FILE SAVING ########

#file1 = open("policy_model",'rb')
#policy = pickle.load(file1)
#file1.close()
#file2 = open("q1_model",'rb')
#q1 = pickle.load(file2)
#file2.close()
#file3 = open("q2_model",'rb')
#q2 = pickle.load(file3)
#file3.close()

###########################

q1_target = Critic(state_dim, action_dim)
q2_target = Critic(state_dim, action_dim)
q1_target.load_state_dict(q1.state_dict())
q2_target.load_state_dict(q2.state_dict())

pi_opt = torch.optim.Adam(policy.parameters(), 3e-4)
q1_opt = torch.optim.Adam(q1.parameters(), 3e-4)
q2_opt = torch.optim.Adam(q2.parameters(), 3e-4)

gamma = 0.99
alpha = 0.2
tau = 0.005
batch_size = 50
buffer = []
max_buffer = 10000

seed = 1
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)



#Training
returns = []
average_returns= []
guess_numbers = []
average_guess_numbers = []

for episode in range(episodes):
    state, _ = env.reset()
    done = False
    episode_returns = 0
    guesses = 0

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        dist = policy(state_tensor)
        action = torch.distributions.Categorical(dist).sample().item()
        guesses +=1

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.append((state, action, reward, next_state, done))
        if len(buffer) > max_buffer:
            buffer.pop(0)

        state = next_state
        episode_returns += reward

        if len(buffer) < batch_size:
            continue

        #Sample batch
        batch = random.sample(buffer, batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = map(np.array, zip(*batch))

        batch_state  = torch.tensor(batch_state,  dtype=torch.float32)
        batch_next_state = torch.tensor(batch_next_state, dtype=torch.float32)
        batch_action  = torch.tensor(batch_action).unsqueeze(1)
        batch_reward  = torch.tensor(batch_reward, dtype=torch.float32)
        batch_done  = torch.tensor(batch_done, dtype=torch.float32)

        #Critic target
        with torch.no_grad():
            next_dist = policy(batch_next_state)
            log_next_dist = torch.log(next_dist + 1e-8)
            min_q_target = torch.min(q1_target(batch_next_state), q2_target(batch_next_state))
            v_next = (next_dist * (min_q_target - alpha * log_next_dist)).sum(dim=1)
            y = batch_reward + gamma * (1 - batch_done) * v_next

        #Critic update
        q1_pred = q1(batch_state).gather(1, batch_action).squeeze()
        q2_pred = q2(batch_state).gather(1, batch_action).squeeze()

        q1_loss = F.mse_loss(q1_pred, y)
        q2_loss = F.mse_loss(q2_pred, y)

        q1_opt.zero_grad()
        q1_loss.backward()
        q1_opt.step()
        q2_opt.zero_grad() 
        q2_loss.backward() 
        q2_opt.step()

        #Policy update
        dist = policy(batch_state)
        log_dist = torch.log(dist + 1e-8)
        min_q = torch.min(q1(batch_state), q2(batch_state))
        pi_loss = (dist * (alpha * log_dist - min_q)).sum(dim=1).mean()

        pi_opt.zero_grad()
        pi_loss.backward()
        pi_opt.step()

        #Target update
        soft_update(q1_target, q1, tau)
        soft_update(q2_target, q2, tau)

    if env_name == "CartPole-v1":
        print(f"Episode {episode}, Return {episode_returns}")
        returns.append(episode_returns)
        #if episode_returns == 500:
        #    break
    else:

        guess_numbers.append(guesses)
        if episode_returns > 0:
            episode_returns = 10
        returns.append(episode_returns)
        if episode < 100:
            mean_guess_number = np.mean(guess_numbers)
            average_guess_numbers.append(mean_guess_number)
        else:
            mean_guess_number = np.mean(guess_numbers[-100:-1])
            average_guess_numbers.append(mean_guess_number)
            average_return = np.mean(returns[-100:-1])
            average_returns.append(average_return)        
        #print(f"Episode: {episode}, Guesses: {guesses}, Average: {mean_return}")
        if episode % 100 == 0 and episode > 0:
            print(f"Episode: {episode}, Guesses: {guesses}, Average guesses: {mean_guess_number}, Reward average: {average_return}")



env.close()


####### FILE LOADING ###############

#file1 = open("policy_model",'wb')
#pickle.dump(policy,file1)
#file1.close()
#file2 = open("q1_model",'wb')
#pickle.dump(q1,file2)
#file2.close()
#file3 = open("q2_model",'wb')
#pickle.dump(q2,file3)
#file3.close()

############################

plt.plot(average_returns)
plt.ylabel ('Average reward in the last 100 episodes')
plt.xlabel ('Episode')
if env_name == "WordleEnv10-v0":
    plt.title ('SAC 10 Word Wordle Rewards')
elif env_name == "WordleEnv100-v0":
    plt.title ('SAC 100 Word Wordle Rewards')
elif env_name == "WordleEnv1000-v0":
    plt.title ('SAC 1000 Word Wordle Rewards')
plt.show()
plt.plot(average_guess_numbers)
plt.ylabel ('Average guesses required in the last 100 episodes')
plt.xlabel ('Episode')
if env_name == "WordleEnv10-v0":
    plt.title ('SAC 10 Word Wordle Guesses')
elif env_name == "WordleEnv100-v0":
    plt.title ('SAC 100 Word Wordle Guesses')
elif env_name == "WordleEnv1000-v0":
    plt.title ('SAC 1000 Word Wordle Guesses')
plt.show()