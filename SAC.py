import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

import __init__


class NNpolicy(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim,16),
            nn.ReLU(),
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,output_dim),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        return self.model(x)

class NNcritic(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim+output_dim,16),
            nn.ReLU(),
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )
    def forward(self,x):
        return self.model(x)


def soft_update(target_NN,source_NN,rho):
    with torch.no_grad():
        for target_parameters, source_parameters in zip(target_NN.parameters(), source_NN.parameters()):
            target_parameters.copy_(rho * source_parameters + (1 - rho) * target_parameters)
    return target_NN



learning_rate = 3e-4
gamma = 0.95
alpha = 0.001
rho = 0.001

seed = 1

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

#env_name = "WordleEnv10-v0"
env_name = "WordleEnv100-v0"
#env_name = "WordleEnv1000-v0"
env = gym.make(env_name)

input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

policy_model = NNpolicy(input_dim,output_dim)
policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)

critic1_model = NNcritic(input_dim,output_dim)
critic1_optimizer = torch.optim.Adam(critic1_model.parameters(), lr=learning_rate)
critic2_model = NNcritic(input_dim,output_dim)
critic2_optimizer = torch.optim.Adam(critic2_model.parameters(), lr=learning_rate)
critic1_target = NNcritic(input_dim,output_dim)
critic2_target = NNcritic(input_dim,output_dim)

critic1_target = soft_update(critic1_target,critic1_model,1)
critic2_target = soft_update(critic2_target,critic2_model,1)



relay_buffer_batches = 10000
relay_buffer_batch_size = 10
update_batch_size = 10
relay_buffer = []
max_buffer = 10000

for batch in range(relay_buffer_batches):
    if batch % 100 == 0:
        print(batch)
    #print(batch)

    for episode in range(relay_buffer_batch_size):
        state,_ = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)

        done = False
        while not done:
            action_p = policy_model(state)
            dist = torch.distributions.Categorical(action_p)
            action = dist.sample()
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            relay_buffer.append([state,action,reward,next_state,done])
            if len(relay_buffer) > max_buffer:
                relay_buffer.pop(0)
            state = torch.FloatTensor(next_state).unsqueeze(0)

    critic1_loss = 0
    critic2_loss = 0
    policy_loss = 0
    for i in range(update_batch_size):
        
#        rand_index = random.randint(0, len(relay_buffer) - 1)
#        sample = relay_buffer[rand_index]
#
#        state      = torch.FloatTensor(sample[0])
#        action     = sample[1]
#        reward     = sample[2]
#        next_state = torch.FloatTensor(sample[3]).unsqueeze(0)
#        done       = sample[4]
#
#        with torch.no_grad():
#            probs = policy_model(next_state)
#            log_probs = torch.log(probs + 1e-8)
#
#            q_vals = []
#            for a in range(output_dim):
#                one_hot = torch.zeros(output_dim)
#                one_hot[a] = 1
#                one_hot = one_hot.unsqueeze(0)
#                q_vals.append(
#                    torch.min(
#                        critic1_target(torch.cat((next_state, one_hot), dim=1)),
#                        critic2_target(torch.cat((next_state, one_hot), dim=1))
#                    )
#                )
#
#            q_vals = torch.cat(q_vals, dim=1)
#            v_next = torch.sum(probs * (q_vals - alpha * log_probs))
#            if done:
#                y = reward
#            else:
#                y = reward + gamma * v_next
#
#            state = torch.Tensor.float(sample[0])
#            b = np.zeros(output_dim)
#            b[sample[1]] = 1
#            concatenated_input2 = torch.cat((state, torch.Tensor.float(torch.from_numpy(np.array([b])))),dim=1)
#        critic1_loss += (1/update_batch_size) * ((critic1_model(concatenated_input2)-y)**2)
#        critic2_loss += (1/update_batch_size) * ((critic2_model(concatenated_input2)-y)**2)
#        
#        
#        action_probs = policy_model(state)              # [1, A]
#        log_probs = torch.log(action_probs + 1e-8)
#
#        q_vals = []
#        for a in range(output_dim):
#            one_hot = torch.zeros(output_dim)
#            one_hot[a] = 1
#            one_hot = one_hot.unsqueeze(0)
#            q_vals.append(
#                torch.min(
#                critic1_model(torch.cat((state, one_hot), dim=1)),
#                critic2_model(torch.cat((state, one_hot), dim=1))
#                )
#            )   
#
#        q_vals = torch.cat(q_vals, dim=1)               # [1, A]
#
#        policy_loss += (1/update_batch_size) * torch.sum(
#            action_probs * (alpha * log_probs - q_vals)
#        )
        
        with torch.no_grad():
            rand_index = random.randint(0,len(relay_buffer)-1)
            sample = relay_buffer[rand_index]
            reward = sample[2]
            if sample[4]:
                y = reward
            else:
                next_state = torch.Tensor.float(torch.from_numpy(np.array([sample[3]]))) 
                second_action_dist = policy_model(next_state)
                dist = torch.distributions.Categorical(second_action_dist)
                second_action = dist.sample()
                a = np.zeros(output_dim)
                a[second_action] = 1
                concatenated_input = torch.cat((next_state, torch.Tensor.float(torch.from_numpy(np.array([a])))),dim=1)
                y = reward + gamma * (torch.min(critic1_target(concatenated_input),critic2_target(concatenated_input))
                                      - alpha * torch.log(second_action_dist[0,second_action]))
            state = torch.Tensor.float(sample[0])
            b = np.zeros(output_dim)
            b[sample[1]] = 1
            concatenated_input2 = torch.cat((state, torch.Tensor.float(torch.from_numpy(np.array([b])))),dim=1)
        critic1_loss += (1/update_batch_size) * ((critic1_model(concatenated_input2)-y)**2)
        critic2_loss += (1/update_batch_size) * ((critic2_model(concatenated_input2)-y)**2)        
        state = torch.Tensor.float(sample[0])
        chosen_action_dist = policy_model(state)
        dist = torch.distributions.Categorical(chosen_action_dist)
        chosen_action = dist.sample()
        c = np.zeros(output_dim)
        c[chosen_action] = 1
        critic1_policy_estimate = critic1_model(torch.cat((state, torch.Tensor.float(torch.Tensor(np.array([c])))),dim=1))
        critic2_policy_estimate = critic2_model(torch.cat((state, torch.Tensor.float(torch.Tensor(np.array([c])))),dim=1))
        policy_loss += -(1/update_batch_size) * (torch.min(critic1_policy_estimate,critic2_policy_estimate))- alpha* torch.log(policy_model(state)[0,chosen_action])

    critic1_optimizer.zero_grad()
    critic2_optimizer.zero_grad()
    policy_optimizer.zero_grad()
    critic1_loss.backward()
    critic2_loss.backward()
    policy_loss.backward()
    critic1_optimizer.step()
    critic2_optimizer.step()
    policy_optimizer.step()
    soft_update(critic1_target,critic1_model,rho)
    soft_update(critic2_target,critic2_model,rho)

count2 = 0
for i in range(100):
    state,_ = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0)
    count = 0
    done = False
    while not(done):
        count += 1
        action_p = policy_model(state)
        #print(action_p)
        dist = torch.distributions.Categorical(action_p)
        action = dist.sample()
        #print('action is',action)
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        state = torch.FloatTensor(next_state).unsqueeze(0)
    print('Score is',count)
    if count < 6:
        count2 += 1
print('count2 is',count2)
env.close()

