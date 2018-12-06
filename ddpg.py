import numpy as np
import random
from collections import namedtuple, deque
import copy
import time

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 8e-2             # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate for actor
LR_CRITIC = 1e-3        # learning rate for critic
WEIGHT_DECAY = 0     # L2 weight decay for critic
UPDATE_EVERY = 10
NUM_UPDATES = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class DDPGAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, n_agents, seed, pretrainedWeightsFile='checkpoint_actor.pth', train = True):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            n_agents (int): number of agents in the multi-agent env
            pretrainedWeightsFile (string): filename for pretrained weights when running in test mode
            train (bool): True when training, False when Testing
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents
        self.seed = random.seed(seed)

        self.train = train
        if self.train:
            self.actor = Actor(state_size, action_size, seed).to(device)                    # Actor Q network
            self.critic = Critic(state_size, action_size, seed).to(device)                  # Critic Q network
            
            self.actor_tgt = Actor(state_size, action_size, seed).to(device)                # Target Actor Q network
            self.soft_update(self.actor, self.actor_tgt, 1.0)     
            self.critic_tgt = Critic(state_size, action_size, seed).to(device)              # Target Critic Q network
            self.soft_update(self.critic, self.critic_tgt, 1.0)     
            
            self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)         # Optimizer for training the actor
            self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)    # Optimizer for training the critic
            
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)          # Replay memory
            self.t_step = 0                                                                 # Initialize time step (for updating every UPDATE_EVERY steps)
            self.noise = OUNoise(action_size, seed)                                         # Noise Process
        else:
            self.actor = Actor(state_size, action_size, seed).to(device)                    # Local Q network
            self.actor.load_state_dict(torch.load(pretrainedWeightsFile))                   # Load pre trained weights for Q network from file if testing
    
    def step(self, states, actions, rewards, next_states, dones):
        """
        Define step behavior of agent

        Params
        ======
            states (array of array): current state(s) of the agent(s)
            actions (array of array): action(s) taken
            rewards (array_like): reward(s) procured  
            next_state (array of array): transitioned state(s)
            dones (array_like): indicates whether the episode has ended
        """
        # Save experience in replay memory
        self.t_step+=1;
        for i in range(self.n_agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        if (len(self.memory) > BATCH_SIZE) and (self.t_step%UPDATE_EVERY == 0):
        # if (len(self.memory) > BATCH_SIZE):
            self.t_step = 0
            for _ in range(NUM_UPDATES):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array of array): current state(s)
            add_noise(bool): indicates whether to add random noise to the actions
        """
        states = torch.from_numpy(states).float().to(device)
        self.actor.eval()
        with torch.no_grad():
            action_values = self.actor(states).cpu().data.numpy()
        
        if self.train:
          self.actor.train()

        if self.train and add_noise:
            action_values += [self.noise.sample() for _ in range(self.n_agents)]

        return np.clip(action_values, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## ------------------- Update Critic ----------------------- #
        next_actions = self.actor_tgt(next_states)
        critic_tgt_next = self.critic_tgt(next_states, next_actions)
        
        critic_tgt = rewards + (gamma*critic_tgt_next*(1-dones))
        # print(actions.size())
        critic_exp = self.critic(states, actions)

        critic_loss = F.mse_loss(critic_exp, critic_tgt)
        
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        ## -------------------- Update Actor ----------------------- #
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.critic, self.critic_tgt, TAU)
        self.soft_update(self.actor, self.actor_tgt, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state