# drlnd-collab
Reinforcement Learning Nanodegree Project (Udacity) - Collaboration and Competition

# Introduction
The project aims to solve an environment similar to the Tennis Unity environment employing Policy-Based Methods for Deep Reinforcement Learning. In this environment, two agents control rackets to bounce a ball over a net and the goal of each agent is to keep the ball in play.

# Getting Started
Follow this [link](https://github.com/udacity/deep-reinforcement-learning#dependencies) to setup the Udacity DRLND conda enviroment.

There are 2 ways to explore the code:
1. By following the guided iPython notebook - _Continuous_Control.ipynb_
2. By directly running _continuouscontrol.py_ from command line
   * Command Line argument to be passed : _train_ for training and _test_ for testing
   
       `($) python continuouscontrol.py train`
   
       `($) python continuouscontrol.py test`
   
Download the environment for this project as per your OS from the following links:
  Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
  Linux Headless: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip)
  Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
  Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
  Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Update the path to the environment in the respecting file (_Tennis.ipynb_ or _collabcompet.py_)    

- **Mac**: `path/to/Tennis.app`,
- **Windows** (x86): `path/to/Tennis_Windows_x86/Reacher.exe`,
- **Windows** (x86_64): `path/to/Tennis_Windows_x86_64/Tennis.exe`,
- **Linux** (x86): `path/to/Tennis_Linux/Tennis.x86`,
- **Linux** (x86_64): `path/to/Tennis_Linux/Tennis.x86_64`,
- **Linux** (x86, headless): `path/to/Tennis_Linux_NoVis/Tennis.x86`,
- **Linux** (x86_64, headless): `path/to/Tennis_Linux_NoVis/Tennis.x86_64`,

Update the following files to tweak the model or the agent:
- `model.py` defines the architectures for Actor and Critic networks
- `ddpg.py`defines the behavior of the DDPG Agent

# The Environment
## State Space
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.

## Action Space
Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

## Rewards
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.

## Solution Criteria
The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically, after each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

# General Instructions
## Training
Follow the Jupyter notebook or run _collabcompet.py_ with argument _train_.

This will run a round of training as per details in _collabcompet.py_. You will be able to observe the agent performance if unity environment visualization is enabled. Upon completion, the trained model parameters will be saved in _checkpoint_actor.pth_ and _checkpoint_critic.pth_ for the _Actor_ and _Critic_ respectively.

## Testing
Follow the Jupyter notebook or run _collabcompet.py_ with argument _test_.

This wil run one episode of the agent with the model parameters saved in _checkpoint_actor.pth_ and _checkpoint_critic.pth_ for _Actor_ and _Critic_ respectively. You can observe the performance of the agent if the unity environment visualization is enabled.
