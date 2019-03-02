# [ pytorl ] Components Description
### [PROJECT CURRENTLY UNDER DEVELOPMENT]

&nbsp;  

#### 1. agents:
<pre>This module contains implementations of RL agents. The goal is to make all
sophisticated choice-making, optimization, or utilities such as DQN replay buffer
completely inside of agent scope. Then wen can use it easily as a black-box.</pre>

#### 2. distributed:
<pre>This module contains implementations of distributed intialization and commu-
nication methods. The goal is to provided distributed support for pytorl.</pre>

#### 3. envs:
<pre>This module contains implementations of RL environment. Currently it contains
gym classic control and atari environment. The goal for this module is to regulate 
and provide general interface for rl environmments so that we can probably DIY our 
own learning environment and use it without changing training files. </pre>

#### 4. lib:
<pre>This module contains implementations of reinforcement learning algorithm 
related support. </pre>

#### 5. networks:
<pre>This module contains implementations of deep reinforcement learning neural 
network files. </pre>

#### 6. settings:
<pre>This module contains utilities for command line fast entry (rl-run and lrun), 
and pytorl package config.</pre>

#### 7. utils:
<pre>This module contains implementations of non-RL related general support like 
config reader and tensorboard recorder setup. The goal is to provide useful 
and convenient general tools.</pre>


