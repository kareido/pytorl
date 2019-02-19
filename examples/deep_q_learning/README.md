### Deep Q-learning

This DQN example contains following implementations. Check the config to see what you can change and how to switch between different versions of deep q-learning.  

#### 1. Human-Level Control Through Deep Reinforcement Learning
Source: https://www.nature.com/articles/nature14236  

This is the original deep q-learning (Nature version). It is also the basic one. 

In this repo the only difference is that I "maxpooled" through all skipped frames rather than just using the last two frames (as in original publication and openai baselines repo) since I found the former works better. 

This DQN is also known as "natural DQN".  

#### 1. Deep Reinforcement Learning with Double Q-learning
Source: https://www.nature.com/articles/nature14236  

This is the improved version of deep q-learning trying to tackle the problem of agent overestimating action values in the preivous natural DQN.

**run example**

```bash
$ cd run_project/
$ sh <script name> <partition>
```
