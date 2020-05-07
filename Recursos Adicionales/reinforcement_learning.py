#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 18:50:03 2020

@author: juangabriel
"""

# Reinforcement Learning

obs = env.reset()
h = mdnrnn.initial_state
done = False
cumulative_reward = 0
while not done:
    z = cnnvae(obs)
    a = controller([z, h])
    obs, reward, done = env.step(a)
    cumulative_reward += reward
    h = mdnrnn([a, z, h])