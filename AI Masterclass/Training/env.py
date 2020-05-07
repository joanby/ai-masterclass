# Building the Environment

# Importing the libraries

import numpy as np
from scipy.misc import imresize as resize
from gym.spaces.box import Box
from gym.envs.box2d.car_racing import CarRacing

# Setting the dimensions of the game screen

SCREEN_X = 64
SCREEN_Y = 64

# Making a function that resizes the frame into 64x64 dimensions

def _process_frame(frame):
  obs = frame[0:84, :, :].astype(np.float)/255.0
  obs = resize(obs, (64, 64))
  obs = ((1.0 - obs) * 255).round().astype(np.uint8)
  return obs

# Building a class that creates the CarRacing game window and overrides the step function which is used to move the car to the next state (frame)

class CarRacingWrapper(CarRacing):

  # Initializing all the parameters and variables of the CarRacingWrapper class
  def __init__(self, full_episode=False):
    super(CarRacingWrapper, self).__init__()
    self.full_episode = full_episode
    self.observation_space = Box(low=0, high=255, shape=(SCREEN_X, SCREEN_Y, 3))

  # Making a method that plays an action and returns the next state (frame), the reward and done
  def _step(self, action):
    obs, reward, done, _ = super(CarRacingWrapper, self)._step(action)
    if self.full_episode:
      return _process_frame(obs), reward, False, {}
    return _process_frame(obs), reward, done, {}

# Making a function that builds and returns a CarRacing environment in reality

def make_env(env_name, seed=-1, render_mode=False, full_episode=False):
  env = CarRacingWrapper(full_episode=full_episode)
  if (seed >= 0):
    env.seed(seed)
  return env

# Running the main code with the game controls

def game_runner():
  from pyglet.window import key
  a = np.array( [0.0, 0.0, 0.0] )
  def key_press(k, mod):
    global restart
    if k==0xff0d: restart = True
    if k==key.LEFT:  a[0] = -1.0
    if k==key.RIGHT: a[0] = +1.0
    if k==key.UP:    a[1] = +1.0
    if k==key.DOWN:  a[2] = +0.8
  def key_release(k, mod):
    if k==key.LEFT  and a[0]==-1.0: a[0] = 0
    if k==key.RIGHT and a[0]==+1.0: a[0] = 0
    if k==key.UP:    a[1] = 0
    if k==key.DOWN:  a[2] = 0
  env = CarRacing()
  env.render()
  env.viewer.window.on_key_press = key_press
  env.viewer.window.on_key_release = key_release
  while True:
    env.reset()
    total_reward = 0.0
    steps = 0
    restart = False
    while True:
      s, r, done, info = env.step(a)
      total_reward += r
      if steps == 900:
        print("\n")
        print("_______________________________")
        print("\n")
        print("Human Intelligence Result:")
        print("Total Steps: {}".format(steps))
        print("Total Reward: {:.0f}".format(total_reward))
        print("\n")
        print("_______________________________")
        print("\n")
        break
      steps += 1
      env.render()
      if restart: break
  env.monitor.close()

if __name__=="__main__":
	game_runner()
