# Evolution Strategies Toolkit

# Importing the libraries

import numpy as np
import cma

# Making a function that takes as input a vector x and returns ranks in [0, len(x)]

def compute_ranks(x):
  assert x.ndim == 1
  ranks = np.empty(len(x), dtype=int)
  ranks[x.argsort()] = np.arange(len(x))
  return ranks

# Making a function that normalizes the ranks which will later be used to compute the reward matrix

def compute_centered_ranks(x):
  y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
  y /= (x.size - 1)
  y -= .5
  return y

# Making a function that applies weight decay to solutions produced by ES optimizers. This is done to faster optimize the model.

def compute_weight_decay(weight_decay, model_param_list):
  model_param_grid = np.array(model_param_list)
  return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)

# Building the optimizer root structure within a class

class Optimizer(object):

  # Initializing all the parameters and variables of the Optimizer class
  def __init__(self, pi, epsilon=1e-08):
    self.pi = pi
    self.dim = pi.num_params
    self.epsilon = epsilon
    self.t = 0

  # Making a method that updates the optimizers parameters based on the optimization process
  def update(self, globalg):
    self.t += 1
    step = self._compute_step(globalg)
    theta = self.pi.mu
    ratio = np.linalg.norm(step) / (np.linalg.norm(theta) + self.epsilon)
    self.pi.mu = theta + step
    return ratio

  # Making a method that acts as a placeholder for the step function in an optimizer
  def _compute_step(self, globalg):
    raise NotImplementedError

# Building the Basic Stochastic Gradient Descent optimizer within a class

class BasicSGD(Optimizer):

  # Initializing all the parameters and variables of the BasicSGD class
  def __init__(self, pi, stepsize):
    Optimizer.__init__(self, pi)
    self.stepsize = stepsize

  # Making a method that performs the update of the parameters by the SGD optimizer
  def _compute_step(self, globalg):
    step = -self.stepsize * globalg
    return step

# Building the Momentum SGD optimizer within a class

class SGD(Optimizer):

  # Initializing all the parameters and variables of the SGD class
  def __init__(self, pi, stepsize, momentum=0.9):
    Optimizer.__init__(self, pi)
    self.v = np.zeros(self.dim, dtype=np.float32)
    self.stepsize, self.momentum = stepsize, momentum

  # Making a method that performs the update of the parameters by the Momentum SGD optimizer
  def _compute_step(self, globalg):
    self.v = self.momentum * self.v + (1. - self.momentum) * globalg
    step = -self.stepsize * self.v
    return step

# Building the Adam optimizer within a class

class Adam(Optimizer):

  # Initializing all the parameters and variables of the Adam class
  def __init__(self, pi, stepsize, beta1=0.99, beta2=0.999):
    Optimizer.__init__(self, pi)
    self.stepsize = stepsize
    self.beta1 = beta1
    self.beta2 = beta2
    self.m = np.zeros(self.dim, dtype=np.float32)
    self.v = np.zeros(self.dim, dtype=np.float32)

  # Making a method that performs the update of the parameters by the Adam optimizer
  def _compute_step(self, globalg):
    a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
    self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
    self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
    step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
    return step

# Building the Covariance Matrix Adapatation Evolution Strategy (CMA-ES) optimizer within a class

class CMAES:

  # Initializing all the parameters and variables of the CMAES class
  def __init__(self, num_params, sigma_init=0.10, popsize=255, weight_decay=0.01):
    self.num_params = num_params
    self.sigma_init = sigma_init
    self.popsize = popsize
    self.weight_decay = weight_decay
    self.solutions = None
    self.es = cma.CMAEvolutionStrategy(self.num_params * [0], self.sigma_init, {'popsize': self.popsize,})

  # Making a method that returns the value of the standard deviation (sigma parameter) in the CMA-ES optimizer
  def rms_stdev(self):
    sigma = self.es.result[6]
    return np.mean(np.sqrt(sigma*sigma))

  # Making a method that asks the CMA-ES optimizer to give us a set of candidate solutions
  def ask(self):
    self.solutions = np.array(self.es.ask())
    return self.solutions

  # Making a method that gives the list of fitness results back to the CMA-ES optimizer
  def tell(self, reward_table_result):
    reward_table = -np.array(reward_table_result)
    if self.weight_decay > 0:
      l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
      reward_table += l2_decay
    self.es.tell(self.solutions, (reward_table).tolist())

  # Making a method that returns the current result from the Evolution Strategy algorithm
  def current_param(self):
    return self.es.result[5]

  # Making a method that sets the parameters obtained by each optimizer into an array, but unnecessary to do that for CMAES so pass
  def set_mu(self, mu):
    pass

  # Making a method that returns the best set of optimized parameters
  def best_param(self):
    return self.es.result[0]

  # Making a method that returns all the results (look for the CMA-ES documentation for more details)
  def result(self):
    r = self.es.result
    return (r[0], -r[1], -r[1], r[6])

# Building the Genetic Algorithm optimizer within a class

class SimpleGA:

  # Initializing all the parameters and variables of the SimpleGA class
  def __init__(self, num_params, sigma_init=0.1, sigma_decay=0.999, sigma_limit=0.01, popsize=256, elite_ratio=0.1, forget_best=False, weight_decay=0.01,):
    self.num_params = num_params
    self.sigma_init = sigma_init
    self.sigma_decay = sigma_decay
    self.sigma_limit = sigma_limit
    self.popsize = popsize
    self.elite_ratio = elite_ratio
    self.elite_popsize = int(self.popsize * self.elite_ratio)
    self.sigma = self.sigma_init
    self.elite_params = np.zeros((self.elite_popsize, self.num_params))
    self.elite_rewards = np.zeros(self.elite_popsize)
    self.best_param = np.zeros(self.num_params)
    self.best_reward = 0
    self.first_iteration = True
    self.forget_best = forget_best
    self.weight_decay = weight_decay

  # Making a method that returns the value of the standard deviation (sigma parameter) in the GA optimizer
  def rms_stdev(self):
    return self.sigma

  # Making a method that asks the GA optimizer to give us a set of candidate solutions
  def ask(self):
    self.epsilon = np.random.randn(self.popsize, self.num_params) * self.sigma
    solutions = []
    def mate(a, b):
      c = np.copy(a)
      idx = np.where(np.random.rand((c.size)) > 0.5)
      c[idx] = b[idx]
      return c
    elite_range = range(self.elite_popsize)
    for i in range(self.popsize):
      idx_a = np.random.choice(elite_range)
      idx_b = np.random.choice(elite_range)
      child_params = mate(self.elite_params[idx_a], self.elite_params[idx_b])
      solutions.append(child_params + self.epsilon[i])
    solutions = np.array(solutions)
    self.solutions = solutions
    return solutions

  # Making a method that gives the list of fitness results back to the GA optimizer
  def tell(self, reward_table_result):
    assert(len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."
    reward_table = np.array(reward_table_result)
    if self.weight_decay > 0:
      l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
      reward_table += l2_decay
    if (not self.forget_best or self.first_iteration):
      reward = reward_table
      solution = self.solutions
    else:
      reward = np.concatenate([reward_table, self.elite_rewards])
      solution = np.concatenate([self.solutions, self.elite_params])
    idx = np.argsort(reward)[::-1][0:self.elite_popsize]
    self.elite_rewards = reward[idx]
    self.elite_params = solution[idx]
    self.curr_best_reward = self.elite_rewards[0]
    if self.first_iteration or (self.curr_best_reward > self.best_reward):
      self.first_iteration = False
      self.best_reward = self.elite_rewards[0]
      self.best_param = np.copy(self.elite_params[0])
    if (self.sigma > self.sigma_limit):
      self.sigma *= self.sigma_decay

  # Making a method that returns the current model parameters
  def current_param(self):
    return self.elite_params[0]

  # Making a method that sets the parameters obtained by each optimizer into an array, but unnecessary to do that for GA so pass
  def set_mu(self, mu):
    pass

  # Making a method that returns the best set of optimized parameters
  def best_param(self):
    return self.best_param

  # Making a method that returns the best parameters obtained so far, along with the historically best reward, the current reward, and the standard deviation sigma
  def result(self):
    return (self.best_param, self.best_reward, self.curr_best_reward, self.sigma)

# Building the OpenAI Evolution Strategies optimizer within a class

class OpenES:

  # Initializing all the parameters and variables of the OpenES class
  def __init__(self, num_params, sigma_init=0.1, sigma_decay=0.999, sigma_limit=0.01, learning_rate=0.01, learning_rate_decay = 0.9999, learning_rate_limit = 0.001, popsize=256, antithetic=False, weight_decay=0.01, rank_fitness=True, forget_best=True):
    self.num_params = num_params
    self.sigma_decay = sigma_decay
    self.sigma = sigma_init
    self.sigma_init = sigma_init
    self.sigma_limit = sigma_limit
    self.learning_rate = learning_rate
    self.learning_rate_decay = learning_rate_decay
    self.learning_rate_limit = learning_rate_limit
    self.popsize = popsize
    self.antithetic = antithetic
    if self.antithetic:
      assert (self.popsize % 2 == 0), "Population size must be even"
      self.half_popsize = int(self.popsize / 2)
    self.reward = np.zeros(self.popsize)
    self.mu = np.zeros(self.num_params)
    self.best_mu = np.zeros(self.num_params)
    self.best_reward = 0
    self.first_interation = True
    self.forget_best = forget_best
    self.weight_decay = weight_decay
    self.rank_fitness = rank_fitness
    if self.rank_fitness:
      self.forget_best = True
    self.optimizer = Adam(self, learning_rate)

  # Making a method that returns the value of the standard deviation (sigma parameter) in the OpenAI ES optimizer
  def rms_stdev(self):
    sigma = self.sigma
    return np.mean(np.sqrt(sigma*sigma))

  # Making a method that asks the OpenAI ES optimizer to give us a set of candidate solutions
  def ask(self):
    if self.antithetic:
      self.epsilon_half = np.random.randn(self.half_popsize, self.num_params)
      self.epsilon = np.concatenate([self.epsilon_half, - self.epsilon_half])
    else:
      self.epsilon = np.random.randn(self.popsize, self.num_params)
    self.solutions = self.mu.reshape(1, self.num_params) + self.epsilon * self.sigma
    return self.solutions

  # Making a method that gives the list of fitness results back to the OpenAI ES optimizer
  def tell(self, reward_table_result):
    assert(len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."
    reward = np.array(reward_table_result)
    if self.rank_fitness:
      reward = compute_centered_ranks(reward)
    if self.weight_decay > 0:
      l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
      reward += l2_decay
    idx = np.argsort(reward)[::-1]
    best_reward = reward[idx[0]]
    best_mu = self.solutions[idx[0]]
    self.curr_best_reward = best_reward
    self.curr_best_mu = best_mu
    if self.first_interation:
      self.first_interation = False
      self.best_reward = self.curr_best_reward
      self.best_mu = best_mu
    else:
      if self.forget_best or (self.curr_best_reward > self.best_reward):
        self.best_mu = best_mu
        self.best_reward = self.curr_best_reward
    normalized_reward = (reward - np.mean(reward)) / np.std(reward)
    change_mu = 1./(self.popsize*self.sigma)*np.dot(self.epsilon.T, normalized_reward)
    self.optimizer.stepsize = self.learning_rate
    update_ratio = self.optimizer.update(-change_mu)
    if (self.sigma > self.sigma_limit):
      self.sigma *= self.sigma_decay
    if (self.learning_rate > self.learning_rate_limit):
      self.learning_rate *= self.learning_rate_decay

  # Making a method that returns the current parameters
  def current_param(self):
    return self.curr_best_mu

  # Making a method that sets the parameters obtained by the OpenES optimizer into an array
  def set_mu(self, mu):
    self.mu = np.array(mu)

  # Making a method that returns the best set of optimized parameters
  def best_param(self):
    return self.best_mu

  # Making a method that returns the best parameters obtained so far, along with the historically best reward, the current reward, and the standard deviation sigma
  def result(self):
    return (self.best_mu, self.best_reward, self.curr_best_reward, self.sigma)

# Building the PEPG optimizer within a class

class PEPG:

  # Initializing all the parameters and variables of the PEPG class
  def __init__(self, num_params, sigma_init=0.10, sigma_alpha=0.20, sigma_decay=0.999, sigma_limit=0.01, sigma_max_change=0.2, learning_rate=0.01, learning_rate_decay = 0.9999, learning_rate_limit = 0.01, elite_ratio = 0, popsize=256, average_baseline=True, weight_decay=0.01, rank_fitness=True, forget_best=True):
    self.num_params = num_params
    self.sigma_init = sigma_init
    self.sigma_alpha = sigma_alpha
    self.sigma_decay = sigma_decay
    self.sigma_limit = sigma_limit
    self.sigma_max_change = sigma_max_change
    self.learning_rate = learning_rate
    self.learning_rate_decay = learning_rate_decay
    self.learning_rate_limit = learning_rate_limit
    self.popsize = popsize
    self.average_baseline = average_baseline
    if self.average_baseline:
      assert (self.popsize % 2 == 0), "Population size must be even"
      self.batch_size = int(self.popsize / 2)
    else:
      assert (self.popsize & 1), "Population size must be odd"
      self.batch_size = int((self.popsize - 1) / 2)
    self.elite_ratio = elite_ratio
    self.elite_popsize = int(self.popsize * self.elite_ratio)
    self.use_elite = False
    if self.elite_popsize > 0:
      self.use_elite = True
    self.forget_best = forget_best
    self.batch_reward = np.zeros(self.batch_size * 2)
    self.mu = np.zeros(self.num_params)
    self.sigma = np.ones(self.num_params) * self.sigma_init
    self.curr_best_mu = np.zeros(self.num_params)
    self.best_mu = np.zeros(self.num_params)
    self.best_reward = 0
    self.first_interation = True
    self.weight_decay = weight_decay
    self.rank_fitness = rank_fitness
    if self.rank_fitness:
      self.forget_best = True
    self.optimizer = Adam(self, learning_rate)

  # Making a method that returns the value of the standard deviation (sigma parameter) in the PEPG optimizer
  def rms_stdev(self):
    sigma = self.sigma
    return np.mean(np.sqrt(sigma*sigma))

  # Making a method that asks the PEPG optimizer to give us a set of candidate solutions
  def ask(self):
    self.epsilon = np.random.randn(self.batch_size, self.num_params) * self.sigma.reshape(1, self.num_params)
    self.epsilon_full = np.concatenate([self.epsilon, - self.epsilon])
    if self.average_baseline:
      epsilon = self.epsilon_full
    else:
      epsilon = np.concatenate([np.zeros((1, self.num_params)), self.epsilon_full])
    solutions = self.mu.reshape(1, self.num_params) + epsilon
    self.solutions = solutions
    return solutions

  # Making a method that gives the list of fitness results back to the PEPG optimizer
  def tell(self, reward_table_result):
    assert(len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."
    reward_table = np.array(reward_table_result)    
    if self.rank_fitness:
      reward_table = compute_centered_ranks(reward_table)    
    if self.weight_decay > 0:
      l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
      reward_table += l2_decay
    reward_offset = 1
    if self.average_baseline:
      b = np.mean(reward_table)
      reward_offset = 0
    else:
      b = reward_table[0]
    reward = reward_table[reward_offset:]
    if self.use_elite:
      idx = np.argsort(reward)[::-1][0:self.elite_popsize]
    else:
      idx = np.argsort(reward)[::-1]
    best_reward = reward[idx[0]]
    if (best_reward > b or self.average_baseline):
      best_mu = self.mu + self.epsilon_full[idx[0]]
      best_reward = reward[idx[0]]
    else:
      best_mu = self.mu
      best_reward = b
    self.curr_best_reward = best_reward
    self.curr_best_mu = best_mu
    if self.first_interation:
      self.sigma = np.ones(self.num_params) * self.sigma_init
      self.first_interation = False
      self.best_reward = self.curr_best_reward
      self.best_mu = best_mu
    else:
      if self.forget_best or (self.curr_best_reward > self.best_reward):
        self.best_mu = best_mu
        self.best_reward = self.curr_best_reward
    epsilon = self.epsilon
    sigma = self.sigma
    if self.use_elite:
      self.mu += self.epsilon_full[idx].mean(axis=0)
    else:
      rT = (reward[:self.batch_size] - reward[self.batch_size:])
      change_mu = np.dot(rT, epsilon)
      self.optimizer.stepsize = self.learning_rate
      update_ratio = self.optimizer.update(-change_mu)
    if (self.sigma_alpha > 0):
      stdev_reward = 1.0
      if not self.rank_fitness:
        stdev_reward = reward.std()
      S = ((epsilon * epsilon - (sigma * sigma).reshape(1, self.num_params)) / sigma.reshape(1, self.num_params))
      reward_avg = (reward[:self.batch_size] + reward[self.batch_size:]) / 2.0
      rS = reward_avg - b
      delta_sigma = (np.dot(rS, S)) / (2 * self.batch_size * stdev_reward)
      change_sigma = self.sigma_alpha * delta_sigma
      change_sigma = np.minimum(change_sigma, self.sigma_max_change * self.sigma)
      change_sigma = np.maximum(change_sigma, - self.sigma_max_change * self.sigma)
      self.sigma += change_sigma
    if (self.sigma_decay < 1):
      self.sigma[self.sigma > self.sigma_limit] *= self.sigma_decay
    if (self.learning_rate_decay < 1 and self.learning_rate > self.learning_rate_limit):
      self.learning_rate *= self.learning_rate_decay

  # Making a method that returns the current parameters
  def current_param(self):
    return self.curr_best_mu

  # Making a method that sets the parameters obtained by the PEPG optimizer into an array
  def set_mu(self, mu):
    self.mu = np.array(mu)

  # Making a method that returns the best set of optimized parameters
  def best_param(self):
    return self.best_mu

  # Making a method that returns the best parameters obtained so far, along with the historically best reward, the current reward, and the standard deviation sigma
  def result(self):
    return (self.best_mu, self.best_reward, self.curr_best_reward, self.sigma)
