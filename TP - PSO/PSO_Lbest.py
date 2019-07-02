# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 19:24:56 2019

@author: Guilherme
"""

import numpy as np

class PSO(object):
  """
    Class implementing PSO algorithm.
  """
  def __init__(self, func, particle_dim, n_particles, init_space, end_space, XX, w):
    """
      Initialize the key variables.
      Args:
        func (function): the fitness function to optimize.
        n_particles (int): the number of particles of the swarm.
        particle_dim (int): dimension of particles.
        init_space (int): space initial value.
        end_space (int): space final value.
        w (float): the inertia weight
        XX (float): controls the exploration and exploitation of the swarm.
    """
    self.func = func
    self.n_particles = n_particles
    self.particle_dim = particle_dim
    # Initialize particle positions using a uniform distribution
    self.particles_pos = np.random.uniform(init_space, end_space, size=(n_particles, self.particle_dim))
    # Initialize particle velocities using a uniform distribution
    self.velocities = np.random.uniform(0, 0, size=(n_particles, self.particle_dim))

    # Initialize the best positions
    self.g_best = self.particles_pos
    self.p_best = self.particles_pos
    
    self.XX = XX
    self.w = w
    
    self.init_space = init_space
    self.end_space = end_space

  def update_position(self, x, v):
    """
      Update particle position.
      Args:
        x (array-like): particle current position.
        v (array-like): particle current velocity.
      Returns:
        The updated position (array-like).
    """
    x = np.array(x)
    v = np.array(v)
    new_x = np.zeros(len(x));
    
    for i in range(len(x)):
        if((x[i] + v[i]) > self.end_space):
            new_x[i] = (x[i] + v[i]) - 2*(x[i] + v[i] - self.end_space)
        elif((x[i] + v[i]) < self.init_space):
            new_x[i] = (x[i] + v[i]) - 2*((x[i] + v[i]) + self.end_space)
            #print("xi: " + str(x[i]) + " vi: " + str(v[i]))
            #print("new_x[i]: " + str(new_x[i]))
        else:
            new_x[i] = (x[i] + v[i])
    return new_x;

  def update_velocity(self, x, v, p_best, g_best, XX, w, c1=2.05, c2=2.05):
    """
      Update particle velocity.
      Args:
        x (array-like): particle current position.
        v (array-like): particle current velocity.
        p_best (array-like): the best position found so far for a particle.
        g_best (array-like): the best position regarding
                             all the particles found so far.
        c1 (float): the cognitive scaling constant.
        c2 (float): the social scaling constant.
        w (float): the inertia weight
        XX (float): controls the exploration and exploitation of the swarm.
      Returns:
        The updated velocity (array-like).
    """
    x = np.array(x)
    v = np.array(v)
    assert x.shape == v.shape, 'Position and velocity must have same shape'
    # a random number between 0 and 1.
    r1 = np.random.uniform()
    r2 = np.random.uniform()
    p_best = np.array(p_best)
    g_best = np.array(g_best)

    new_v = XX*(w*v + (c1 * r1 * (p_best - x)) + (c2 * r2 * (g_best - x)))
    return new_v

  def local_best(self, index):
    neighbors = [];
    neighbors_index = [];

    if index == 0:
        neighbors_index = [self.n_particles - 1, index, index + 1]
        neighbors = [self.p_best[self.n_particles - 1], self.p_best[index], self.p_best[index + 1]]
    elif (index == (self.n_particles - 1)):
        neighbors_index = [index - 1, index, 0]
        neighbors = [self.p_best[index - 1], self.p_best[index], self.p_best[0]]
    else:
        neighbors_index = [index - 1, index, index + 1]
        neighbors = [self.p_best[index - 1], self.p_best[index], self.p_best[index + 1]]
        
    min_index = np.argmin([self.func(neighbors[0]), self.func(neighbors[1]), self.func(neighbors[2])]);
    self.g_best[i] = self.p_best[neighbors_index[min_index]]
    return

  def optimize(self, maxiter=200):
    """
      Run the PSO optimization process untill the stoping criteria is met.
      Case for minimization. The aim is to minimize the cost function.
      Args:
          maxiter (int): the maximum number of iterations before stopping
                         the optimization.
      Returns:
          The best solution found (array-like).
    """
    gBest = [];
    gBestValue = 999999999;

    for _ in range(maxiter):
      for i in range(self.n_particles):
          x = self.particles_pos[i]
          v = self.velocities[i]
          p_best = self.p_best[i]
          self.local_best(i);
          self.velocities[i] = self.update_velocity(x, v, p_best, self.g_best[i], XX=self.XX, w=self.w)
          self.particles_pos[i] = self.update_position(x, v)
          # Update the best position for particle i
          if self.func(self.particles_pos[i]) < self.func(p_best):
              self.p_best[i] = self.particles_pos[i]
          # Update the best position overall
          if self.func(self.particles_pos[i]) < self.func(self.g_best[i]):
              self.g_best[i] = self.particles_pos[i]
    for i in range(self.n_particles):
        if self.func(self.g_best[i]) < gBestValue:
            gBestValue = self.func(self.g_best[i])
            gBest = self.g_best[i]

    return gBest, gBestValue
  
def sphere(x):                 # F.O. Sphere
  """
    In 3D: f(x,y,z) = x² + y² + z²
  """
  return np.sum(np.square(x))

def rastrigin(x):           # F.O. Rantringin
  soma = 0.0
  for i in range(len(x)):
    soma = soma + x[i]**2 - 10*np.cos(2*np.pi*x[i])
  f = 10*len(x) + soma
  return f

if __name__ == '__main__':
  results = [];
  for i in range(31):
    n_particles = 50
    n_func_valuation = 100000
    maxiter = int(n_func_valuation/n_particles)
    #     func=sphere / func=rastrigin
    PSO_s = PSO(func=rastrigin, particle_dim=10, n_particles=n_particles, init_space=-100, end_space=100, XX=0.5, w=1)
    res_s = PSO_s.optimize(maxiter=maxiter)
    results.append(res_s[1]);
    print(f'x = {res_s[0]}') # x = [0 0 0]
    print(f'f = {res_s[1]}') # f = 0
    
mean = np.mean(results)
std = np.std(results)