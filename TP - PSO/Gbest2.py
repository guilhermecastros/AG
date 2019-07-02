# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 20:58:53 2019

@author: Guilherme
"""

from __future__ import division
import random
import numpy as np

def sphere(x):                 # F.O. Sphere
  """
    In 3D: f(x,y,z) = x² + y² + z²
  """
  return np.sum(np.square(x))

def rastrigin(x):           # F.O. Rantringin
  soma = 0.0
  x = (5.12/100)*np.array(x) # Normalization
  for i in range(len(x)):
    soma = soma + x[i]**2 - 10*np.cos(2*np.pi*x[i])
  f = 10*len(x) + soma
  return f

#--- MAIN 
class Particle:
    def __init__(self,particle_dim):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.value_best_i=-1        # best value individual
        self.value_i=-1             # value individual

        # Initialize particle positions using a uniform distribution
        self.position_i = np.random.uniform(-100,100, size=(particle_dim))
        # Initialize particle velocities using a uniform distribution
        self.velocity_i = np.random.uniform(0, 0, size=(particle_dim))

    # evaluate current fitness
    def evaluate(self,costFunc):
        self.value_i = costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.value_i < self.value_best_i or self.value_best_i==-1:
            self.pos_best_i = self.position_i
            self.value_best_i = self.value_i

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        c1 = 2.05   # cognative constant
        c2 = 2.05   # social constant
        r1=random.random()
        r2=random.random()

        new_v = XX*(w*self.velocity_i + (c1 * r1 * (self.pos_best_i - self.position_i)) + (c2 * r2 * (pos_best_g - self.position_i)))
        self.velocity_i = new_v;

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        new_x = (self.position_i + self.velocity_i)
        self.position_i = new_x
        for i in range(0,num_dimensions):
            # adjust maximum position if necessary
            if self.position_i[i] > bounds[1]:
                self.position_i[i] = bounds[1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[0]:
                self.position_i[i] = bounds[0]
                
class PSO():
    def __init__(self):
        pass

    def optimize(self, costFunc, particle_dim, bounds, num_particles, maxiter):
        global num_dimensions

        num_dimensions = particle_dim
        value_best_g=-1                   # best value for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(particle_dim))

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,value_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].value_i < value_best_g or value_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    value_best_g=float(swarm[j].value_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        # print final results
        print('FINAL:')
        print(pos_best_g)
        print(value_best_g)

        return pos_best_g, value_best_g

if __name__ == "__PSO__":
    main()

#--- EXECUTE
# w (float): constant inertia weight (how much to weigh the previous velocity)
# XX (float): controls the exploration and exploitation of the swarm.
global w, XX
w = 0.5
XX = 0.5
particle_dim = 10
bounds=[-100,100]  # input bounds [min, max]
n_particles = 50
n_func_valuation = 100000
maxiter = int(n_func_valuation/n_particles)

results = [];
for i in range(31):
    #     func=sphere / func=rastrigin
    gbest = PSO();
    res_s = gbest.optimize(costFunc=rastrigin, particle_dim=particle_dim,
                           bounds=bounds, num_particles=n_particles, maxiter=maxiter)
    results.append(res_s[1]);
    print(f'x = {res_s[0]}') # x = [0 0 0]
    print(f'f = {res_s[1]}') # f = 0

#results = np.around(results, decimals=2)
mean = np.mean(results)
std = np.std(results)