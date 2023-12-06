#weather model

import tensorflow_probability as tfp
import tensorflow as tf
'''1. Cold days are encoded by a 0 and hot days are encoded by a 1
2. The first day in the sequence has an 80% chance of being cold
3. A cold day has 30% chance of being followed by a hot day
4. A hot day has a 20% chance of being followed by a cold day
5. On each day the temperature is normally distributed with mean and standard deviation 0 and mean and standard deviation 15 and 10 on a hot day
'''
tfd = tfp.distributions 
initial_distribution = tfd.Categorical(probs=[0.8, 0.2])
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3], [0.2, 0.8]])
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])

model = tfd.HiddenMarkovModel(
    initial_distribution = initial_distribution,
    transition_distribution = transition_distribution,
    observation_distribution = observation_distribution,
    num_steps = 7)
mean = model.mean()

with tf.compat.v1.Session() as sess:
    print(mean.numpy())
    

