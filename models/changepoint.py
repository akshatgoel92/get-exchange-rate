# Import packages
import numpy as np 
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

# Conventional tensorflow assignments
tfk = tfp.math.psd_kernels
tfd = tfp.distributions 
tfb = tfp.bijectors



def build_gp(amplitude, length_scale, observation_noise_variance):
  """
  -------------------------------------------
  Defines the conditional dist. of GP outputs, 
  given kernel parameters.
  -------------------------------------------
  """
  kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)

  # Create the GP prior distribution
  return(tfd.GaussianProcess(kernel=kernel, 
                             index_points=range(len(observation_noise_variance)),
                             observation_noise_variance=observation_noise_variance))


def main(): 
  """
  -------------------------------------------
  Defines the conditional dist. of GP outputs, 
  given kernel parameters.
  -------------------------------------------
  """
  gp_joint_model = tfd.JointDistributionNamed({
        'amplitude': tfd.LogNormal(loc=0., scale=np.float64(1.)),
        'length_scale': tfd.LogNormal(loc=0., scale=np.float64(1.)),
        'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(1.)),
        'observations': build_gp})
  
  x = gp_joint_model.sample()
  lp = gp_joint_model.log_prob(x)

  print("Sampled {}".format(x))
  print("Log probability of sample: {}".format(lp))


if __name__ == '__main__':

    main()