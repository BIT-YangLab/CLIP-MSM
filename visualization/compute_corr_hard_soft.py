import numpy as np
import scipy.stats as stats

# This is an example
soft = np.load('YourPath/soft-dissect-score_bodies.npy')
hard = np.load('YourPath/hard-dissect-score_bodies.npy')
floc = np.load('YourPath/floc_beta_value_bodies.npy')
pearson_corr = np.corrcoef(soft, floc)[0,1]
