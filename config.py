import numpy as np

width = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 4, 4, 4, 4, 4])
centers = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 5, 13, 21, 29, 37])
nbins = len(centers)

hist_by_prof_edges = [0, 0.2, 0.4, 0.6, 0.8, 1] # 15, 30]
use_dropout = True
use_hist = False #True
concat_hist_max = False #True