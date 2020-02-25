"""
Can two-dimensional topological voids exist in two dimensions?
==============================================================

License: AGPLv3
^^^^^^^^^^^^^^^

"""


######################################################################
# The classic example of a two-dimensional homology class is the “void”
# surrounded by a sphere in three-dimensional space. Challenge question:
# **Can two-dimensional topological voids arise from point clouds in
# two-dimensional space?** We will answer this question programmatically
# by computing Vietoris-Rips persistence homology of random point clouds
# in the square :math:`[0, 1] \times [0, 1] \subset \mathbb{R}^2`.
# 

import numpy as np
from gtda.homology import VietorisRipsPersistence as VR
import itertools

import matplotlib.pyplot as plt

# Initializing the Vietoris-Rips transformer
vr = VR(homology_dimensions=(2,), max_edge_length=np.inf)
n_samples = 15000
n_points = 6

# Create n_samples point clouds of n_points points
PCS = np.random.random((n_samples, n_points, 2))  
# Compute persistence diagrams of all point clouds
DGMS = vr.fit_transform(PCS)  

diffs = np.nan_to_num(DGMS[:, :, 1] - DGMS[:, :, 0])  # Compute lifetimes
indices = np.argwhere(diffs != 0)  # Indices with non-zero lifetime
print('There are {} persistent homology classes in dimension 2 across all samples!'.format(len(indices[:, 0])))
print('There are {} different point clouds with at least one persistent homology class in dimension 2.'.format(len(np.unique(indices[:, 0]))))


######################################################################
# We can now plot the edges which exist when these persistent homology
# classes are born. What do the clique complexes of the resulting graphs
# remind you of?
# 

for i in indices[:, 0]:
    for e in itertools.combinations(PCS[i], 2):
        if np.linalg.norm(e[0] - e[1]) < DGMS[i, 0, 1] - 0.00001:
            edge = np.stack([e[0], e[1]])
            plt.plot(edge[:, 0], edge[:, 1])
    plt.show()

