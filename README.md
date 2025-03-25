# ElasticMaps
 Implementations of Elastic Maps for Trajectory Learning from Demonstration.

This repository implements Elastic Maps, specifically polyline Elastic Maps for robot trajectories. Two versions are implemented, using a matrix-inversion least squares approach as well as a convex optimization approach. For further details and references please see the following papers:
"Robot Learning from Demonstration Using Elastic Maps" by B. Hertel, M. Pelland, and S. R. Ahmadzadeh, available [here](https://arxiv.org/abs/2208.02207).
"Confidence-Based Skill Reproduction Through Perturbation Analysis" by B. Hertel and S. R. Ahmadzadeh, available [here](https://arxiv.org/abs/2305.03091).

This repository implements the method described in the papers above using Python. Necessary libraries include [numpy](https://numpy.org/), [matplotlib](https://matplotlib.org/), and [cvxpy](https://www.cvxpy.org/index.html). Scripts which perform individual experiments are included, as well as other necessary utilities. If you have any questions, please contact Brendan Hertel (brendan_hertel@student.uml.edu).
