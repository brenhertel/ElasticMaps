import numpy as np
import matplotlib.pyplot as plt
from downsampling import *
from utils import *

class ElmapEM(object):

    def __init__(self, demos, n=100, weighting='uniform', downsampling='naive', stretch=0.1, bend=0.01):
        self.demos = demos
        self.data = np.vstack(demos)
        self.n_pts, self.n_dims = np.shape(self.data)
        self.n_nodes = n
        
        #set up weights
        self.weights = np.ones((self.n_pts,))
        if weighting == 'curvature':
            self.set_weights(curvature_weighting(self.demos))
        if weighting == 'jerk':
            self.set_weights(jerk_weighting(self.demos))
           
        #set up initial map
        self.nodes = downsample_traj(self.demos[0], self.n_nodes)
        if downsampling == 'distance':
            self.set_nodes(db_downsample(self.demos[0], self.n_nodes))
        if downsampling == 'douglaspeucker':
            self.set_nodes(DouglasPeuckerPoints(self.demos[0], self.n_nodes))
            
        #set up E & S matrices for optimization
        self.lmda = stretch
        self.mu = bend
            
        self.weight_sum = np.sum(self.weights)
    
        self.E = 2.*np.diag(np.ones((self.n_nodes,))) - np.diag(np.ones((self.n_nodes-1,)),1) - np.diag(np.ones((self.n_nodes-1,)),-1)
        self.E[0, 0] = 1
        self.E[-1, -1] = 1
        self.E = self.E * self.lmda
    
        self.S = 6.*np.diag(np.ones((self.n_nodes,))) - 4.*np.diag(np.ones((self.n_nodes-1,)),1) - 4.*np.diag(np.ones((self.n_nodes-1,)),-1) + np.diag(np.ones((self.n_nodes-2,)),2) + np.diag(np.ones((self.n_nodes-2,)),-2)
        self.S[0, 0] = 1
        self.S[0, 1] = -2
        self.S[1, 0] = -2
        self.S[1, 1] = 5
        self.S[-1, -1] = 1
        self.S[-2, -1] = -2
        self.S[-1, -2] = -2
        self.S[-2, -2] = 5
        self.S = self.S * self.mu / 4
        
        self.A = np.zeros((self.n_nodes, self.n_nodes))
        
        #change to see output of optimization
        self.DEBUG = False

    def set_weights(self, new_weights):
        self.weights = new_weights
        
    def set_nodes(self, new_nodes):
        self.nodes = new_nodes
        
    #cluster nodes (Expectation step)
    def associate(self):
        self.C = np.zeros((self.n_nodes, self.n_dims))
        
        #association & partial C matrix calculation
        
        #self.clusters = np.array([-1 for _ in range(self.n_pts)])
        #for i in range(self.n_pts):
        #    self.clusters[i] = np.argmin([np.linalg.norm(self.data[i] - self.nodes[j]) for j in range(self.n_nodes)])
        #    self.C[self.clusters[i]] = self.C[self.clusters[i]] + self.weights[i] * self.data[i]
        
        #alternative faster version, less readable:
        dist = (np.sum(self.data ** 2, axis=1,keepdims=1)+np.sum(self.nodes**2, axis=1,keepdims=1).T) - 2 * (self.data @ self.nodes.T) 
        self.clusters = np.argmin(dist, axis=1) 
        for i in range(self.n_pts):
            self.C[self.clusters[i]] = self.C[self.clusters[i]] + self.weights[i] * self.data[i]
        
        return self.clusters
    
    #calculate A matrix for optimization
    def calc_A(self):
        self.A = self.E + self.S
        for i in range(self.n_nodes):
            self.A[i, i] = self.A[i, i] + (np.sum(self.weights[self.clusters == i]) / self.weight_sum)
        return self.A
        
    #incorporate any given constraints by giving high weight to those data points
    def add_consts(self, consts=[], inds=[]):
        for i in range(len(inds)):
            self.nodes[inds[i]] = np.array(consts[i])
            self.data = np.vstack((self.data, np.array(consts[i]).reshape((1, self.n_dims))))
            self.weights = np.append(self.weights, np.array([np.size(self.data)]))
        self.weight_sum = np.sum(self.weights)
        self.n_pts, self.n_dims = np.shape(self.data)
        return
    
    #optimize elastic map, return solution
    def optimize_map(self, consts=[], inds=[]):
    
        self.add_consts(consts, inds)
    
        # EXPECTATION
        clusters = self.associate()
        C = self.C / self.weight_sum
        
        
        # MAXIMIZATION
        self.calc_A()
        self.nodes = np.linalg.lstsq(self.A, C, rcond=None)[0]
        
        # REPEAT
        new_clusters = self.associate()
        C = self.C / self.weight_sum #must recalculate C every time
        
        iter = 1
        if self.DEBUG:
            print(iter, self.calc_nrg())
        
        #repeat EM until convergence or certain number of iterations
        while (np.any(new_clusters != clusters)) and (iter < 50):
            clusters = new_clusters
            
            self.calc_A()
            self.nodes = np.linalg.lstsq(self.A, C, rcond=None)[0]
            
            new_clusters = self.associate()
            C = self.C / self.weight_sum
            
            iter = iter + 1
            if self.DEBUG:
                print(iter, self.calc_nrg())
            
        return self.nodes

    def calc_nrg(self): #Note: This is not for the optimization, just utility
        Uy = 0.
        for i in range(self.n_pts):
            Uy = Uy + self.weights[i] * np.linalg.norm(self.data[i] - self.nodes[self.clusters[i]])**2
            
        Ue = 0.
        for i in range(self.n_nodes - 1):
            Ue = Ue + np.linalg.norm(self.nodes[i] - self.nodes[i+1])**2
            
        Ur = 0.
        for i in range(self.n_nodes - 2):
            Ur = Ur + np.linalg.norm(self.nodes[i] - 2*self.nodes[i+1] + self.nodes[i+2])**2
            
        return (Uy / self.weight_sum) + (self.lmda * Ue) + (self.mu * Ur)

def main2D():
    # demonstration
    num_points = 1000
    t = np.linspace(0, 10, num_points).reshape((num_points, 1))
    x_demo = np.sin(t) + 0.01 * t**2 - 0.05 * (t-5)**2
    y_demo = np.cos(t) - 0.01 * t - 0.03 * t**2
    
    traj = np.hstack((x_demo, y_demo))
    
    elmap1 = ElmapEM([traj], n=100, weighting='uniform', downsampling='distance', stretch=0.01, bend=0.01)
    new_traj = elmap1.optimize_map([np.array([x_demo[0], y_demo[0]]).flatten(), np.array([x_demo[-1], y_demo[-1]]).flatten()], [0, 100-1])
    
    elmap2 = ElmapEM([traj], n=100, weighting='uniform', downsampling='distance', stretch=0.01, bend=0.01)
    new_traj2 = elmap2.optimize_map([np.array([x_demo[0]+0.5, y_demo[0]-0.2]).flatten(), np.array([x_demo[-1], y_demo[-1]]).flatten()], [0, 100-1])
    
    elmap3 = ElmapEM([traj], n=100, weighting='uniform', downsampling='distance', stretch=0.01, bend=0.01)
    new_traj3 = elmap3.optimize_map([np.array([x_demo[0]+0.5, y_demo[0]-0.2]).flatten(), np.array([x_demo[400]-0.1, y_demo[400]-0.3]).flatten(), np.array([x_demo[-1], y_demo[-1]+0.2]).flatten()], [0, 40, 100-1])
    
    plt.rcParams['figure.figsize'] = (6.5, 6.5)
    fig, axs = plt.subplots(2, 2)
    axs[0][0].set_title('Demonstration')
    axs[1][0].set_title('Same Constraints')
    axs[0][1].set_title('New Initial Point')
    axs[1][1].set_title('New Initial, Final and Viapoint')
    axs[0][0].plot(traj[:, 0], traj[:, 1], 'k', lw=3)
    axs[1][0].plot(traj[:, 0], traj[:, 1], 'k', lw=3)
    axs[0][1].plot(traj[:, 0], traj[:, 1], 'k', lw=3)
    axs[1][1].plot(traj[:, 0], traj[:, 1], 'k', lw=3)
    
    axs[1][0].plot(new_traj[:, 0], new_traj[:, 1], 'm', lw=3)
    axs[1][0].plot(x_demo[0], y_demo[0], 'k.', ms=10)
    axs[1][0].plot(x_demo[-1], y_demo[-1], 'k.', ms=10)
    axs[1][0].plot(x_demo[0], y_demo[0], 'rx', ms=10, mew=2)
    axs[1][0].plot(x_demo[-1], y_demo[-1], 'rx', ms=10, mew=2)
    
    axs[0][1].plot(new_traj2[:, 0], new_traj2[:, 1], 'g', lw=3)
    axs[0][1].plot(x_demo[0], y_demo[0], 'k.', ms=10)
    axs[0][1].plot(x_demo[-1], y_demo[-1], 'k.', ms=10)
    axs[0][1].plot(x_demo[0]+0.5, y_demo[0]-0.2, 'rx', ms=10, mew=2)
    axs[0][1].plot(x_demo[-1], y_demo[-1], 'rx', ms=10, mew=2)
    
    axs[1][1].plot(new_traj3[:, 0], new_traj3[:, 1], 'b', lw=3)
    axs[1][1].plot(x_demo[0], y_demo[0], 'k.', ms=10)
    axs[1][1].plot(x_demo[400], y_demo[400], 'k.', ms=10)
    axs[1][1].plot(x_demo[-1], y_demo[-1], 'k.', ms=10)
    axs[1][1].plot(x_demo[0]+0.5, y_demo[0]-0.2, 'rx', ms=10, mew=2)
    axs[1][1].plot(x_demo[400]-0.1, y_demo[400]-0.3, 'rx', ms=10, mew=2)
    axs[1][1].plot(x_demo[-1], y_demo[-1]+0.2, 'rx', ms=10, mew=2)
    
    plt.show()
    
if __name__ == '__main__':
    main2D()