import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from downsampling import *
from utils import *

class ElmapCVX(object):

    def __init__(self, demos, n=100, weighting='uniform', downsampling='naive', stretch=0.1, bend=0.01):
        self.demos = demos
        self.data = np.vstack(demos)
        self.n_pts_full, self.n_dims = np.shape(self.data)
        self.n_nodes = n
        self.n_size = self.n_nodes * self.n_dims
        
        self.num_trajs = len(self.demos)
        self.n_pts, self.n_dims = np.shape(self.demos[0])
        
        self.demos_stacked = np.reshape(self.data, (self.n_pts_full * self.n_dims, 1), order='F')
        
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
            
        self.nodes_stacked = np.reshape(self.nodes, (self.n_size, 1), order='F')
        
        #set up E & R matrices for optimization
        self.lmda = stretch
        self.mu = bend
            
        self.weight_sum = np.sum(self.weights)
    
       
        e1 = np.diag(-1*np.ones(self.n_nodes-1), -1)
        e2 = np.diag(np.ones(self.n_nodes))
        self.E = np.zeros((self.n_nodes, self.n_nodes))
        self.E += e1
        self.E += e2
        self.E[0, 0] = 0
        
        self.E_cvx = np.zeros((self.n_size, self.n_size))
        for i in range(self.n_dims):
            self.E_cvx[self.n_nodes*i : self.n_nodes*(i + 1), self.n_nodes*i : self.n_nodes*(i + 1)] = self.E
            
        r1 = 0.5*np.diag(np.ones(self.n_nodes-1))
        r2 = -1*np.diag(np.ones(self.n_nodes))
        self.R = np.zeros((self.n_nodes, self.n_nodes))
        self.R[1:,0:self.n_nodes-1] += r1
        self.R += r2
        self.R[:-1,1:self.n_nodes] += r1
        self.R[0, 1] = 1
        self.R[-1, -2] = 1
        
        self.R_cvx = np.zeros((self.n_size, self.n_size))
        for i in range(self.n_dims):
            self.R_cvx[self.n_nodes*i : self.n_nodes*(i + 1), self.n_nodes*i : self.n_nodes*(i + 1)] = self.R
        
        
        #change to see output of optimization
        self.DEBUG = False

    def set_weights(self, new_weights):
        self.weights = new_weights
        
    def set_nodes(self, new_nodes):
        self.nodes = new_nodes
        self.nodes_stacked = np.reshape(self.nodes, (self.n_size, 1), order='F')
        
    #cluster nodes (Expectation step)
    def associate(self):
        
        #association 
        
        #self.clusters = np.array([-1 for _ in range(self.n_pts)])
        #for i in range(self.n_pts):
        #    self.clusters[i] = np.argmin([np.linalg.norm(self.data[i] - self.nodes[j]) for j in range(self.n_nodes)])
        
        #alternative faster version, less readable:
        dist = (np.sum(self.data ** 2, axis=1,keepdims=1)+np.sum(self.nodes**2, axis=1,keepdims=1).T) - 2 * (self.data @ self.nodes.T) 
        self.clusters = np.argmin(dist, axis=1) 
        
        return self.clusters
    
    #incorporate any given constraints by giving high weight to those data points
    def add_consts(self, consts=[], inds=[]):
        self.consts = []
        for i in range(len(inds)):
            for j in range(self.n_dims):
                self.consts.append( cp.abs(self.x[inds[i] + (self.n_nodes * j)][0] - consts[i][j]) <= 0 )
        return
        
    def setup_problem(self):
        #setup optimization in cvxpy terms
        A = np.zeros((self.n_nodes*self.n_dims, self.n_nodes*self.n_dims))
        B = np.zeros((self.n_nodes*self.n_dims, self.n_pts_full*self.n_dims))
        for i in range(self.n_nodes):
            clustered_pts = self.clusters == i
            #print(clustered_pts)
            #print(self.weights * clustered_pts)
            for j in range(self.n_dims):
                B[i + self.n_nodes*j, (self.n_pts_full)*j:(self.n_pts_full)*(j+1)] = self.weights * clustered_pts
            for j in range(self.n_dims):
                A[i + self.n_nodes*j, i + self.n_nodes*j] = np.sum(B[i + self.n_nodes*j, :])
                
        self.x = cp.Variable((self.n_size, 1))
        self.x.value = self.nodes_stacked
        self.objective = cp.Minimize((1 / self.weight_sum) * cp.sum_squares(A @ self.x - B @ self.demos_stacked)  
                                + self.lmda * cp.sum_squares(self.E_cvx @ self.x)
                                + self.mu   * cp.sum_squares(self.R_cvx @ self.x))
        return
        
    def solve_problem(self):
        self.problem = cp.Problem(self.objective, self.consts)
        self.problem.solve(solver=cp.OSQP, warm_start=True, verbose=self.DEBUG, max_iter=1000000)
        
        if self.DEBUG:
            print("status:", self.problem.status)
            print("optimal value", self.problem.value)
            for i in range(len(self.consts)):
                print("dual value for constraint " + str(i), ": ", self.consts[i].dual_value)
            
        #update solution
        self.nodes = np.reshape(self.x.value, (self.n_nodes, self.n_dims), order='F')
        self.nodes_stacked = np.reshape(self.nodes, (self.n_size, 1), order='F')
        
        return self.nodes
    
    #optimize elastic map, return solution
    def optimize_map(self, consts=[], inds=[]):
    
        # EXPECTATION
        clusters = self.associate()
    
        # MAXIMIZATION
        self.setup_problem()
        self.add_consts(consts, inds)
        self.solve_problem()
        
        # REPEAT
        new_clusters = self.associate()
        
        iter = 1
        if self.DEBUG:
            print(iter, self.calc_nrg())
        
        #repeat EM until convergence or certain number of iterations
        while (np.any(new_clusters != clusters)) and (iter < 50):
            clusters = new_clusters
            
            self.setup_problem()
            self.add_consts(consts, inds)
            self.solve_problem()
            
            new_clusters = self.associate()
            
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
    
    elmap1 = ElmapCVX([traj], n=100, weighting='uniform', downsampling='distance', stretch=1.0, bend=10.0)
    new_traj = elmap1.optimize_map([np.array([x_demo[0], y_demo[0]]).flatten(), np.array([x_demo[-1], y_demo[-1]]).flatten()], [0, 100-1])
    
    elmap2 = ElmapCVX([traj], n=100, weighting='uniform', downsampling='distance', stretch=1.0, bend=10.0)
    new_traj2 = elmap2.optimize_map([np.array([x_demo[0]+0.5, y_demo[0]-0.2]).flatten(), np.array([x_demo[-1], y_demo[-1]]).flatten()], [0, 100-1])
    
    elmap3 = ElmapCVX([traj], n=100, weighting='uniform', downsampling='distance', stretch=1.0, bend=10.0)
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