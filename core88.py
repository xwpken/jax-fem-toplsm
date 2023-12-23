from dataclasses import dataclass
import numpy as onp
import jax
import jax.numpy as np

from jax_am.fem.generate_mesh import get_meshio_cell_type, Mesh
from jax_am.common import rectangle_mesh
from jax_am.fem.solver import solver

from jax_am.fem.core import FEM

import os
import time
from jax_am.fem.utils import save_sol

@dataclass
class opt88:
    
    cells: onp.ndarray
    points: onp.ndarray
    var: onp.ndarray
    dim: int
    ele_type: str = 'QUAD4'
    
    def __post_init__(self):
        
        self.num_cells = np.shape(self.cells)[0]
        self.num_nodes = np.shape(self.cells)[1]
        self.num_points = np.shape(self.points)[0]
        assert np.shape(self.points)[1]==self.dim
        
        self.num_cell_dofs = self.dim * self.num_nodes
        
        self.cell_vol, self.cell_sum_vol = self.get_cell_vol()
        
        self.vol = np.sum(self.cell_vol)
        
        self.get_cell_volfrac_helper()

    def get_cell_vol(self):

        cell_points = np.take(self.points, self.cells, axis=0)
        
        # (num_cells, num_nodes, dim) ---> (num_cells, 1)
        cell_vol = jax.vmap(self.get_ploy_area)(cell_points)

        cell_sum_vol = np.zeros(self.num_points)
        # (num_points, )
        cell_sum_vol = cell_sum_vol.at[self.cells.reshape(-1)].\
            add(np.repeat(cell_vol, self.num_nodes, axis=-1).reshape(-1))

        return cell_vol, cell_sum_vol
    

    def get_ploy_area(self,vertex_coos):
        """
        Calculate the area of convex ploygon
        Reference: https://en.wikipedia.org/wiki/Shoelace_formula
        """
        area = 0
        p = vertex_coos[-1]
        for q in vertex_coos:
            area += p[0] * q[1] - p[1] * q[0]
            p = q
        return abs(area) / 2
    
    
    def get_cell_volfrac_helper(self):
        
        if self.ele_type == 'QUAD4':
            
            sub_size = 20
            interval = np.linspace(-1,1,sub_size+1)
            self. sub_nodes = (sub_size+1)**2
            
            # (sub_nodes, sub_nodes)
            xi , eta = np.meshgrid(interval, interval)
            
            # (sub_nodes, sub_nodes, num_nodes) * (1, 1, num_nodes)
            prod_xi = 1 + (np.repeat(xi[:,:,None], self.num_nodes, axis = -1)\
                                       * np.array([-1,1,1,-1])[None,None,:])
                
            prod_eta = 1 + (np.repeat(eta[:,:,None], self.num_nodes, axis = -1)\
                                        * np.array([-1,-1,1,1])[None,None,:])
            
            shape_vals_ref = 1/4. * prod_xi * prod_eta
            
            self.shape_vals_ref = shape_vals_ref
            
        elif self.ele_type == 'TRI3':
            a = 1
        
        return None
    
    def get_cell_volfrac(self, var):
        
        if self.ele_type == 'QUAD4':
            cell_var = np.take(var, self.cells, axis=0)
            # (1, num_nodes, )
            sub_var = np.sum(self.shape_vals_ref[None,:,:,:] * cell_var[:,None,None,:],axis=-1)
            cell_volfrac = np.sum(sub_var>=0,axis=(1,2)) / self.sub_nodes
            
        elif self.ele_type == 'TRI3':
            cell_volfrac = 0

        elif self.ele_type  =='TET4':
            cell_volfrac = 0
            
        elif self.ele_type == 'HEX8':
            
            cell_volfrac = 0
        
        return cell_volfrac
    
    def get_volfrac(self, var):
        return np.sum(self.get_cell_volfrac(var) * self.cell_vol)/self.vol
    
    
    
    def point2cell(self, var_p):
        # points ---> cells
        return np.sum(np.take(var_p, self.cells, axis=0),axis=-1)\
                      *self.cell_vol/self.num_nodes
    
    
    def cell2point(self, var_c):
        # cells ---> points
        var_sum_p = np.zeros(self.num_points)
        # (num_points, )
        var_sum_p = var_sum_p.at[self.cells.reshape(-1)].\
            add(np.repeat(var_c, self.num_nodes, axis=-1).reshape(-1))
        
        return var_sum_p / self.cell_sum_vol
    
    
    def filter_MAF(self, var_p, nfilter):
        # Multi-average filter
        for i in range(0, nfilter):
            var_c = self.point2cell(var_p)
            var_p = self.cell2point(var_c)
        
        return var_p
    
    
    def get_init_phi(self):
        # Full material design
        return np.ones(self.num_points)
    
    def get_shape_sensitivity(self, cells_jac, sol, nfilter):
        
        def compliance(sol,jac):
            return sol.reshape(1,-1) @ jac.reshape(8,8) @ sol.reshape(-1,1)
        
        cells_sol = np.take(sol,self.cells,axis=0)
        cells_comp = jax.vmap(compliance)(cells_sol, cells_jac)
        vn = self.filter_MAF(self.cell2point(cells_comp), nfilter)
        
        return vn, onp.array(np.sum(cells_comp))
    
    

    
    
    def optimize(self, problem, optimizationParams):
        
        var = self.get_init_phi()
        
        loop = 0
        
        conv_flag = 0
        tol_obj = 1e-4
        tol_vol = 1e-3
        
        volfrac = optimizationParams['volfrac']
        loop_max = optimizationParams['maxIters']
        loop_relax = optimizationParams['relaxIters']
        nfilter = optimizationParams['nfilter']
        
        obj = onp.zeros((loop_max,1))
        vol = onp.zeros((loop_max,1))
        
        while loop < loop_max or conv_flag == 1:
            
            # forward analysis
            cell_volfrac = self.get_cell_volfrac(var)
            problem.set_params(cell_volfrac.reshape(-1,1))
            sol = solver(problem, linear=True, use_petsc=True)
            
            # shape sensitiivty analysis
            vn, obj[loop] = self.get_shape_sensitivity(problem.cells_jac, sol,nfilter)
            vol[loop] = self.get_volfrac(var)
            print(f'No,{loop+1}, obj:{obj[loop]}, vol:{vol[loop]}')
            # time.sleep(1)
            
            # Convergence check
            if loop > loop_relax - 1:
                change_obj = np.max(np.abs(obj[loop]-obj[loop-9:loop])/obj[loop])
                change_vol = np.abs(vol[loop]-volfrac)/volfrac
                if change_obj < tol_obj and change_vol < tol_vol:
                    conv_flag = 1
            else:
                volfrac_loop = vol[0] - (vol[0] - volfrac) * (loop + 1)/loop_relax
            
            # Design variables update
            dt = 0.5
            delta = 10
            mean_grad = 1.1
            
            index_delta = (np.abs(var) <= delta)
            delta_phi = onp.zeros(self.num_points)
            delta_phi[index_delta] = 0.75 / delta * \
                  (1 - var[index_delta]**2/(delta ** 2))
                  
            l1 = -1e4
            l2 =  1e4
            tol = 1e-4
            while (l2 - l1) > tol:
                lag = 0.5 * (l1 + l2)
                B = (vn / np.median(vn) - lag) * delta_phi * delta / 0.75
                temp_var = (var + dt * B) / mean_grad
                temp_volfrac = self.get_volfrac(temp_var)
                if temp_volfrac > volfrac_loop:
                    l1 = lag
                else:
                    l2 = lag
            
            var = temp_var     
            print(f'lagrange multiplier: {lag}')
            
            loop = loop + 1
            
        self.var = var
        self.vol = vol
        self.obj = obj
    
    


class LinearElasticity(FEM):
    
    def custom_init(self):
        """Override base class method.
        Modify self.flex_inds so that location-specific TO can be realized.
        """
        self.flex_inds = np.arange(len(self.cells))

    def get_tensor_map(self):

        def stress(u_grad, theta):
            E = 70e3
            Emin = E * 1e-6
            E = (E-Emin) * theta[0] + Emin
            nu = 0.3
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            sigma = lmbda * np.trace(epsilon) * np.eye(
                self.dim) + 2 * mu * epsilon
            return sigma

        return stress
    
    def set_params(self, params):
        """Override base class method.
        """
        full_params = np.ones((self.num_cells, params.shape[1]))
        full_params = full_params.at[self.flex_inds].set(params)
        thetas = np.repeat(full_params[:, None, :], self.num_quads, axis=1)
        self.full_params = full_params
        self.internal_vars['laplace'] = [thetas]

ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly = 4., 1.
meshio_mesh = rectangle_mesh(Nx=160, Ny=40, domain_x=Lx, domain_y=Ly)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def zero_dirichlet_val(point):
    return 0.

dirichlet_bc_info = [[left] * 2, [0, 1], [zero_dirichlet_val] * 2]

def load_location(point):
    return np.logical_and(np.isclose(point[0], Lx, atol=1e-5), np.isclose(point[1], 0., atol=0.1*Ly + 1e-5))

def neumann_val(point):
    return np.array([0., -100.])

neumann_bc_info = [[load_location], [neumann_val]]

problem = LinearElasticity(mesh, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, neumann_bc_info=neumann_bc_info)

var = np.ones(28)
var = var.at[1].set(-1.)

optimizer = opt88(mesh.cells, mesh.points, dim = 2, var = var)

optimizationParams = {'volfrac':0.5, 'maxIters':120, 'relaxIters':40, 'nfilter':4}

optimizer.optimize(problem, optimizationParams)

data_dir = os.path.join(os.path.dirname(__file__), 'opt')
vtk_path = os.path.join(data_dir, 'vtk/u.vtu')
save_sol(problem, optimizer.var, vtk_path)

from topo_tools import figure_yyaxis


var_dict = {'Compliance':onp.hstack((onp.linspace(1,120,120).reshape(-1,1), optimizer.obj)),
            'Volume fraction':onp.hstack((onp.linspace(1,120,120).reshape(-1,1), optimizer.vol)),}
figure_yyaxis(var_dict, labels={'ax': 'Iterations','ay1': 'Compliance','ay2': 'Volume fraction'}, yymode=1 ,save_path='fig.svg')



print('Finish!')