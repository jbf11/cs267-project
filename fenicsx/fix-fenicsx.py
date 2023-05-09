#!/usr/bin/env python
from mpi4py import MPI

# Initialize MPI section
mesh_comm = MPI.COMM_WORLD  # distribute built mesh

from petsc4py import PETSc

import ufl
import numpy as np

from dolfinx import mesh, fem, io, nls, log, cpp, plot, geometry

import time

start = time.time()

log.set_log_level(log.LogLevel.WARNING)

# Define mesh
Lx = 20
Ly = 20
h = 6
nx, ny = 160, 160
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([Lx, Ly])], 
                               [nx, ny], mesh.CellType.triangle)

num_steps = 10

# define element and function space
P = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
U = fem.FunctionSpace(domain, ufl.MixedElement([P, P]))

u = fem.Function(U)
cl, cr = ufl.split(u)
dcl, dcr = ufl.TestFunctions(U)
cl_pos, cr_pos = 0, 1

CL, CL_map = U.sub(cl_pos).collapse()
CR, CR_map = U.sub(cr_pos).collapse()

tol = 1e-8
boundaries = [(1, lambda x: np.logical_and(np.isclose(x[0], Lx), np.abs(x[1]-Ly/2) < h+tol)),
              (2, lambda x: np.logical_and(np.isclose(x[0], 0),  np.abs(x[1]-Ly/2) < h+tol)),
              (3, lambda x: np.isclose(x[1], 0))]

facet_indices, facet_markers = [], []
fdim = domain.topology.dim - 1
for (marker, locator) in boundaries:
    facets = mesh.locate_entities(domain, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = mesh.meshtags(domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)

# Functions for initial condition and previous time step
cl0 = fem.Function(CL)
cr0 = fem.Function(CR)

DLT = fem.Constant(domain, PETSc.ScalarType(0.1))
NUM = fem.Constant(domain, PETSc.ScalarType(0.1))
INP = fem.Constant(domain, PETSc.ScalarType(0.1))

# define variational form for each domain

x = ufl.SpatialCoordinate(domain)
v = 6/Lx**2 * x[0] * (Lx - x[0])

Fl = ( (cl-cl0)/DLT*dcl + cl * ufl.inner(ufl.grad(cl), ufl.grad(dcl)) + v * cl.dx(1)*dcl ) * ufl.dx
Fr = ( (cr-cr0)/DLT*dcr + cr * ufl.inner(ufl.grad(cr), ufl.grad(dcr))                ) * ufl.dx
Fs = NUM*dcl*ds(1) - NUM*dcr*ds(2) - INP*dcl*ds(3)
F = Fl + Fr + Fs


problem = fem.petsc.NonlinearProblem(F, u, bcs=[])

solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-12
solver.report = True
solver.max_it = 10

#"""
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "gmres"#"preonly"
opts[f"{option_prefix}pc_type"] = "asm"#"lu"
#opts[f"{option_prefix}pc_factor_mat_solver_type"] = "superlu_dist"
ksp.setFromOptions()
#"""

# Define temporal parameters
dt = 0.1*Lx*Ly/((nx+1)*(ny+1))
T = dt * num_steps
DLT.value = dt

# Initial condition - ph, cp, cm are DEPARTURE variables
c_int = 1.0

NUM.value = 0.15
INP.value = 0.1

u.x.array[:] = c_int
u.x.scatter_forward()


for i in range(num_steps):
    
    cl0.x.array[:] = u.x.array[CL_map]
    cr0.x.array[:] = u.x.array[CR_map]
    cl0.x.scatter_forward()
    cr0.x.scatter_forward()
    
    _, converged = solver.solve(u)
    assert(converged)

end = time.time()

print(str(mesh_comm.rank) + ": Success! Size of u: " + str(U.dofmap.index_map.size_local) + ", " + str(U.dofmap.index_map.num_ghosts) + 
      "\nTime: " + str(end-start))
