
static char help[] = "Solves a linear system in parallel with KSP. Modified from ex2.c \n\
                      Illustrate how to use external packages MUMPS, SUPERLU and STRUMPACK \n\
Input parameters include:\n\
  -m <mesh_x>       : number of mesh points in x-direction\n\
  -n <mesh_y>       : number of mesh points in y-direction\n\n";

#include <petscksp.h>
#include <math.h>
#include <time.h>
#include <omp.h>

int main(int argc, char **args)
{
  Vec         c, f, l, u, f0; /* approx solution, RHS, exact solution */
  Mat         J, K, M;
  KSP         ksp; /* linear solver context */
  PC          pc;
  PetscReal   c0 = 1.0, inp = 0.1, Lx = 20.0, Ly = 20.0, W = 12.0, h, k, x0;
  PetscReal   dt, kap = 3.0, norm, tol = 1e-12;
  PetscInt    nt = 1, sind, find, n_wall, pos, iter, max_iter = 10, lr;
  PetscInt    i, j, t, Ii, m = 16, n = 16, ncol, nloc;
  PetscBool   flg = PETSC_FALSE, flg_ilu = PETSC_FALSE, verbose = PETSC_FALSE;
#if defined(PETSC_HAVE_MUMPS)
  PetscBool flg_mumps = PETSC_FALSE;
#endif
#if defined(PETSC_HAVE_SUPERLU) || defined(PETSC_HAVE_SUPERLU_DIST)
  PetscBool flg_superlu = PETSC_FALSE;
#endif
  PetscReal vals[16], grads[3], prods[9], coefs[6], locs[3], sum, *r_ghst = NULL, *f_vals = NULL, *myvals = NULL;
  PetscInt  inds[4], *f_rows = NULL;
  PetscMPIInt rank, size;
#if defined(PETSC_USE_LOG)
  PetscLogStage stage;
#endif
  clock_t t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17;
  PetscReal t_loc[9], t_tmp;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nt", &nt, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-v", &verbose, NULL));

  t1 = clock();

  dt = 0.1*Lx*Ly/(n*m);
  h = Lx / (m-1);
  k = Ly / (n-1);
  n_wall = (PetscInt) ((1.0 - W/Ly)*(n-1)/2);
  nloc = 2 * m * n;
  ncol = 2 * m;
  f_rows = (PetscInt*)  malloc( ncol * sizeof(PetscInt));
  f_vals = (PetscReal*) malloc( ncol * sizeof(PetscReal));
  for (i = 0; i < ncol; ++i) {
    f_rows[i] = n*i;
    f_vals[i] = -h*inp;
  }
  f_vals[0] = -0.5*h*inp;
  f_vals[ncol-1] = -0.5*h*inp;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &J));
  PetscCall(MatSetSizes(J, nloc, nloc, 2*m*n, 2*m*n));
  PetscCall(MatSetFromOptions(J));
  PetscCall(MatMPIAIJSetPreallocation(J, 7, NULL, 3, NULL));
  PetscCall(MatSetOption(J, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &K));
  PetscCall(MatSetSizes(K, nloc, nloc, 2*m*n, 2*m*n));
  PetscCall(MatSetFromOptions(K));
  PetscCall(MatMPIAIJSetPreallocation(K, 7, NULL, 3, NULL));
  PetscCall(MatSetOption(K, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &M));
  PetscCall(MatSetSizes(M, nloc, nloc, 2*m*n, 2*m*n));
  PetscCall(MatSetFromOptions(M));
  PetscCall(MatMPIAIJSetPreallocation(M, 7, NULL, 3, NULL));
  PetscCall(MatSetOption(M, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));

  /*
     Set matrix elements for the 2-D, five-point stencil in parallel.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly).
      - Always specify global rows and columns of matrix entries.

     Note: this uses the less common natural ordering that orders first
     all the unknowns for x = h then for x = 2h etc; Hence you see J = Ii +- n
     instead of J = I +- m as you might expect. The more standard ordering
     would first do all variables for y = h, then y = 2h etc.

   */

  PetscCall(PetscLogStageRegister("Assembly", &stage));
  PetscCall(PetscLogStagePush(stage));

  grads[0] =  0.0;               grads[1] = -1.0/k;                 grads[2] =  1.0/k;
  prods[0] =  1.0/(h*h);         prods[1] = -1.0/(h*h);             prods[2] =  0.0;
  prods[3] = -1.0/(h*h);         prods[4] =  1.0/(h*h) + 1.0/(k*k); prods[5] = -1.0/(k*k);
  prods[6] =  0.0;               prods[7] = -1.0/(k*k);             prods[8] =  1.0/(k*k);
  coefs[0] =  2.0*h/Lx;          coefs[1] =  1.0*h/Lx;              coefs[2] =  1.0*h/Lx;
  coefs[3] = -0.6*h*h/Lx/Lx;     coefs[4] = -0.2*h*h/Lx/Lx;         coefs[5] = -0.2*h*h/Lx/Lx;

  // Locals, no ghosts needed
  for (t = 0; t < 2; t++) {
    if (t==0) {
      sind = 0;
      find = (n-1)*(m-1);
      lr = 1;
    } else {
      sind = (n-1)*m;
      find = (n-1)*(2*m-1);
      lr = 0;
    }
    for (Ii = sind; Ii < find; Ii++) {
      inds[0] = Ii + Ii / (n-1);
      inds[1] = inds[0] + n;
      inds[2] = inds[1] + 1;
      x0 = ((PetscReal) (inds[1]/n))/(m-1);
      for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
          vals[3*i + j] = -(1.0 + (i==j))/24.0 * h * k / dt;
        }
      }
      PetscCall(MatSetValues(M, 3, inds, 3, inds, vals, ADD_VALUES));
      for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
          vals[3*i + j] = -vals[3*i + j] + lr * grads[j] * 0.5*h*k * ( 2.0 * x0*(1-x0) + coefs[i]*(x0-0.5) + coefs[3+i]);
        }
      }
      PetscCall(MatSetValues(K, 3, inds, 3, inds, vals, ADD_VALUES));
      inds[1] = inds[0];
      inds[0] = inds[2];
      inds[2] = inds[1];
      inds[1] = inds[2] + 1;
      x0 = ((PetscReal) (inds[1]/n))/(m-1);
      for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
          vals[3*i + j] = -(1.0 + (i==j))/24.0 * h * k / dt;
        }
      }
      PetscCall(MatSetValues(M, 3, inds, 3, inds, vals, ADD_VALUES));
      for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
          vals[3*i + j] = -vals[3*i + j] - lr * grads[j] * 0.5*h*k * ( 2.0 * x0*(1-x0) - coefs[i]*(x0-0.5) + coefs[3+i]);
        }
      }
      PetscCall(MatSetValues(K, 3, inds, 3, inds, vals, ADD_VALUES));
    }
    PetscCall(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY));
  }

  for (Ii = n*(m-1) + n_wall; Ii < n*m-1 - n_wall; ++Ii) {
    inds[0] = Ii;
    inds[1] = inds[0] + 1;
    inds[2] = inds[0] + n;
    inds[3] = inds[2] + 1;
    for (i = 0; i < 4; ++i){
      for (j = 0; j < 4; ++j){
        vals[4*i + j] = kap * ((i<2 && j<2 || i>1 && j>1) ? 1 : -1) * (1.0 + ((3+i-j) % 2))/6.0*k;
      }
    }
  PetscCall(MatSetValues(K, 4, inds, 4, inds, vals, ADD_VALUES));
  }

  t2 = clock();
  t_loc[0] = ((PetscReal) (t2-t1)) / CLOCKS_PER_SEC;

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  PetscCall(MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscLogStagePop());

  /* Create parallel vectors */
  PetscCall(MatCreateVecs(K, &c, &f));
  PetscCall(VecDuplicate(c, &u));
  PetscCall(VecDuplicate(f, &f0));
  PetscCall(VecDuplicate(f, &l));


  /*
     Set exact solution; then compute right-hand-side vector.
     By default we use an exact solution of a vector with all
     elements of 1.0;  Alternatively, using the runtime option
     -random_sol forms a solution vector with random components.
  */
  PetscCall(VecSet(c, c0));
  PetscCall(VecSetValues(f0, ncol, f_rows, f_vals, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(f0));
  PetscCall(VecAssemblyEnd(f0));

  t3 = clock();
  t_loc[1] = ((PetscReal) (t3-t2)) / CLOCKS_PER_SEC;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create linear solver context
  */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, K, K));

  /*
    Example of how to use external package MUMPS
    Note: runtime options
          '-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps -mat_mumps_icntl_7 3 -mat_mumps_icntl_1 0.0'
          are equivalent to these procedural calls
  */
#if defined(PETSC_HAVE_MUMPS)
  flg_mumps    = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_mumps_lu", &flg_mumps, NULL));
  if (flg_mumps) {
    PetscCall(KSPSetType(ksp, KSPPREONLY));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCLU));
    PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS));
    PetscCall(PCFactorSetUpMatSolverType(pc)); /* call MatGetFactor() to create F */
  }

#endif

  /*
    Example of how to use external package SuperLU
    Note: runtime options
          '-ksp_type preonly -pc_type ilu -pc_factor_mat_solver_type superlu -mat_superlu_ilu_droptol 1.e-8'
          are equivalent to these procedual calls
  */
#if defined(PETSC_HAVE_SUPERLU) || defined(PETSC_HAVE_SUPERLU_DIST)
  flg_ilu     = PETSC_FALSE;
  flg_superlu = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_superlu_lu", &flg_superlu, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_superlu_ilu", &flg_ilu, NULL));
  if (flg_superlu || flg_ilu) {
    PetscCall(KSPSetType(ksp, KSPPREONLY));
    PetscCall(KSPGetPC(ksp, &pc));
    if (flg_superlu) PetscCall(PCSetType(pc, PCLU));
    else if (flg_ilu) PetscCall(PCSetType(pc, PCILU));
    if (size == 1) {
  #if !defined(PETSC_HAVE_SUPERLU)
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "This test requires SUPERLU");
  #else
      PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERSUPERLU));
  #endif
    } else {
  #if !defined(PETSC_HAVE_SUPERLU_DIST)
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "This test requires SUPERLU_DIST");
  #else
      PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERSUPERLU_DIST));
  #endif
    }
    PetscCall(PCFactorSetUpMatSolverType(pc)); /* call MatGetFactor() to create F */
  }
#endif

  /*
    Example of how to use procedural calls that are equivalent to
          '-ksp_type preonly -pc_type lu/ilu -pc_factor_mat_solver_type petsc'
  */
  flg     = PETSC_FALSE;
  flg_ilu = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_petsc_lu", &flg, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_petsc_ilu", &flg_ilu, NULL));
  if (flg || flg_ilu) {

    PetscCall(KSPSetType(ksp, KSPPREONLY));
    PetscCall(KSPGetPC(ksp, &pc));
    if (flg) PetscCall(PCSetType(pc, PCLU));
    else if (flg_ilu) PetscCall(PCSetType(pc, PCILU));
    PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERPETSC));
    PetscCall(PCFactorSetUpMatSolverType(pc)); /* call MatGetFactor() to create F */

  }

  PetscCall(KSPSetFromOptions(ksp));

  /* Get info from matrix factors */
  PetscCall(KSPSetUp(ksp));

  t4 = clock();
  t_loc[2] = ((PetscReal) (t4-t3)) / CLOCKS_PER_SEC;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  for (t = 0; t < nt; ++t) {

    t5 = clock();

    PetscCall(MatMultAdd(M, c, f0, l));

    t6 = clock();
    t_loc[3] += ((PetscReal) (t6-t5)) / CLOCKS_PER_SEC;

    for (iter = 0; iter < max_iter; ++iter) {

      if (verbose) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));

      t7 = clock();
      PetscCall(MatDuplicate(K, MAT_COPY_VALUES, &J));
      t8 = clock();
      t_loc[4] += ((PetscReal) (t8-t7)) / CLOCKS_PER_SEC;
      PetscCall(MatMultAdd(K, c, l, f));
      PetscCall(VecGetArray(c, &myvals));
      t9 = clock();
      t10 = clock();
      t_loc[5] += ((PetscReal) (t10-t9)) / CLOCKS_PER_SEC;
  
      for (Ii = 0; Ii < (n-1)*(2*m-1); Ii++) {
        if (Ii/(n-1) == m-1) continue;
        inds[0] = Ii + Ii / (n-1);
        inds[1] = inds[0] + n;
        inds[2] = inds[1] + 1;
        for (i = 0; i < 12; ++i) {
          vals[i] = 0;
        }
        for (i = 0; i < 3; ++i) {
          locs[i] = myvals[inds[i]];
        }
        sum = locs[0] + locs[1] + locs[2];
        for (i = 0; i < 3; ++i) {
          for (j = 0; j < 3; ++j) {
            vals[12     ]  = prods[3*i + j] * h * k / 6.0;
            vals[3*i + j] += vals[12] * sum;
            vals[12     ] *= locs[j];
            vals[3*i    ] += vals[12];
            vals[3*i + 1] += vals[12];
            vals[3*i + 2] += vals[12];
            vals[9 + i  ] += vals[12] * sum;
          }
        }
        PetscCall(VecSetValues(f, 3, inds, vals + 9, ADD_VALUES));
        PetscCall(MatSetValues(J, 3, inds, 3, inds, vals, ADD_VALUES));
        inds[1] = inds[0];
        inds[0] = inds[2];
        inds[2] = inds[1];
        inds[1] = inds[2] + 1;
        for (i = 0; i < 12; ++i) {
          vals[i] = 0;
        }
        for (i = 0; i < 3; ++i){
          locs[i] = myvals[inds[i]]; 
        }
        sum = locs[0] + locs[1] + locs[2];
        for (i = 0; i < 3; ++i) {
          for (j = 0; j < 3; ++j) {
            vals[12     ]  = prods[3*i + j] * h * k / 6.0;
            vals[3*i + j] += vals[12] * sum;
            vals[12     ] *= locs[j];
            vals[3*i    ] += vals[12];
            vals[3*i + 1] += vals[12];
            vals[3*i + 2] += vals[12];
            vals[9 + i  ] += vals[12]*sum;
          }
        }
        PetscCall(VecSetValues(f, 3, inds, vals + 9, ADD_VALUES));
        PetscCall(MatSetValues(J, 3, inds, 3, inds, vals, ADD_VALUES));
      }

      t_tmp = 0;
      t13 = clock();
      t_loc[5] += t_tmp;
      t_loc[6] += ((PetscReal) (t13-t10)) / CLOCKS_PER_SEC - t_tmp;

      PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
      PetscCall(VecAssemblyBegin(f));
      PetscCall(VecRestoreArray(c, &myvals));
      PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
      PetscCall(VecAssemblyEnd(f));

      t14 = clock();
      t_loc[7] += ((PetscReal) (t14-t13)) / CLOCKS_PER_SEC;

      PetscCall(VecNorm(f, NORM_2, &norm));
      t15 = clock();
      t_loc[3] += ((PetscReal) (t15-t14)) / CLOCKS_PER_SEC;
      if (verbose) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Step: %2d, Iteration: %1d, Rnorm: %e", t, iter, norm));
      if (norm < tol) break;

      PetscCall(KSPSetOperators(ksp, J, J));
      PetscCall(KSPSolve(ksp, f, u));
      t16 = clock();
      t_loc[8] += ((PetscReal) (t16-t15)) / CLOCKS_PER_SEC;

      PetscCall(VecAXPY(c, -1.0, u));
      PetscCall(VecNorm(u, NORM_2, &norm));
      t17 = clock();
      t_loc[3] += ((PetscReal) (t17-t16)) / CLOCKS_PER_SEC;
      if (verbose) PetscCall(PetscPrintf(PETSC_COMM_WORLD, ", Unorm: %e", norm));
      if (norm < tol) break;

    }

    if (iter == max_iter) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nDid not converge!"));
    if (verbose) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nStep: %2d, Iterations: %1d\n", t, iter));

  }

  if (verbose) PetscCall(VecView(c, PETSC_VIEWER_STDOUT_WORLD));

  PetscReal f_tot = t_loc[0] + t_loc[1] + t_loc[2] + t_loc[3] + t_loc[4] + t_loc[5] + t_loc[6] + t_loc[7] + t_loc[8];
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Time to compute initial values: %e\n", t_loc[0]));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Time to assemble mats and vecs: %e\n", t_loc[1]));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Time to initialize solver obj.: %e\n", t_loc[2]));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Time to perform vec operations: %e\n", t_loc[3]));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Time to duplicate matrix for J: %e\n", t_loc[4]));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Time to comm and wait for msgs: %e\n", t_loc[5]));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Time to compute values in loop: %e\n", t_loc[6]));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Time to assemble Newton matrix: %e\n", t_loc[7]));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Time to solve Newton iteration: %e\n", t_loc[8]));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "\nTOTAL TIME                    : %e\n", f_tot));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  free(r_ghst);
  free(f_vals);
  free(f_rows);
  free(myvals);
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&c));
  PetscCall(VecDestroy(&f));
  PetscCall(VecDestroy(&l));
  PetscCall(VecDestroy(&f0));
  PetscCall(MatDestroy(&J));
  PetscCall(MatDestroy(&M));
  PetscCall(MatDestroy(&K));

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
  PetscCall(PetscFinalize());
  return 0;
}
