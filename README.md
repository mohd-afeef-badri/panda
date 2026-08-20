# <img width="100" align="left" src="./logo/PANDA_white.png" /> [PANDA: Polytopal Algorithm for Numerical Discretization and Analysis](https://github.com/mohd-afeef-badri/panda)

PANDA is a research-oriented numerical framework for solving partial differential equations using **polytopal methods** on **general polyhedral meshes**. The code is designed to work directly with *real polytopal meshes* and supports problems such as **Poisson**, **Stokes**....

## Features

- Polytopal discretization methods
- Support for arbitrary polyhedral meshes
- Solvers for:
  - Poisson equation
  - Stokes equations (and extensions)
- Direct reading of **MED mesh format**
- Output in **VTK** format for visualization
- Lightweight dependency model

## Dependencies

PANDA has a **single external dependency**:

- **MedCoupling** (from the SALOME platform)

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/mohd-afeef-badri/panda.git
````

### 2. Download MedCoupling

Download a [**MedCoupling binary distribution**](https://www.salome-platform.org/?page_id=2430) (standalone or via SALOME) compatible with your system. Simply untar/unzip at `your/MEDCOUPLING_INSTALL_LOCATION`

---

## Running the Code

Before running PANDA, you must initialize the SALOME/MedCoupling environment.

### 1. Load the MedCoupling environment

```bash
source /your/MEDCOUPLING_INSTALL_LOCATION/env_launch.sh
```

### 2. Run an example solver

#### Poisson problem

```bash
cd panda/poisson
python main.py
```

The solver:

* Reads a **polytopal mesh in MED format** in the `panda/poisson/mesh` folder
* Assembles and solves the discrete system
* Writes results to **VTK files**
* Results can be visualized using **ParaView**

Example:

```bash
paraview panda/poisson/solution/solution.vtk
```

## Applications

PANDA is suitable for:

* Research in polytopal / polygonal / polyhedral methods
* Benchmarking PDE solvers on general meshes
* Rapid prototyping of new discretization schemes
* Educational use in numerical PDEs

## Linear solvers

The PDE solvers use SciPy's sparse direct solver by default. Krylov solvers can be selected without changing system assembly:

```python
solver = P1DGPoissonSolver(
    mesh,
    bc_manager,
    linear_solver="cg",
    solver_options={
        "preconditioner": "jacobi",
        "rtol": 1e-8,
        "maxiter": 1000,
    },
)
u_dofs = solver.solve(f)
print(solver.last_solve_info)
```

Supported methods are `direct`, `cg`, `minres`, `gmres`, and `bicgstab`; preconditioners are `jacobi` and `ilu`. CG is intended for symmetric positive definite Poisson and elasticity systems. Stokes is indefinite, so GMRES (or MINRES with a symmetric positive-definite preconditioner) is the safer choice. For difficult Stokes saddle-point matrices, pass SciPy `spilu` ordering and pivot options through `preconditioner_options`:

```python
solver_options={
    "preconditioner": "ilu",
    "preconditioner_options": {
        "drop_tol": 1e-3,
        "fill_factor": 10,
        "permc_spec": "MMD_AT_PLUS_A",
        "diag_pivot_thresh": 0.0,
    },
    "restart": 100,
    "rtol": 1e-8,
    "maxiter": 1000,
}
```

> By default, an ILU zero pivot is retried with small, scale-aware diagonal shifts. These shifts affect only the preconditioner, not the PDE system.

---

## License

@@@@
