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

Download a [**MedCoupling binary distribution**](https://www.salome-platform.org/?page_id=2430) (standalone or via SALOME) compatible with your system.

---

## Running the Code

Before running PANDA, you must initialize the SALOME/MedCoupling environment.

### 1. Load the MedCoupling environment

```bash
./salome context
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

---

## License

@@@@

