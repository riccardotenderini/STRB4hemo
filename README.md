# STRB4hemo
This repository provides a Python-based implementation of space-time reduced basis (ST-RB) solvers 
for hemodynamic problems. Specifically, it allows to use ST-RB to solve the following problems:
* Stokes problem
* Navier-Stokes problem
* Coupled momentum problem (reduced fluid-structure interaction), defined in 
  [Figueroa et al. (2006)](https://www.sciencedirect.com/science/article/abs/pii/S004578250500513X)

This is the reference repository for the following papers: 
  * [Space-time reduced basis methods for parametrized unsteady Stokes equations](https://epubs.siam.org/doi/full/10.1137/22M1509114?casa_token=8gTtT37emsoAAAAA:7W1EQSPnDxM5LcLok5icCwnwfUR5F9XsWLtfr_ZK1aQozxR5mm1teGDQjM-h3cD522inRPWE5Tw&casa_token=TRTtydhDGR8AAAAA:ie1HJGZIwfSN-jA7nLBSkc11fpEH1soQe0qdujUC1mTthvrr9BnhZd8x4RFd2pTV9Lhc2UyBqR8) by R.Tenderini, N.Mueller, S.Deparis (2024).
  * [Model order reduction of hemodynamics by space-time reduced basis and reduced fluid-structure 
interaction](https://arxiv.org/abs/2505.00548) by R.Tenderini and S.Deparis (2025).

## Installation
The code is written in Python and requires **Python 3.10 or higher**.
All the dependencies are listed in the `requirements.txt` file. They can be installed using *pip*:
```bash
pip install -r requirements.txt
```
We recommend using a virtual environment to avoid conflicts.

## Data 
The data (high-fidelity snapshots and matrices, eventually pre-assembled reduced bases 
and reduced quantities) should be located in a folder whose path must be specified in the
configuration file of each example. 

The results of the tests are stored by default in the same folder; this
can be changed in the configuration file of the examples of choice.

A light-weight dataset, featuring simple test cases on coarse meshes, will be soon made available.

The data used to generate the results presented in the paper can be made available upon request. 
To this aim, please contact [Riccardo Tenderini](mailto:riccardo.tenderini@epfl.ch).

## Usage
The `main.py` files to be executed are located in the `examples` folder, inside the subfolder 
`problem_name/SpaceTime`. The specifics of each problem can be configured through the `config.py` 
file, located in the same folder.
Upon proper configuration, the code can be executed by running:
```bash
python main.py
```

The source code is located in the `src` folder. In particular, the ST-RB solvers for the
hemodynamic problems tackled in the reference paper can be found at
`src/rb_library/rb_manager/space_time`.

## License
This project is licensed under the [BSD 3-Clause License](LICENSE).

## Credits
The code builds upon previous implementations of the RB method for steady elliptic problems,
written by [Luca Pegolotti](https://www.researchgate.net/profile/Luca_Pegolotti) and 
[Niccolo' Dal Santo](https://www.researchgate.net/profile/Niccolo_Dal_Santo) 
at [EPFL](https://www.epfl.ch/en/).
