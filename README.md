# Thermalporous
DG0 solver for non-isothermal flow in porous media

## Requirements
Please install the open-source [Firedrake](https://www.firedrakeproject.org) finite element library first.

## Installation
Run:

    git clone https://github.com/tlroy/thermalporous.git

To use Thermalporous, activate the Firedrake virtualenv and then run

    cd thermalporous
    source activate.sh
    
### Model features
* Single and two- phase non-isothermal flow of viscous oil and water
* Anisotropic and heterogeneous permeability fields
* Well and heater source terms

### Solver features
* CPR preconditioner from [Wallis, 1983](https://arxiv.org/abs/1907.04229) with decoupling operators
* Block preconditioner for single phase flow from [Roy et al., 2019a](https://doi.org/10.1016/j.jcp.2019.06.038)
* CPTR preconditioner for multiphase flow from [Roy et al., 2019b](https://arxiv.org/abs/1907.04229)
* DG0 formulation of the Finite Volume method

### Piecewise constant discontinuous Galerkin (DG0)
For some simple examples of using DG0 in Firedrake, please look in the intro folder. More details can be in found [Roy et al., 2019a](https://doi.org/10.1016/j.jcp.2019.06.038).
