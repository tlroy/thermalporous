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
* Block preconditioner for single phase flow from [Roy et al., 2019](https://doi.org/10.1016/j.jcp.2019.06.038)
* CPTR preconditioner for multiphase flow from [Roy et al., 2020](https://doi.org/10.1137/19M1292023)
* DG0 formulation of the Finite Volume method

### Piecewise constant discontinuous Galerkin (DG0)
The DG0 discretization is equivalent in the weak sense to the Finite Volume method with two-point flux approximation (TPFA). For some simple examples of using DG0 in Firedrake, please look in the intro folder. More details can be in found [Roy et al., 2019](https://doi.org/10.1016/j.jcp.2019.06.038) and [Roy, 2019](https://www.ora.ox.ac.uk/objects/uuid:478d84ed-fd67-4fd5-ba69-b7c962792c67).
