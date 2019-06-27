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
## Piecewise constant discontinuous Galerkin (DG0)
For some simple examples of using DG0 in Firedrake, please look in the intro folder. More details can be found in this [paper](https://arxiv.org/abs/1902.00095).
