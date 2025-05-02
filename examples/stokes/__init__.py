#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The *thermal_block_unsteady* submodule contains the implementation of the test files that allow to assess the
performances of the implemented dimensionality reduce methods for unsteady parametrized PDE problems. In particular,
it contains a file featuring the implementation of a class defining the unsteady thermal block problem and two other
submodules, namely:

    * *Multistep*: submodule that uses RB-projected multistep solvers to handle the time-dependent problem at hand
    * *SpaceTime*: submodule that uses the ST-LSPG projection approach to handle the time-dependent problem at hand
"""

