#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

The *SpaceTime* subsubmodule contains the implementation of test files that allows to assess the performances of the
implemented ST-LSPG on the unsteady parametrized PDE problem at hand. In particular, two test files are present:

    * *main_spacetime*: standard main file, that allows to test the ST-LSPG solver for a fixed set of hyper-parameters,
      to be chosen at configuration stage
    * *test_spacetime*: file that checks how the performances of the ST-LSPG solver are influenced by the choice of some
      hyper-parameters, as the POD tolerance, the number of snapshots used to compute the Reduced Basis (in both space
      and time) or the fraction of SpaceTime locations to be sampled for the final hyper-reduction prior to residual
      minimization
"""

