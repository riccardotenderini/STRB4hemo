#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28, 09:46:20 2022
@author: Riccardo Tenderini
@email : riccardo.tenderini@epfl.ch
"""

import examples.navier_stokes.navier_stokes_problem as nsp


class NavierStokesMembraneProblem(nsp.NavierStokesProblem):
    """MODIFY
    """

    def __init__(self, _parameter_handler):
        """MODIFY
        """
        super().__init__(_parameter_handler)

        return
