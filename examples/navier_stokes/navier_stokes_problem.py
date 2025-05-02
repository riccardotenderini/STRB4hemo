#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29, 16:56:22, 2022
@author: Riccardo Tenderini
@email : riccardo.tenderini@epfl.ch
"""

import examples.stokes.stokes_problem as sp


class NavierStokesProblem(sp.StokesProblem):
    """MODIFY
    """

    def __init__(self, _parameter_handler):
        """MODIFY
        """
        super().__init__(_parameter_handler)
        return
