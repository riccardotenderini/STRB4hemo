#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 22:06:27 2019
@author: Riccardo Tenderini
@email : riccardo.tenderini@epfl.ch
"""
import numpy as np
import os

import src.pde_problem.fom_problem_unsteady as fpu

import logging.config
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def tb_theta_a(_param, _q):
    """MODIFY
    """
    try:
        assert _q == 0
    except AssertionError:
        raise ValueError(f"-Stokes problem has 1 parameter-dependent function associated to the stiffness "
                         f"matrix operator. Index {_q} is not a valid index")
   
    return 1.0


def tb_full_theta_a(_param):
    """MODIFY
    """
    diffusions = np.ones(len(_param))
    return diffusions


def tb_theta_f(_param, _q):
    """MODIFY
    """

    try:
        assert _q == 0
    except AssertionError:
        raise Exception(f"-Stokes problem has 1 parameter-dependent functions associated to the right-hand "
                        f"side vector operator. Index {_q} is not a valid index")
    return 1.0


def tb_full_theta_f(_param):
    """MODIFY
    """

    return np.array([0.0])


class StokesProblem(fpu.FomProblemUnsteady):
    """MODIFY
    """

    def __init__(self, _parameter_handler):
        """MODIFY
        """
        super().__init__(_parameter_handler)
        
        return

    def define_theta_functions(self):
        """MODIFY
        """
        self.M_theta_a = tb_theta_a
        self.M_theta_f = tb_theta_f

        self.M_full_theta_a = tb_full_theta_a
        self.M_full_theta_f = tb_full_theta_f

        return


__all__ = [
    "tb_theta_a",
    "tb_theta_f",
    "tb_full_theta_a",
    "tb_full_theta_f",
    "StokesProblem"
]
