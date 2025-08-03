#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 1 16:59:52 2022
@author: Riccardo Tenderini
@mail: riccardo.tenderini@epfl.ch
"""

import numpy as np
import os

import src.rb_library.rb_manager.space_time.Stokes.rb_manager_SRB_TFO_Stokes as rbmsrbtfoS
import src.rb_library.rb_manager.space_time.Navier_Stokes.rb_manager_space_time_Navier_Stokes as rbmstNS
from src.utils.newton import Newton

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../../../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class RbManagerSRB_TFO_NavierStokes(rbmstNS.RbManagerSpaceTimeNavierStokes,
                                    rbmsrbtfoS.RbManagerSRB_TFO_Stokes):
    """
    MODIFY
    """

    def __init__(self, _fom_problem, _affine_decomposition=None):
        """
        MODIFY
        """

        super().__init__(_fom_problem, _affine_decomposition=_affine_decomposition)

        return

    def build_ST_basis(self, _tolerances, which=None):
        """
        MODIFY
        """

        rbmsrbtfoS.RbManagerSRB_TFO_Stokes.build_ST_basis(self, _tolerances, which=which)

        return

    def compute_NLterm_offline_quantities_time(self):
        """
        MODIFY
        """
        return

    def assemble_reduced_structures_nlterm(self, x_hat, param=None):
        """
        Assemble the reduced structures needed to assemble the reduced non-linear term and its jacobian
        """
        return

    def reset_reduced_structures_nlterm(self):
        """
        Reset the reduced structures needed to assemble the reduced non-linear term and its jacobian
        """
        return

    def reset_reduced_structures(self):
        """
        MODIFY
        """
        rbmstNS.RbManagerSpaceTimeNavierStokes.reset_reduced_structures(self)
        return

    def build_rb_affine_decompositions(self, operators=None):
        """
        MODIFY
        """
        raise NotImplementedError

    def build_rb_parametric_RHS(self, param):
        """
        MODIFY
        """
        raise NotImplementedError

    def check_dataset(self, _nsnap):
        """
        MODIFY
        """
        rbmstNS.RbManagerSpaceTimeNavierStokes.check_dataset(self, _nsnap)
        return

    def _solve(self, lhs_block, rhs_block, sol):
        """
        Solve pipeline for navier-Stokes with SRB-TFO
        """

        def residual(x):
            nl_term = self.build_reduced_convective_term(x)
            res = lhs_block.dot(x) + nl_term - rhs_block
            return res

        def jacobian(x):
            if self.M_newton_specifics['use convective jacobian']:
                nl_jac = self.build_reduced_convective_jacobian(x)
                return lhs_block + nl_jac
            return lhs_block

        my_newton = Newton(tol=self.M_newton_specifics['tolerance'],
                           tol_abs=self.M_newton_specifics['absolute tolerance'],
                           max_err=self.M_newton_specifics['max error'],
                           max_err_abs=self.M_newton_specifics['absolute max error'],
                           max_iter=self.M_newton_specifics['max iterations'], jac_iter=1, alpha=1.0)

        initial_guess = self._extrapolate_solution(sol)

        return my_newton(residual, jacobian, initial_guess, verbose=False)

    def build_reduced_convective_term(self, x):
        """
        Assemble the reduced convective term
        """

        N_comp = len(self.M_NLTerm_affine_components)
        if N_comp == 0:
            # logger.warning("No affine components for the non-linear convective term have been loaded!")
            return 0

        u = x[:N_comp]  # (n_c,)
        affine_components = np.array(self.M_NLTerm_affine_components)  # (n_c, n_c, n_s)

        nl_term = np.sum(u[None, :, None] * u[:, None, None] * affine_components, axis=(0,1))
        nl_term *= self.M_bdf_rhs * self.dt

        res = np.hstack([nl_term,
                         np.zeros(self.M_N_space['pressure']),
                         np.zeros(np.sum(self.M_N_space['lambda']).astype(int))])

        return res

    def build_reduced_convective_jacobian(self, x):
        """
        Assemble the Jacobian matrix of the reduced convective term
        """

        N_comp = len(self.M_NLJacobian_affine_components)
        if N_comp == 0:
            logger.warning("No affine components for the non-linear convective jacobian have been loaded!")
            return 0

        u = x[:N_comp]  # (n_c,)
        affine_components = np.array(self.M_NLJacobian_affine_components)  # (n_c, n_s, n_s)

        nl_jac = np.sum(u[:, None, None] * affine_components, axis=0)
        nl_jac *= self.M_bdf_rhs * self.dt

        dim = self.M_N_space['velocity'] + self.M_N_space['pressure'] + np.sum(self.M_N_space['lambda']).astype(int)
        res = np.zeros((dim, dim))
        res[:self.M_N_space['velocity'], :self.M_N_space['velocity']] = nl_jac

        return res

