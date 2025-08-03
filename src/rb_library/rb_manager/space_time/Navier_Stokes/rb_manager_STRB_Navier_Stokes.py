#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 1 16:59:52 2022
@author: Riccardo Tenderini
@mail: riccardo.tenderini@epfl.ch
"""

import numpy as np
import os

import src.rb_library.rb_manager.space_time.Stokes.rb_manager_STRB_Stokes as rbmstrbS
import src.rb_library.rb_manager.space_time.Navier_Stokes.rb_manager_space_time_Navier_Stokes as rbmstNS

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../../../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class RbManagerSTRBNavierStokes(rbmstNS.RbManagerSpaceTimeNavierStokes,
                                rbmstrbS.RbManagerSTRBStokes):
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

        rbmstrbS.RbManagerSTRBStokes.build_ST_basis(self, _tolerances, which=which)

        return

    def build_reduced_convective_term(self, x_hat):
        """
        MODIFY
        """

        N_comp = len(self.M_NLTerm_affine_components)
        if not N_comp:
            logger.warning("No affine components for the non-linear convective term have been loaded!")
            return 0

        # FAST ASSEMBLING --> does not depend on N_t
        u_hat = np.reshape(x_hat[:self.M_N['velocity']], (self.M_N_space['velocity'], self.M_N_time['velocity']))[:N_comp]  # (n_c, n_t)
        u_hat2 = np.swapaxes(np.multiply.outer(u_hat, u_hat), 1, 2)  # (n_c, n_c, n_t, n_t)

        affine_components = np.array(self.M_NLTerm_affine_components)  # (n_c, n_c, n_s)

        res = np.sum(affine_components[..., None] *
                     np.sum(u_hat2[..., None] *
                            self.M_NLterm_offline_time_tensor_uuu[None, None], axis=(2,3))[:, :, None],
                     axis=(0,1)).flatten()
        res *= (self.M_bdf_rhs * self.dt)

        res += self.build_reduced_convective_term_IC(x_hat)

        K = np.block([res, np.zeros(self.M_N['pressure']), np.zeros(self.M_N_lambda_cumulative[-1])])

        return K

    def build_reduced_convective_term_IC(self, x_hat):
        """
        Build convective term contributions related to the initial conditions
        """

        N_comp = len(self.M_NLJacobian_affine_components)

        if not N_comp or not self._has_IC():
            return 0

        u_hat = np.reshape(x_hat[:self.M_N['velocity']], (self.M_N_space['velocity'],
                                                          self.M_N_time['velocity']))[:N_comp]  # (n_c, n_t)
        u0 = self.M_u0['velocity'][-1, :N_comp]  # (n_c, )

        affine_components = np.array(self.M_NLTerm_affine_components)   # (n_c, n_c, n_s)

        res1 = (np.sum(u0[:, None, None] * u0[None, :, None] * affine_components, axis=(0,1))[:, None] *
                self.M_basis_time_IC_elements['velocity'][None]).flatten()

        res2 = np.sum((np.sum(u0[None, :, None] * affine_components, axis=1) +
                       np.sum(u0[:, None, None] * affine_components, axis=0))[..., None] *
                      np.sum(u_hat[..., None] * self.M_basis_time_IC_elements['velocity_2'][None], axis=1)[:, None],
                      axis=0).flatten()

        res = (res1 + res2) * self.M_bdf_rhs * self.dt

        return res

    def build_reduced_convective_jacobian(self, x_hat):
        """
        MODIFY
        """

        N_comp = len(self.M_NLJacobian_affine_components)
        if not N_comp:
            logger.warning("No affine components for the non-linear jacobian matrix have been loaded!")
            return 0

        # FAST ASSEMBLING --> does not depend on Nt
        u_hat = np.reshape(x_hat[:self.M_N['velocity']], (self.M_N_space['velocity'], self.M_N_time['velocity']))[:N_comp]  # (n_c, n_t)

        affine_components = np.array(self.M_NLJacobian_affine_components)  # (n_c, n_s, n_s)

        res = np.sum(affine_components[..., None, None] *
                     np.sum(u_hat[..., None, None] * self.M_NLterm_offline_time_tensor_uuu[None], axis=1)[:, None, None],
                     axis=0)
        res = res.swapaxes(1, 2).reshape(self.M_N['velocity'], self.M_N['velocity'])  # (n_st, n_st)
        res *= (self.M_bdf_rhs * self.dt)

        res += self.build_reduced_convective_jacobian_IC()

        N = self.M_N['velocity'] + self.M_N['pressure'] + self.M_N_lambda_cumulative[-1]
        J = np.zeros((N, N))
        J[:self.M_N['velocity'], :self.M_N['velocity']] = res

        return J

    def build_reduced_convective_jacobian_IC(self):
        """
        Build convective Jacobian contributions related to the initial conditions
        """

        N_comp = len(self.M_NLJacobian_affine_components)

        if not N_comp or not self._has_IC():
            return 0

        u0 = self.M_u0['velocity'][-1, :N_comp]  # (n_c, )

        affine_components = np.array(self.M_NLJacobian_affine_components)  # (n_c, n_s, n_s)

        res = (np.sum(affine_components * u0[:, None, None], axis=0)[..., None, None] *
               self.M_basis_time_IC_elements['velocity_2'][None, None])
        res = res.swapaxes(1, 2).reshape(self.M_N['velocity'], self.M_N['velocity'])  # (n_st, n_st)
        res *= (self.M_bdf_rhs * self.dt)

        return res

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

        rbmstrbS.RbManagerSTRBStokes.build_rb_affine_decompositions(self, operators=operators)

        return

    def build_rb_parametric_RHS(self, param):
        """
        MODIFY
        """

        return rbmstrbS.RbManagerSTRBStokes.build_rb_parametric_RHS(self, param)

    def check_dataset(self, _nsnap):
        """
        MODIFY
        """

        rbmstNS.RbManagerSpaceTimeNavierStokes.check_dataset(self, _nsnap)

        return
