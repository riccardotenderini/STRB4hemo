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
        u_hat2 = np.swapaxes(np.multiply.outer(u_hat, u_hat), 1, 2)[..., None]  # (n_c, n_c, n_t, n_t, 1)

        affine_components = np.array(self.M_NLTerm_affine_components)[..., None]  # (n_c, n_c, n_s, 1)

        res = np.zeros(self.M_N['velocity'])
        for i in range(N_comp):
            for j in range(N_comp):
                tmp = np.sum(u_hat2[i, j] * self.M_NLterm_offline_time_tensor_uuu, axis=(0, 1))[None]  # (1, n_t)
                res += (affine_components[i, j] * tmp).flatten()
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

        u_hat = np.reshape(x_hat[:self.M_N['velocity']], (self.M_N_space['velocity'], self.M_N_time['velocity']))[:N_comp]  # (n_c, n_t)
        u0 = self.M_u0['velocity'].T[:N_comp]
        uhat_0 = np.swapaxes(np.multiply.outer(u_hat, u0), 1, 2)[..., None]  # (n_c, n_c, n_t, S, 1
        u0_2 = np.swapaxes(np.multiply.outer(u0, u0), 1, 2)[..., None]  # (n_c, n_c, S, S, 1)

        affine_components = np.array(self.M_NLTerm_affine_components) [..., None]  # (n_c, n_c, n_s, 1)

        res1 = np.zeros(self.M_N['velocity'])
        time_factor = self.M_NLterm_offline_time_tensor_uuu_IC_2  # (S, S, n_t)
        for i in range(N_comp):
            for j in range(N_comp):
                tmp = np.sum(u0_2[i, j] * time_factor, axis=(0, 1))[None]  # (1, n_t)
                res1 += (affine_components[i, j] * tmp).flatten()
        res1 *= (self.M_bdf_rhs * self.dt)

        res2 = np.zeros(self.M_N['velocity'])
        time_factor = np.swapaxes(self.M_NLterm_offline_time_tensor_uuu_IC_1, 0, 1)  # (n_t, S, n_t)
        for i in range(N_comp):
            for j in range(N_comp):
                tmp = np.sum(uhat_0[i, j] * time_factor, axis=(0, 1))[None]  # (1, n_t)
                res2 += (affine_components[i, j] * tmp).flatten()
        res2 *= (self.M_bdf_rhs * self.dt)

        res3 = np.zeros(self.M_N['velocity'])
        time_factor = self.M_NLterm_offline_time_tensor_uuu_IC_1  # (S, n_t, n_t)
        for i in range(N_comp):
            for j in range(N_comp):
                tmp = np.sum(np.swapaxes(uhat_0[i, j], 0, 1) * time_factor, axis=(0, 1))[None]  # (1, n_t)
                res3 += (affine_components[i, j] * tmp).flatten()
        res3 *= (self.M_bdf_rhs * self.dt)

        return res1 + res2 + res3

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
        u_hat = np.expand_dims(u_hat, (2, 3))  # (n_c, n_t, 1, 1)

        affine_components = np.expand_dims(np.array(self.M_NLJacobian_affine_components), (3, 4))  # (n_c, n_s, n_s, 1, 1)

        res = np.zeros((self.M_N['velocity'], self.M_N['velocity']))
        for j in range(N_comp):
            tmp = np.sum(u_hat[j] * self.M_NLterm_offline_time_tensor_uuu, axis=0)[None, None]  # (1, 1, n_t, n_t)
            res += (affine_components[j] * tmp).swapaxes(1,2).reshape(self.M_N['velocity'], self.M_N['velocity'])  # (n_st, n_st)
        res *= (self.M_bdf_rhs * self.dt)

        res += self.build_reduced_convective_jacobian_IC()

        N = self.M_N['velocity'] + self.M_N['pressure'] + self.M_N_lambda_cumulative[-1]
        J = np.zeros((N, N))
        J[:self.M_N['velocity'], :self.M_N['velocity']] = res

        return J

    def build_reduced_convective_jacobian_IC(self):
        """
        Build convective jacobian contributions related to the initial conditions
        """

        N_comp = len(self.M_NLJacobian_affine_components)

        if not N_comp or not self._has_IC():
            return 0

        u0 = np.expand_dims(self.M_u0['velocity'].T[:N_comp], (2, 3))  # (n_c, S, 1, 1)

        affine_components = np.expand_dims(np.array(self.M_NLJacobian_affine_components),(3, 4))  # (n_c, n_s, n_s, 1, 1)

        time_factor = self.M_NLterm_offline_time_tensor_uuu_IC_1  # (S, n_t, n_t)

        res = np.zeros((self.M_N['velocity'], self.M_N['velocity']))
        for j in range(N_comp):
            tmp = np.sum(u0[j] * time_factor, axis=0)[None, None]  # (1, 1, n_t, n_t)
            res += (affine_components[j] * tmp).swapaxes(1, 2).reshape(self.M_N['velocity'],
                                                                       self.M_N['velocity'])  # (n_st, n_st)
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
