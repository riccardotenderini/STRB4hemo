#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 5 10:16:48 2023
@author: Riccardo Tenderini
@mail: riccardo.tenderini@epfl.ch
"""

import numpy as np
import os

import src.rb_library.rb_manager.space_time.Navier_Stokes.rb_manager_STRB_Navier_Stokes as rbmstrbNS
import src.rb_library.rb_manager.space_time.Membrane.rb_manager_space_time_Membrane as rbmstM


import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../../../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class RbManagerSTRBMembrane(rbmstrbNS.RbManagerSTRBNavierStokes,
                            rbmstM.RbManagerSpaceTimeMembrane):
    """
    MODIFY
    """

    def __init__(self, _fom_problem, _affine_decomposition=None):
        """
        MODIFY
        """

        super().__init__(_fom_problem, _affine_decomposition=_affine_decomposition)

        return

    def build_IC_basis_elements(self):
        """
        Build quantities needed to enforce initial conditions with the ST-RB method.
        """

        rbmstrbNS.RbManagerSTRBNavierStokes.build_IC_basis_elements(self)

        basis_IC_int = self._bdf2_integration(np.ones(self.M_Nt), ic_mode='constant', zero_ic=True)
        self.M_basis_time_IC_elements['velocity_int_full'] = basis_IC_int
        self.M_basis_time_IC_elements['velocity_int'] = self.M_basis_time['velocity'].T.dot(basis_IC_int)[None]

        return

    def build_ST_basis(self, _tolerances, which=None):
        """
        MODIFY
        """

        if which is None:
            which = {'velocity-space', 'velocity-time',
                     'displacement-space', 'displacement-time',
                     'pressure-space', 'pressure-time',
                     'lambda-space', 'lambda-time'}

        which_d = set()
        if 'displacement-space' in which:
            which_d.add('displacement-space')
            which.remove('displacement-space')
        if 'displacement-time' in which:
            which_d.add('displacement-time')
            which.remove('displacement-time')

        rbmstrbNS.RbManagerSTRBNavierStokes.build_ST_basis(self, _tolerances, which=which)

        if 'displacement-space' in which_d:
            rbmstM.RbManagerSpaceTimeMembrane.perform_pod_space(self, field="displacement")
        if 'displacement-time' in which_d:
            rbmstM.RbManagerSpaceTimeMembrane.perform_pod_time(self, method='reduced', field="displacement")
        if {'displacement-space', 'displacement-time'} & which_d:
            self.M_N['displacement'] = self.M_N_space['displacement'] * self.M_N_time['displacement']

        if self.M_save_offline_structures:
            self.save_ST_basis(which=which_d)

        logger.info('Finished building displacement reduced bases \n')

        return

    def build_rb_nonparametric_LHS(self):
        """
        MODIFY
        """

        rbmstrbNS.RbManagerSTRBNavierStokes.build_rb_nonparametric_LHS(self)

        K = len(self.M_Abd_matrices)

        expand_space = lambda X: np.expand_dims(X, (2, 3))
        expand_time = lambda X: np.expand_dims(X, (0, 1))
        reshape_space_time = lambda X, N1, N2: np.reshape(np.transpose(X, (0, 2, 1, 3)), (N1, N2))

        self.M_Blocks_param_affine['structure'] = dict()
        self.M_Blocks_param_affine['structure'][0] = [np.zeros(0)] * (K + 1)

        psi_u_psi_u = expand_time(self.M_basis_time['velocity'].T.dot(self.M_basis_time['velocity']))
        psi_u_psi_u_shift1 = expand_time(self.M_basis_time['velocity'][1:].T.dot(self.M_basis_time['velocity'][:-1]))
        psi_u_psi_u_shift2 = expand_time(self.M_basis_time['velocity'][2:].T.dot(self.M_basis_time['velocity'][:-2]))
        psi_u_psi_d = expand_time(self.M_basis_time['velocity'].T.dot(self.M_basis_time['displacement']))

        logger.debug('Constructing boundary ST-RB blocks')
        Mbd_matrix_t = expand_space(self.M_Mbd_matrix)
        block = Mbd_matrix_t * (psi_u_psi_u + self.M_bdf[0] * psi_u_psi_u_shift1 + self.M_bdf[1] * psi_u_psi_u_shift2)
        self.M_Blocks_param_affine['structure'][0][0] = reshape_space_time(block,
                                                                           self.M_N['velocity'], self.M_N['velocity'])

        for k in range(K):
            Abd_matrix_t = expand_space(self.M_Abd_matrices[k])
            block = Abd_matrix_t * psi_u_psi_d * (self.M_bdf_rhs * self.dt)  # implicit velocity-displacement coupling
            self.M_Blocks_param_affine['structure'][0][k+1] = reshape_space_time(block,
                                                                                 self.M_N['velocity'], self.M_N['velocity'])

        if self._has_wall_elasticity():
            MbdWall_matrix_t = expand_space(self.M_MbdWall_matrix)
            block = MbdWall_matrix_t * psi_u_psi_d * (self.M_bdf_rhs * self.dt * self.M_wall_elasticity)
            self.M_Blocks[0] += reshape_space_time(block, self.M_N['velocity'], self.M_N['velocity'])

        return

    def _reconstruct_IC(self, field, n=0):
        """
        Reconstruct the initial condition for a given field from the available solution
        """

        if field == 'displacement':
            d0 = np.dot(self.get_solution_field("velocity"), self.M_basis_time['displacement'][-2:].T).T

            d0_ic, u0_ic = 0, 0
            if self._has_IC(field='displacement'):
                d0_ic = self.M_u0['displacement'][-1][None]
            if self._has_IC(field='velocity'):
                u0_ic = self.M_u0['velocity'][-1, None] * self.M_basis_time_IC_elements['velocity_int_full'][-2:, None]

            return d0 + d0_ic + u0_ic

        else:
            return rbmstrbNS.RbManagerSTRBNavierStokes._reconstruct_IC(self, field, n=n)

    def update_IC_terms(self, update_IC=False):
        """
        MODIFY
        """

        rbmstrbNS.RbManagerSTRBNavierStokes.update_IC_terms(self, update_IC=update_IC)

        K = len(self.M_Abd_matrices)
        self.M_f_Blocks_param_affine['structure'] = dict()
        self.M_f_Blocks_param_affine['structure'][0] = [np.zeros(0)] * (K + 1)
        for k in range(K+1):
            self.M_f_Blocks_param_affine['structure'][0][k] = np.zeros(self.M_N['velocity'])

        self.set_param_functions()

        if not (self._has_IC('velocity') or self._has_IC('displacement')):
            return

        # IC contributions for time marching
        Mu0 = self.M_Mbd_matrix.dot(self.M_u0['velocity'].T)
        tmp_M = np.stack([self.M_bdf[0] * Mu0[:, 1] + self.M_bdf[1] * Mu0[:, 0],
                         self.M_bdf[1] * Mu0[:, 1]], axis=1)
        self.M_f_Blocks_param_affine['structure'][0][0] = - (tmp_M.dot(self.M_basis_time['velocity'][:2])).flatten()

        # IC contribution stemming from solution reconstruction
        expand_space = lambda X: np.expand_dims(X, 1)

        u0 = self.M_u0['velocity'][-1]
        d0 = self.M_u0['displacement'][-1]

        B0_M = expand_space(Mu0[:, -1])
        self.M_f_Blocks_param_affine['structure'][0][0] -= (B0_M * (self.M_basis_time_IC_elements['velocity'] +
                                                                    self.M_bdf[0] * self.M_basis_time_IC_elements['velocity_S1'] +
                                                                    self.M_bdf[1] * self.M_basis_time_IC_elements['velocity_S2'])).flatten()

        for k in range(K):
            BO_Abd = expand_space(self.M_bdf_rhs * self.dt * self.M_Abd_matrices[k].dot(d0))
            self.M_f_Blocks_param_affine['structure'][0][k+1] = - (BO_Abd * self.M_basis_time_IC_elements['velocity']).flatten()
            BO_Abd_u = expand_space(self.M_bdf_rhs * self.dt * self.M_Abd_matrices[k].dot(u0))
            self.M_f_Blocks_param_affine['structure'][0][k+1] -= (BO_Abd_u * self.M_basis_time_IC_elements['velocity_int']).flatten()

        if self._has_wall_elasticity():
            B0_E = expand_space(self.M_bdf_rhs * self.dt * self.M_MbdWall_matrix.dot(d0))
            self.M_f_Blocks_no_param[0] -= (B0_E * self.M_basis_time_IC_elements['velocity']).flatten()
            B0_E_u = expand_space(self.M_bdf_rhs * self.dt * self.M_MbdWall_matrix.dot(u0))
            self.M_f_Blocks_no_param[0] -= (B0_E_u * self.M_basis_time_IC_elements['velocity_int']).flatten()

        return

    def _save_results_snapshot(self, param_nb, errors, is_test=False):
        """
        MODIFY
        """
        rbmstM.RbManagerSpaceTimeMembrane._save_results_snapshot(self, param_nb, errors, is_test=is_test)
        return

    def _save_results_general(self):
        rbmstM.RbManagerSpaceTimeMembrane._save_results_general(self)
        return

    def compute_online_errors(self, param_nb, sol=None, is_test=False, ss_ratio=1):
        return rbmstM.RbManagerSpaceTimeMembrane.compute_online_errors(self, param_nb, sol=sol,
                                                                       is_test=is_test, ss_ratio=ss_ratio)

    def _reset_errors(self):
        rbmstM.RbManagerSpaceTimeMembrane._reset_errors(self)
        return

    def _update_errors(self, N=None, is_IG=False):
        rbmstM.RbManagerSpaceTimeMembrane._update_errors(self, N=N, is_IG=is_IG)
        return
