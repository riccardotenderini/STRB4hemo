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

        psi_u_psi_u = expand_time(np.eye(self.M_N_time['velocity']))
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

        # TODO: add variant with semi-implicit coupling (?)

        return

    def _update_IC(self):
        """
        Update initial conditions
        """

        rbmstrbNS.RbManagerSTRBNavierStokes._update_IC(self)

        d0 = np.dot(self.get_solution_field("velocity"), self.M_basis_time['displacement'][-2:].T).T

        d0_ic, u0_ic = 0, 0
        if self._has_IC(field='displacement'):
            d0_ic = np.sum((self.M_basis_time_IC[:, np.newaxis, -2:] *
                            self.M_u0['displacement'][..., np.newaxis]), axis=0).T
        if self._has_IC(field='velocity'):
            u0_ic = np.sum((self.M_basis_time_IC_int[:, np.newaxis, -2:] *
                            self.M_u0['velocity'][..., np.newaxis]), axis=0).T

        self.M_u0['displacement'] = d0 + d0_ic + u0_ic

        return

    def update_IC_terms(self, update_IC=False):
        """
        MODIFY
        """

        rbmstrbNS.RbManagerSTRBNavierStokes.update_IC_terms(self, update_IC=update_IC)

        if not self._has_IC():
            return

        K = len(self.M_Abd_matrices)

        self.M_f_Blocks_param_affine['structure'] = dict()
        self.M_f_Blocks_param_affine['structure'][0] = [np.zeros(0)] * (K+1)

        # IC contributions for time marching
        Mu0 = self.M_Mbd_matrix.dot(self.M_u0['velocity'].T)
        tmp_M = - np.stack([self.M_bdf[1] * Mu0[:, 0] + self.M_bdf[0] * Mu0[:, 1],
                           self.M_bdf[1] * Mu0[:, 1]], axis=1)
        self.M_f_Blocks_param_affine['structure'][0][0] = (tmp_M.dot(self.M_basis_time['velocity'][:2])).flatten()

        # IC contribution stemming from solution reconstruction
        expand_space = lambda X: np.expand_dims(X, 1)
        expand_time = lambda X: np.expand_dims(X, 0)

        psi_u_IC = expand_time(self.M_basis_time['velocity'].T.dot(self.M_basis_time_IC.T))  # (1, n_u^t, 2)
        psi_u_IC_shift1 = expand_time(self.M_basis_time['velocity'][1:].T.dot(self.M_basis_time_IC.T[:-1]))
        psi_u_IC_shift2 = expand_time(self.M_basis_time['velocity'][2:].T.dot(self.M_basis_time_IC.T[:-2]))
        psi_u_IC_int = expand_time(self.M_basis_time['velocity'].T.dot(self.M_basis_time_IC_int.T))

        B0_M = expand_space(self.M_Mbd_matrix.dot(self.M_u0['velocity'].T))
        self.M_f_Blocks_param_affine['structure'][0][0] -= np.sum(B0_M * (psi_u_IC +
                                                                          self.M_bdf[0] * psi_u_IC_shift1 +
                                                                          self.M_bdf[1] * psi_u_IC_shift2),
                                                                  axis=-1).flatten()

        for k in range(K):
            BO_Abd = expand_space(self.M_bdf_rhs * self.dt * self.M_Abd_matrices[k].dot(self.M_u0['displacement'].T))
            self.M_f_Blocks_param_affine['structure'][0][k+1] = - np.sum(BO_Abd * psi_u_IC, axis=-1).flatten()
            BO_Abd_u = expand_space(self.M_bdf_rhs * self.dt * self.M_Abd_matrices[k].dot(self.M_u0['velocity'].T))
            self.M_f_Blocks_param_affine['structure'][0][k+1] -= np.sum(BO_Abd_u * psi_u_IC_int, axis=-1).flatten()

        if self._has_wall_elasticity():
            B0_E = expand_space(self.M_bdf_rhs * self.dt * self.M_MbdWall_matrix.dot(self.M_u0['displacement'].T))
            self.M_f_Blocks_no_param[0] -= np.sum(B0_E * psi_u_IC, axis=-1).flatten()
            B0_E_u = expand_space(self.M_bdf_rhs * self.dt * self.M_MbdWall_matrix.dot(self.M_u0['velocity'].T))
            self.M_f_Blocks_no_param[0] -= np.sum(B0_E_u * psi_u_IC_int, axis=-1).flatten()

        self.M_f_Block = np.hstack([self._get_f_block(0), self._get_f_block(1), self._get_f_block(2)])

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
