#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 09:48:05 2021
@author: Riccardo Tenderini
@mail: riccardo.tenderini@epfl.ch
"""

import numpy as np
import os
import time

import src.rb_library.rb_manager.space_time.Stokes.rb_manager_space_time_Stokes as rbmstS

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../../../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class RbManagerSTRBStokes(rbmstS.RbManagerSpaceTimeStokes):
    """MODIFY
    """

    def __init__(self, _fom_problem, _affine_decomposition=None):
        """ MODIFY
        """

        super().__init__(_fom_problem, _affine_decomposition=_affine_decomposition)

        self.M_basis_time_IC_elements = dict()

        self.M_reduction_method = "ST-RB"

        return

    def build_IC_basis_elements(self):
        """
        Build quantities needed to enforce initial conditions with the ST-RB method.
        """

        assert {'velocity', 'pressure', 'lambda'}.issubset(self.M_basis_time), \
            "Time basis for velocity, pressure and lambda must be available"

        self.M_basis_time_IC_elements = dict()

        self.M_basis_time_IC_elements['velocity'] = sum(self.M_basis_time['velocity'], 0)[None]
        self.M_basis_time_IC_elements['velocity_S1'] = sum(self.M_basis_time['velocity'][1:], 0)[None]
        self.M_basis_time_IC_elements['velocity_S2'] = sum(self.M_basis_time['velocity'][2:], 0)[None]
        self.M_basis_time_IC_elements['velocity_2'] = self.M_basis_time['velocity'].T.dot(self.M_basis_time['velocity'])
        self.M_basis_time_IC_elements['pressure'] = sum(self.M_basis_time['pressure'], 0)[None, None]
        self.M_basis_time_IC_elements['lambda'] = [np.zeros(0)] * self.M_n_coupling
        for n in range(self.M_n_coupling):
            self.M_basis_time_IC_elements['lambda'][n] = sum(self.M_basis_time['lambda'][n], 0)[None, None]

        return

    def build_ST_basis(self, _tolerances, which=None):
        """
        MODIFY
        """

        if which is None:
            which = {'velocity-space', 'velocity-time',
                     'pressure-space', 'pressure-time',
                     'lambda-space', 'lambda-time'}

        # N_Fourier = {'velocity': 10, 'pressure': 5, 'lambda': 4}  # TODO: this should become an input
        method_POD_time = 'full'  # TODO: this may become a user-defined input

        if 'pressure-space' in which:
            start = time.time()
            self.perform_pod_space(_tol=_tolerances['pressure-space'], field="pressure")
            logger.debug(f"Pressure space basis computed in {(time.time() - start):.4f} s")
        if 'pressure-time' in which:
            start = time.time()
            self.perform_pod_time(_tol=_tolerances['pressure-time'], method=method_POD_time, field="pressure")
            # self.time_basis_fourier(N_Fourier["pressure"], field="pressure")
            logger.debug(f"Pressure time basis computed in {(time.time() - start):.4f} s")
        if {'pressure-space', 'pressure-time'} & which:
            self.M_N['pressure'] = self.M_N_space['pressure'] * self.M_N_time['pressure']
            logger.info('Finished pressure snapshots PODs \n')

        if 'lambda-space' in which:
            if _tolerances['lambda-space'] is not None:
                start = time.time()
                self.perform_pod_space(_tol=_tolerances['lambda-space'], field="lambda")
                logger.debug(f"Lagrange multipliers space basis computed in {(time.time() - start):.4f} s")
            else:
                self.M_N_space['lambda'] = self.M_Nh['lambda']
                self.M_basis_space['lambda'] = [np.eye(self.M_Nh['lambda'][n])
                                                for n in range(self.M_n_coupling)]
                self.M_sv_space['lambda'] = [np.ones(self.M_Nh['lambda'][n]) / self.M_Nh['lambda'][n]
                                             for n in range(self.M_n_coupling)]
        if 'lambda-time' in which:
            start = time.time()
            self.perform_pod_time(_tol=_tolerances['lambda-time'], method=method_POD_time, field="lambda")
            # self.time_basis_fourier(N_Fourier["lambda"], field="lambda")
            logger.debug(f"Lagrange multipliers time basis computed in {(time.time() - start):.4f} s")
        if {'lambda-space', 'lambda-time'} & which:
            self.M_N['lambda'] = self.M_N_space['lambda'] * self.M_N_time['lambda']
            self.M_N_lambda_cumulative = np.cumsum(np.vstack([self.M_N['lambda'][n] for n in range(self.M_n_coupling)]))
            self.M_N_lambda_cumulative = np.insert(self.M_N_lambda_cumulative, 0, 0)
            logger.info('Finished Lagrange multipliers snapshots PODs \n')

        if 'velocity-space' in which:
            start = time.time()
            self.perform_pod_space(_tol=_tolerances['velocity-space'], field="velocity")
            self.primal_supremizers(stabilize=False)  # omitting stabilization, since it does not make a big difference
            self.dual_supremizers(stabilize=False)  # always omit stabilization with Lagrange multipliers
            self.M_basis_space['velocity'] = np.hstack((self.M_basis_space['velocity'], self.supr_primal, self.supr_dual))
            self.M_N_space['velocity'] += self.supr_primal.shape[1] + self.supr_dual.shape[1]
            logger.debug(f"Velocity space basis computed in {(time.time() - start):.4f} s")
        if 'velocity-time' in which:
            start = time.time()
            self.perform_pod_time(_tol=_tolerances['velocity-time'], method=method_POD_time, field="velocity")
            # self.time_basis_fourier(N_Fourier["velocity"], field="velocity")
            self.primal_supremizers_time(tol=9e-1)
            self.dual_supremizers_time(tol=9e-1)
            logger.debug(f"Velocity time basis computed in {(time.time() - start):.4f} s")
        if {'velocity-space', 'velocity-time'} & which:
            self.M_N['velocity'] = self.M_N_space['velocity'] * self.M_N_time['velocity']
            logger.info('Finished velocity snapshots PODs \n')

        self.build_IC_basis_elements()

        if which:
            self.save_ST_basis(which=which)

        return

    def build_rb_nonparametric_LHS(self):
        """
        MODIFY
        """

        self.M_Blocks = [np.zeros(0)] * 9
        self.set_empty_blocks()

        M_matrix_t = np.expand_dims(self.M_M_matrix, (2, 3))
        A_matrix_t = np.expand_dims(self.M_A_matrix, (2, 3))
        BdivT_matrix_t = np.expand_dims(self.M_BdivT_matrix, (2, 3))
        Bdiv_matrix_t = np.expand_dims(self.M_Bdiv_matrix, (2, 3))
        BT_matrix_t = []
        for n in range(self.M_n_coupling):
            BT_matrix_t.append(np.expand_dims(self.M_BT_matrix[n], (2, 3)))
        B_matrix_t = []
        for n in range(self.M_n_coupling):
            B_matrix_t.append(np.expand_dims(self.M_B_matrix[n], (2, 3)))
        if self.M_has_resistance:
            R_matrix_t = np.expand_dims(self.M_R_matrix, (2, 3))

        psi_u_psi_u = np.expand_dims(self.M_basis_time['velocity'].T.dot(self.M_basis_time['velocity']),
                                     (0, 1))
        psi_u_psi_p = np.expand_dims(self.M_basis_time['velocity'].T.dot(self.M_basis_time['pressure']),
                                     (0, 1))
        psi_u_psi_u_shift1 = np.expand_dims(self.M_basis_time['velocity'][1:].T.dot(self.M_basis_time['velocity'][:-1]),
                                            (0, 1))
        psi_u_psi_u_shift2 = np.expand_dims(self.M_basis_time['velocity'][2:].T.dot(self.M_basis_time['velocity'][:-2]),
                                            (0, 1))
        psi_u_psi_lambda = [np.zeros(0)] * self.M_n_coupling
        for n in range(self.M_n_coupling):
            psi_u_psi_lambda[n] = np.expand_dims(self.M_basis_time['velocity'].T.dot(self.M_basis_time['lambda'][n]),
                                                 (0, 1))

        logger.debug('Constructing the ST-RB momentum equation')
        block0 = ((M_matrix_t + self.M_bdf_rhs * self.dt * A_matrix_t) * psi_u_psi_u
                  + self.M_bdf[0] * M_matrix_t * psi_u_psi_u_shift1
                  + self.M_bdf[1] * M_matrix_t * psi_u_psi_u_shift2)
        if self.M_has_resistance:
            block0 += self.M_bdf_rhs * self.dt * R_matrix_t * psi_u_psi_u

        self.M_Blocks[0] = np.reshape(np.transpose(block0, (0, 2, 1, 3)),
                                      (self.M_N['velocity'], self.M_N['velocity']))

        if 'clot' in self.M_parametrizations:
            self.M_Blocks_param_affine['clot'] = dict()
            self.M_Blocks_param_affine['clot'][0] = []
            for Mclot_matrix in self.M_Mclot_matrices:
                Mclot_matrix_t = np.expand_dims(Mclot_matrix, (2, 3))
                block = self.M_bdf_rhs * self.dt * Mclot_matrix_t * psi_u_psi_u
                self.M_Blocks_param_affine['clot'][0].append(np.reshape(np.transpose(block, (0, 2, 1, 3)),
                                                                    (self.M_N['velocity'], self.M_N['velocity'])))

        block1 = self.M_bdf_rhs * self.dt * BdivT_matrix_t * psi_u_psi_p
        self.M_Blocks[1] = np.reshape(np.transpose(block1, (0, 2, 1, 3)),
                                      (self.M_N['velocity'], self.M_N['pressure']))

        blocks2 = []
        for n in range(self.M_n_coupling):
            tmp = self.M_bdf_rhs * self.dt * BT_matrix_t[n] * psi_u_psi_lambda[n]
            blocks2.append(np.reshape(np.transpose(tmp, (0, 2, 1, 3)),
                                      (self.M_N['velocity'], self.M_N['lambda'][n])))
        self.M_Blocks[2] = np.block([b2 for b2 in blocks2])

        logger.debug('Constructing the ST-RB continuity equation')
        block3 = Bdiv_matrix_t * np.swapaxes(psi_u_psi_p, 2, 3)
        self.M_Blocks[3] = np.reshape(np.transpose(block3, (0, 2, 1, 3)),
                                      (self.M_N['pressure'], self.M_N['velocity']))

        logger.debug('Constructing the ST-RB coupling equation')
        blocks6 = []
        for n in range(self.M_n_coupling):
            tmp = B_matrix_t[n] * np.swapaxes(psi_u_psi_lambda[n], 2, 3)
            blocks6.append(np.reshape(np.transpose(tmp, (0, 2, 1, 3)),
                                      (self.M_N['lambda'][n], self.M_N['velocity'])))
        self.M_Blocks[6] = np.block([[b6] for b6 in blocks6])

        return

    def _reconstruct_IC(self, field, n=0):
        """
        Reconstruct the initial condition for a given field from the available solution
        """

        basis_time = self.M_basis_time[field] if field not in {'lambda'} else self.M_basis_time[field][n]
        f0 = np.dot(self.get_solution_field(field), basis_time[-2:].T).T

        f0_ic = 0
        if self._has_IC(field=field, n=n):
            f0_ic = (self.M_u0[field][-1] if field not in {'lambda'} else self.M_u0[field][n][-1])[None]

        return f0 + f0_ic

    def update_IC_terms(self, update_IC=False):
        """
        MODIFY
        """

        if update_IC:
            self._update_IC()

        self.M_f_Blocks_no_param[0] *= 0
        self.M_f_Blocks_no_param[1] *= 0
        self.M_f_Blocks_no_param[2] *= 0

        if not self._has_IC():
            return

        # IC contributions for time marching
        Mu0 = self.M_M_matrix.dot(self.M_u0['velocity'].T)
        tmp_M = np.stack([self.M_bdf[0] * Mu0[:, 1] + self.M_bdf[1] * Mu0[:, 0],
                          self.M_bdf[1] * Mu0[:, 1]], axis=1)
        self.M_f_Blocks_no_param[0] = -(tmp_M.dot(self.M_basis_time['velocity'][:2])).flatten()

        # IC contribution stemming from solution reconstruction
        expand_space = lambda X: np.expand_dims(X, 1)

        B0_A = expand_space(self.M_bdf_rhs * self.dt * self.M_A_matrix.dot(self.M_u0['velocity'][-1]))  # (n_u^s, 1)
        self.M_f_Blocks_no_param[0] -= (B0_A * self.M_basis_time_IC_elements['velocity']).flatten()
        B0_M = expand_space(Mu0[:, -1])
        self.M_f_Blocks_no_param[0] -= (B0_M * (self.M_basis_time_IC_elements['velocity'] +
                                                self.M_bdf[0] * self.M_basis_time_IC_elements['velocity_S1'] +
                                                self.M_bdf[1] * self.M_basis_time_IC_elements['velocity_S2'])).flatten()
        if self.M_has_resistance:
            B0_R = expand_space((self.M_bdf_rhs * self.dt * self.M_R_matrix).dot(self.M_u0['velocity'][-1]))
            self.M_f_Blocks_no_param[0] -= (B0_R * self.M_basis_time_IC_elements['velocity']).flatten()

        # TODO: clots contributions

        B1 = expand_space(self.M_bdf_rhs * self.dt * self.M_BdivT_matrix.dot(self.M_u0['pressure'][-1]))
        self.M_f_Blocks_no_param[0] -= (B1 * self.M_basis_time_IC_elements['velocity']).flatten()

        for n in range(self.M_n_coupling):
            B2 = expand_space(self.M_bdf_rhs * self.dt * self.M_BT_matrix[n].dot(self.M_u0['lambda'][n][-1]))
            self.M_f_Blocks_no_param[0] -= (B2 * self.M_basis_time_IC_elements['velocity']).flatten()

        # this can also be omitted, because u0 should be (weakly) div-free
        B3 = expand_space(self.M_Bdiv_matrix.dot(self.M_u0['velocity'][-1]))
        self.M_f_Blocks_no_param[1] -= (B3 * self.M_basis_time_IC_elements['pressure']).flatten()

        F6 = []
        for n in range(self.M_n_coupling):
            B6 = expand_space(self.M_B_matrix[n].dot(self.M_u0['velocity'][-1]))
            F6.append((B6 * self.M_basis_time_IC_elements['lambda'][n]).flatten())
        self.M_f_Blocks_no_param[2] -= np.hstack(F6)

        self.M_f_Block = np.hstack([self._get_f_block(0), self._get_f_block(1), self._get_f_block(2)])

        return

    def build_rb_parametric_RHS(self, param):
        """
        MODIFY
        """

        super().build_rb_parametric_RHS(param)

        flow_rates = self.get_flow_rates(param)

        f3 = [np.zeros(0)] * self.M_n_coupling
        for n in range(self.M_n_coupling):
            time_basis_flow_rates = self.M_basis_time['lambda'][n].T.dot(flow_rates[:, n])[None]
            f3[n] = (self.M_RHS_vector[n][:, None] * time_basis_flow_rates).flatten()

        f3 = np.hstack([f3[n] for n in range(self.M_n_coupling)])

        self.M_f_Blocks_param[2] = f3

        return

    def save_rb_affine_decomposition(self, blocks=None, operators=None):
        """
        Saving RB blocks for the LHS matrix
        """

        if blocks is None:
            blocks = [0, 1, 2, 3, 6]

        super().save_rb_affine_decomposition(blocks=blocks, operators=operators)

        return

    def import_rb_affine_components(self, blocks=None, operators=None):
        """
        Importing RB blocks for the LHS matrix
        """

        if blocks is None:
            blocks = [0, 1, 2, 3, 6]

        import_failures_set = super().import_rb_affine_components(blocks=blocks, operators=operators)

        return import_failures_set

    def time_marching(self, *args, **kwargs):
        raise NotImplementedError("Time marching method available only with SRB-TFO method")

    def test_time_marching(self, *args, **kwargs):
        raise NotImplementedError("Time marching method available only with SRB-TFO method")


__all__ = [
    "RbManagerSTRBStokes"
]
