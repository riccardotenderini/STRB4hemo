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
# from sksparse.cholmod import cholesky
import scipy
from scipy.sparse import csc_matrix, load_npz
import scipy.sparse.linalg

import src.rb_library.rb_manager.space_time.Stokes.rb_manager_space_time_Stokes as rbmstS
import src.utils.array_utils as arr_utils
import src.utils.general_utils as gen_utils

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../../../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class RbManagerSTPGRBStokes(rbmstS.RbManagerSpaceTimeStokes):
    """MODIFY
    """

    def __init__(self, _fom_problem, _affine_decomposition=None):
        """
        MODIFY
        """

        super().__init__(_fom_problem, _affine_decomposition=_affine_decomposition)

        self.M_reduction_method = "ST-PGRB"

        self.M_MM_matrix = None
        self.M_MA_matrix = None
        self.M_AA_matrix = None
        self.M_BdivBdiv_matrix = None
        self.M_BdivTBdivT_matrix = None
        self.M_MBdivT_matrix = None
        self.M_ABdivT_matrix = None
        self.M_BB_matrix = None
        self.M_BTBT_matrix = None
        self.M_MBT_matrix = None
        self.M_ABT_matrix = None
        self.M_BdivTBT_matrix = None

        self.M_MMclot_matrices = None
        self.M_AMclot_matrices = None
        self.M_MclotMclot_matrices = None
        self.M_MclotBdivT_matrices = None
        self.M_MclotBT_matrices = None

        self.M_used_norm = ""
        self.M_inv_norm = dict()

        logger.warning("Resistance BCs are not supported in the current implementation of ST-PGRB!")

        return

    def build_ST_basis(self, _tolerances, which=None):
        """MODIFY
        """

        if which is None:
            which = {'velocity-space', 'velocity-time',
                     'pressure-space', 'pressure-time',
                     'lambda-space', 'lambda-time'}

        if 'pressure-space' in which:
            start = time.time()
            self.perform_pod_space(_tol=_tolerances['pressure-space'], field="pressure")
            logger.debug(f"Pressure space basis computed in {time.time() - start} s")
        if 'pressure-time' in which:
            start = time.time()
            self.perform_pod_time(_tol=_tolerances['pressure-time'], method='reduced', field="pressure")
            logger.debug(f"Pressure time basis computed in {time.time() - start} s")
        if {'pressure-space', 'pressure-time'} & which:
            self.M_N['pressure'] = self.M_N_space['pressure'] * self.M_N_time['pressure']
            logger.info('Finished pressure snapshots PODs \n')

        if 'lambda-space' in which:
            if _tolerances['lambda-space'] is not None:
                self.perform_pod_space(_tol=_tolerances['lambda-space'], field="lambda")
            else:
                self.M_N_space['lambda'] = self.M_Nh['lambda']
                self.M_basis_space['lambda'] = [np.eye(self.M_Nh['lambda'][n])
                                                for n in range(self.M_n_coupling)]
                self.M_sv_space['lambda'] = [np.ones(self.M_Nh['lambda'][n]) / self.M_Nh['lambda'][n]
                                             for n in range(self.M_n_coupling)]
        if 'lambda-time' in which:
            start = time.time()
            self.perform_pod_time(_tol=_tolerances['lambda-time'], method='reduced', field="lambda")
            logger.debug(f"Lagrange multipliers time basis computed in {time.time() - start} s")
        if {'lambda-space', 'lambda-time'} & which:
            self.M_N['lambda'] = self.M_N_space['lambda'] * self.M_N_time['lambda']
            self.M_N_lambda_cumulative = np.cumsum(np.vstack([self.M_N['lambda'][n] for n in range(self.M_n_coupling)]))
            self.M_N_lambda_cumulative = np.insert(self.M_N_lambda_cumulative, 0, 0)
            logger.info('Finished Lagrange multipliers snapshots PODs \n')

        if 'velocity-space' in which:
            start = time.time()
            self.perform_pod_space(_tol=_tolerances['velocity-space'], field="velocity")
            logger.debug(f"Velocity space basis computed in {time.time() - start} s")
        if 'velocity-time' in which:
            start = time.time()
            self.perform_pod_time(_tol=_tolerances['velocity-time'], method='reduced', field="velocity")
            self.primal_supremizers_time(tol=9e-1)
            # self.dual_supremizers_time(tol=9e-1)
            logger.debug(f"Velocity time basis computed in {time.time() - start} s")
        if {'velocity-space', 'velocity-time'} & which:
            self.M_N['velocity'] = self.M_N_space['velocity'] * self.M_N_time['velocity']
            logger.info('Finished velocity snapshots PODs \n')

        if which:
            self.save_ST_basis()

        return

    def set_projection_norm(self):

        assert self.M_used_norm in {'l2', 'X', 'P'}, "Invalid norm matrix for the Petrov-Galerkin projection!"

        if self.M_used_norm == 'X':
            raise ValueError("Inverse of full norm matrix cannot be computed!")
            # if os.path.isfile(os.path.join(self.M_fom_structures_path, os.path.normpath('norm0_inv.npz'))) and \
            #         os.path.isfile(os.path.join(self.M_fom_structures_path, os.path.normpath('norm1_inv.npz'))):
            #     self.M_inv_norm['velocity'] = load_npz(os.path.join(self.M_fom_structures_path, 'norm0_inv.npz'))
            #     self.M_inv_norm['pressure'] = load_npz(os.path.join(self.M_fom_structures_path, 'norm1_inv.npz'))
            # else:
            #     factor_norm0 = cholesky(self.M_norm_matrices['velocity'])
            #     self.M_inv_norm['velocity'] = factor_norm0.inv()
            #     factor_norm1 = cholesky(self.M_norm_matrices['pressure'])
            #     self.M_inv_norm['pressure'] = factor_norm1.inv()
            #     scipy.sparse.save_npz(os.path.join(self.M_fom_structures_path, 'norm0_inv.npz'),
            #                           csc_matrix(self.M_inv_norm['velocity']))
            #     scipy.sparse.save_npz(os.path.join(self.M_fom_structures_path, 'norm1_inv.npz'),
            #                           csc_matrix(self.M_inv_norm['pressure']))

        elif self.M_used_norm == 'P':
            if os.path.isfile(os.path.join(self.M_fom_structures_path, os.path.normpath('prec_norm0_inv.npz'))) and \
                    os.path.isfile(os.path.join(self.M_fom_structures_path, os.path.normpath('prec_norm1_inv.npz'))):
                self.M_inv_norm['velocity'] = load_npz(os.path.join(self.M_fom_structures_path, 'prec_norm0_inv.npz'))
                self.M_inv_norm['pressure'] = load_npz(os.path.join(self.M_fom_structures_path, 'prec_norm1_inv.npz'))
            else:
                self.M_inv_norm['velocity'] = scipy.sparse.diags(1.0 / self.M_norm_matrices['velocity'].diagonal(),
                                                                 format='csc')
                self.M_inv_norm['pressure'] = scipy.sparse.diags(1.0 / self.M_norm_matrices['pressure'].diagonal(),
                                                                 format='csc')
                scipy.sparse.save_npz(os.path.join(self.M_fom_structures_path, 'prec_norm0_inv.npz'),
                                      csc_matrix(self.M_inv_norm['velocity']))
                scipy.sparse.save_npz(os.path.join(self.M_fom_structures_path, 'prec_norm1_inv.npz'),
                                      csc_matrix(self.M_inv_norm['pressure']))

        elif self.M_used_norm == 'l2':
            self.M_inv_norm['velocity'] = csc_matrix(np.eye(self.M_Nh['velocity']))
            self.M_inv_norm['pressure'] = csc_matrix(np.eye(self.M_Nh['pressure']))

        else:
            raise ValueError(f"Invalid choice {self.M_used_norm} of norm matrix to use "
                             f"in the Petrov-Galerkin projection!")

        return

    def inverse_norm_premultiplication(self, matrix, norm_field):
        """
        MODIFY
        """

        if not self.check_norm_matrices():
            self.get_norm_matrices()
        self.set_projection_norm()

        assert len(matrix.shape) <= 2
        if len(matrix.shape) == 2 and matrix.shape[1] == 1:
            matrix = matrix[:, 0]

        is_vector = (len(matrix.shape) == 1)

        if self.M_used_norm == 'X':
            new_matrix = arr_utils.solve_sparse_system(self.M_norm_matrices[norm_field], matrix)
        elif self.M_used_norm == 'P':
            if is_vector:
                new_matrix = arr_utils.sparse_matrix_vector_mul(self.M_inv_norm[norm_field], matrix)
            else:
                new_matrix = arr_utils.sparse_matrix_matrix_mul(self.M_inv_norm[norm_field], matrix)
        elif self.M_used_norm == 'l2':
            new_matrix = matrix
        else:
            raise ValueError(f"Invalid choice {self.M_used_norm} of norm matrix to use "
                             f"in the Petrov-Galerkin projection!")

        return new_matrix

    def _assemble_right_reduced_structures(self):
        """
        MODIFY
        """

        structures = {'A', 'M', 'Bdiv', 'B', 'RHS', 'R'}
        if 'clot' in self.M_parametrizations:
            structures.add('Mclot')
        FEM_matrices = self.import_FEM_structures(structures=structures)

        if 'R' in FEM_matrices:
            raise ValueError("Resistance BCs are not supported in the current implementation of ST-PGRB!")

        logger.info("Projecting FEM structures onto the reduced subspace in space; this may take some time...")

        matrices = dict()

        matrices['M'] = arr_utils.sparse_matrix_matrix_mul(FEM_matrices['M'], self.M_basis_space['velocity'])
        matrices['A'] = arr_utils.sparse_matrix_matrix_mul(FEM_matrices['A'], self.M_basis_space['velocity'])
        matrices['Bdiv'] = arr_utils.sparse_matrix_matrix_mul(FEM_matrices['Bdiv'], self.M_basis_space['velocity'])
        matrices['BdivT'] = arr_utils.sparse_matrix_matrix_mul(FEM_matrices['BdivT'], self.M_basis_space['pressure'])
        matrices['BT'] = [arr_utils.sparse_matrix_matrix_mul(FEM_matrices['BT'][n], self.M_basis_space['lambda'][n])
                          for n in range(self.M_n_coupling)]
        matrices['B'] = [arr_utils.sparse_matrix_matrix_mul(FEM_matrices['B'][n], self.M_basis_space['velocity'])
                         for n in range(self.M_n_coupling)]
        matrices['RHS'] = FEM_matrices['RHS']
        if 'clot' in self.M_parametrizations:
            matrices['Mclot'] = [arr_utils.sparse_matrix_matrix_mul(Mclot, self.M_basis_space['velocity'])
                                  for Mclot in FEM_matrices['Mclot']]

        matrices['XM'] = self.inverse_norm_premultiplication(matrices['M'], 'velocity')
        matrices['XA'] = self.inverse_norm_premultiplication(matrices['A'], 'velocity')
        matrices['XBdiv'] = self.inverse_norm_premultiplication(matrices['Bdiv'], 'pressure')
        matrices['XBdivT'] = self.inverse_norm_premultiplication(matrices['BdivT'], 'velocity')
        matrices['XB'] = matrices['B']
        matrices['XBT'] = [self.inverse_norm_premultiplication(matrices['BT'][n], 'velocity')
                           for n in range(self.M_n_coupling)]
        if 'clot' in self.M_parametrizations:
            matrices['XMclot'] = [self.inverse_norm_premultiplication(matrices['Mclot'][k], 'velocity')
                                   for k in range(len(matrices['Mclot']))]

        return matrices
        
    def assemble_reduced_structures(self, _space_projection='standard', matrices=None):
        """
        MODIFY
        """

        if not self.check_norm_matrices():
            self.get_norm_matrices()
        self.set_projection_norm()

        if matrices is None:
            matrices = self._assemble_right_reduced_structures()
        FEM_matrices = self.import_FEM_structures(structures={'q'})

        logger.debug("Computing (MPhi)^T*(MPhi)")
        self.M_MM_matrix = matrices['M'].T.dot(matrices['XM'])
        logger.debug("Computing (MPhi)^T*(APhi)")
        self.M_MA_matrix = matrices['M'].T.dot(matrices['XA'])
        logger.debug("Computing (APhi)^T*(APhi)")
        self.M_AA_matrix = matrices['A'].T.dot(matrices['XA'])

        logger.debug("Computing (BdivPhi)^T*(BdivPhi)")
        self.M_BdivBdiv_matrix = matrices['Bdiv'].T.dot(matrices['XBdiv'])
        logger.debug("Computing (BdivTPhi_p)^T*(BdivTPhi_p)")
        self.M_BdivTBdivT_matrix = matrices['BdivT'].T.dot(matrices['XBdivT'])

        logger.debug("Computing (MPhi)^T*(BdivTPhi_p)")
        self.M_MBdivT_matrix = matrices['M'].T.dot(matrices['XBdivT'])
        logger.debug("Computing (APhi)^T*(BdivTPhi_p)")
        self.M_ABdivT_matrix = matrices['A'].T.dot(matrices['XBdivT'])

        self.M_MBT_matrix = [np.zeros(0)] * self.M_n_coupling
        self.M_ABT_matrix = [np.zeros(0)] * self.M_n_coupling
        self.M_BdivTBT_matrix = [np.zeros(0)] * self.M_n_coupling
        self.M_BB_matrix = [np.zeros(0)] * self.M_n_coupling
        self.M_BTBT_matrix = [np.zeros(0)] * self.M_n_coupling
        self.M_RHS_vector = [np.zeros(0)] * self.M_n_coupling

        for n in range(self.M_n_coupling):
            logger.debug(f"Computing (MPhi)^T*(BTPhi_l), multiplier {n}")
            self.M_MBT_matrix[n] = matrices['M'].T.dot(matrices['XBT'][n])
            logger.debug(f"Computing (APhi)^T*(BTPhi_l), multiplier {n}")
            self.M_ABT_matrix[n] = matrices['A'].T.dot(matrices['XBT'][n])
            logger.debug(f"Computing (BdivTPhi_p)^T*(BTPhi_l), multiplier {n}")
            self.M_BdivTBT_matrix[n] = matrices['BdivT'].T.dot(matrices['XBT'][n])
            logger.debug(f"Computing (BPhi)^T*(BPhi), multiplier {n}")
            self.M_BB_matrix[n] = matrices['B'][n].T.dot(matrices['XB'][n])
            logger.debug(f"Computing (BTPhi_l)^T*(BTPhi_l), multiplier {n}")
            self.M_BTBT_matrix[n] = matrices['BT'][n].T.dot(matrices['XBT'][n])
            logger.debug(f"Computing (Bphi)^T*RHS, multiplier {n}")
            self.M_RHS_vector[n] = matrices['B'][n].T.dot(matrices['RHS'][n])

        logger.debug("Computing Phi^T*u0")
        try:
            initial_condition = self.import_FEM_structures(structures={'u0'})
            self.M_u0['velocity'] = self.project_vector(initial_condition['u0'].T, self.M_basis_space['velocity'],
                                                        norm_matrix=self.M_norm_matrices['velocity']).T
        except ValueError:
            logger.warning("Impossible to load the initial condition. Proceeding with homogeneous initial condition")
            self.M_u0['velocity'] = np.zeros((2, self.M_N_space['velocity']))

        if 'clot' in self.M_parametrizations:
            n_clots = len(matrices['Mclot'])

            self.M_MMclot_matrices = [np.zeros(0)] * n_clots
            self.M_AMclot_matrices = [np.zeros(0)] * n_clots
            self.M_MclotMclot_matrices = []
            self.M_MclotBdivT_matrices = [np.zeros(0)] * n_clots
            self.M_MclotBT_matrices = []

            for k in range(n_clots):
                self.M_MMclot_matrices[k] = matrices['M'].T.dot(matrices['XMclot'][k])
                self.M_AMclot_matrices[k] = matrices['A'].T.dot(matrices['XMclot'][k])
                self.M_MclotBdivT_matrices[k] = matrices['Mclot'][k].T.dot(matrices['XBdivT'])

                self.M_MclotMclot_matrices.append([])
                for kk in range(n_clots):
                    self.M_MclotMclot_matrices[-1].append(matrices['Mclot'][k].T.dot(matrices['XMclot'][kk]))

                self.M_MclotBT_matrices.append([])
                for n in range(self.M_n_coupling):
                    self.M_MclotBT_matrices[-1].append(matrices['Mclot'][k].T.dot(matrices['XBT'][n]))

        logger.debug("Computing Phi^T*q")
        for (idx_q, q) in enumerate(FEM_matrices['q_in']):
            self.M_q_vectors['in'].append(self.project_vector(q, self.M_basis_space['velocity'], norm_matrix=None))
        for (idx_q, q) in enumerate(FEM_matrices['q_out']):
            self.M_q_vectors['out'].append(self.project_vector(q, self.M_basis_space['velocity'], norm_matrix=None))

        logger.info("Projection of FEM structures with Petrov-Galerkin is complete!")

        if self.M_save_offline_structures:
            self.save_reduced_structures()

        return

    def save_reduced_structures(self):
        """
        MODIFY
        """

        logger.debug("Dumping space-reduced structures to file ...")

        gen_utils.create_dir(self.M_reduced_structures_path)

        np.save(os.path.join(self.M_reduced_structures_path, 'MM_rb.npy'), self.M_MM_matrix)
        np.save(os.path.join(self.M_reduced_structures_path, 'MA_rb.npy'), self.M_MA_matrix)
        np.save(os.path.join(self.M_reduced_structures_path, 'AA_rb.npy'), self.M_AA_matrix)
        np.save(os.path.join(self.M_reduced_structures_path, 'BdivBdiv_rb.npy'), self.M_BdivBdiv_matrix)
        np.save(os.path.join(self.M_reduced_structures_path, 'BdivTBdivT_rb.npy'), self.M_BdivTBdivT_matrix)
        np.save(os.path.join(self.M_reduced_structures_path, 'MBdivT_rb.npy'), self.M_MBdivT_matrix)
        np.save(os.path.join(self.M_reduced_structures_path, 'ABdivT_rb.npy'), self.M_ABdivT_matrix)

        for n in range(self.M_n_coupling):
            np.save(os.path.join(self.M_reduced_structures_path, 'MBT' + str(n) + '_rb.npy'), self.M_MBT_matrix[n])
            np.save(os.path.join(self.M_reduced_structures_path, 'ABT' + str(n) + '_rb.npy'), self.M_ABT_matrix[n])
            np.save(os.path.join(self.M_reduced_structures_path, 'BdivTBT' + str(n) + '_rb.npy'), self.M_BdivTBT_matrix[n])
            np.save(os.path.join(self.M_reduced_structures_path, 'BB' + str(n) + '_rb.npy'), self.M_BB_matrix[n])
            np.save(os.path.join(self.M_reduced_structures_path, 'RHS' + str(n) + '_rb.npy'), self.M_RHS_vector[n])
            np.save(os.path.join(self.M_reduced_structures_path, 'BTBT' + '_' + str(n) + '_rb.npy'), self.M_BTBT_matrix[n])

        if 'clot' in self.M_parametrizations:
            n_clots = len(self.M_MMclot_matrices)
            for k in range(n_clots):
                np.save(os.path.join(self.M_reduced_structures_path, f"MMclot{k}_rb.npy"),
                        self.M_MMclot_matrices[k])
                np.save(os.path.join(self.M_reduced_structures_path, f"AMclot{k}_rb.npy"),
                        self.M_AMclot_matrices[k])
                np.save(os.path.join(self.M_reduced_structures_path, f"Mclot{k}BdivT_rb.npy"),
                        self.M_MclotBdivT_matrices[k])

                for kk in range(n_clots):
                    np.save(os.path.join(self.M_reduced_structures_path, f"Mclot{k}Mclot{kk}_rb.npy"),
                            self.M_MclotMclot_matrices[k][kk])

                for n in range(self.M_n_coupling):
                    np.save(os.path.join(self.M_reduced_structures_path, f"Mclot{k}BT{n}_rb.npy"),
                            self.M_MclotBT_matrices[k][n])

        np.save(os.path.join(self.M_reduced_structures_path, 'u0_rb.npy'), self.M_u0['velocity'])

        for k_in in range(self.M_n_inlets):
            np.save(os.path.join(self.M_reduced_structures_path, f"q_in{k_in}_rb.npy"), self.M_q_vectors['in'][k_in])
        for k_out in range(self.M_n_outlets):
            np.save(os.path.join(self.M_reduced_structures_path, f"q_out{k_out}_rb.npy"), self.M_q_vectors['out'][k_out])

        return

    def import_reduced_structures(self):
        """
        MODIFY
        """

        self.reset_reduced_structures()

        logger.info("Importing FEM structures multiplied by RB in space")

        try:
            self.M_MM_matrix = np.load(os.path.join(self.M_reduced_structures_path, 'MM_rb.npy'))
            self.M_MA_matrix = np.load(os.path.join(self.M_reduced_structures_path, 'MA_rb.npy'))
            self.M_AA_matrix = np.load(os.path.join(self.M_reduced_structures_path, 'AA_rb.npy'))
            self.M_BdivBdiv_matrix = np.load(os.path.join(self.M_reduced_structures_path, 'BdivBdiv_rb.npy'),)
            self.M_BdivTBdivT_matrix = np.load(os.path.join(self.M_reduced_structures_path, 'BdivTBdivT_rb.npy'),)
            self.M_MBdivT_matrix = np.load(os.path.join(self.M_reduced_structures_path, 'MBdivT_rb.npy'),)
            self.M_ABdivT_matrix = np.load(os.path.join(self.M_reduced_structures_path, 'ABdivT_rb.npy'),)

            for n in range(self.M_n_coupling):
                self.M_MBT_matrix[n] = np.load(os.path.join(self.M_reduced_structures_path, 'MBT' + str(n) + '_rb.npy'))
                self.M_ABT_matrix[n] = np.load(os.path.join(self.M_reduced_structures_path, 'ABT' + str(n) + '_rb.npy'))
                self.M_BdivTBT_matrix[n] = np.load(os.path.join(self.M_reduced_structures_path, 'BdivTBT' + str(n) + '_rb.npy'))
                self.M_BB_matrix[n] = np.load(os.path.join(self.M_reduced_structures_path, 'BB' + str(n) + '_rb.npy'))
                self.M_RHS_vector[n] = np.load(os.path.join(self.M_reduced_structures_path, 'RHS' + str(n) + '_rb.npy'))
                self.M_BTBT_matrix[n] = np.load(os.path.join(self.M_reduced_structures_path, 'BTBT' + '_' + str(n) + '_rb.npy'))

            if 'clot' in self.M_parametrizations:
                k = 0
                pathM = lambda cnt: os.path.join(self.M_reduced_structures_path, f"MMclot{cnt}_rb.npy")
                pathA = lambda cnt: os.path.join(self.M_reduced_structures_path, f"AMclot{cnt}_rb.npy")
                pathMclot = lambda cnt1, cnt2: os.path.join(self.M_reduced_structures_path, f"Mclot{cnt1}Mclot{cnt2}_rb.npy")
                pathBdivT = lambda cnt: os.path.join(self.M_reduced_structures_path, f"Mclot{cnt}BdivT_rb.npy")
                pathBT = lambda cnt, n: os.path.join(self.M_reduced_structures_path, f"Mclot{cnt}BT{n}_rb.npy")

                assert os.path.isfile(pathM(k))
                while os.path.isfile(pathM(k)):
                    self.M_MMclot_matrices[k] = np.load(pathM(k))
                    self.M_AMclot_matrices[k] = np.load(pathA(k))
                    self.M_MclotBdivT_matrices[k] = np.load(pathBdivT(k))

                    self.M_MclotMclot_matrices.append([])
                    kk = 0
                    while os.path.isfile(pathMclot(k, kk)):
                        self.M_MclotMclot_matrices[-1].append(np.load(pathMclot(k, kk)))
                        kk += 1

                    self.M_MclotBT_matrices.append([])
                    n = 0
                    while os.path.isfile(pathBT(k, n)):
                        self.M_MclotBT_matrices[-1].append(np.load(pathBT(k, n)))
                        n += 1

                    k += 1

            self.M_u0['velocity'] = np.load(os.path.join(self.M_reduced_structures_path, "u0_rb.npy"))

            path_in = lambda cnt: os.path.join(self.M_reduced_structures_path, f"q_in{cnt}_rb.npy")
            k_in = 0
            assert os.path.isfile(path_in(k_in))
            while os.path.isfile(path_in(k_in)):
                self.M_q_vectors['in'].append(np.load(path_in(k_in)))
                k_in += 1
            self.M_n_inlets = k_in

            path_out = lambda cnt: os.path.join(self.M_reduced_structures_path, f"q_out{cnt}_rb.npy")
            k_out = 0
            assert os.path.isfile(path_out(k_out))
            while os.path.isfile(path_out(k_out)):
                self.M_q_vectors['out'].append(np.load(path_out(k_out)))
                k_out += 1
            self.M_n_outlets = k_out

            import_success = True

        except (OSError, FileNotFoundError, AssertionError) as e:
            logger.error(f"Error {e}: failed to import the reduced structures!")
            import_success = False

        if import_success:
            self.M_N_space['velocity'] = self.M_MM_matrix.shape[0]
            self.M_N_space['pressure'] = self.M_BdivTBdivT_matrix.shape[0]
            self.M_N_space['lambda'] = np.zeros(self.M_n_coupling, dtype=int)
            for n in range(self.M_n_coupling):
                self.M_N_space['lambda'][n] = self.M_BTBT_matrix[n].shape[0]

        return import_success

    def reset_reduced_structures(self):
        """
        MODIFY
        """

        self.M_MM_matrix = np.zeros(0)
        self.M_MA_matrix = np.zeros(0)
        self.M_AA_matrix = np.zeros(0)
        self.M_BdivBdiv_matrix = np.zeros(0)
        self.M_BdivTBdivT_matrix = np.zeros(0)
        self.M_MBdivT_matrix = np.zeros(0)
        self.M_ABdivT_matrix = np.zeros(0)
        self.M_BB_matrix = [np.zeros(0)] * self.M_n_coupling
        self.M_BTBT_matrix = [np.zeros(0)] * self.M_n_coupling
        self.M_MBT_matrix = [np.zeros(0)] * self.M_n_coupling
        self.M_ABT_matrix = [np.zeros(0)] * self.M_n_coupling
        self.M_BdivTBT_matrix = [np.zeros(0)] * self.M_n_coupling
        self.M_RHS_vector = [np.zeros(0)] * self.M_n_coupling

        if 'clot' in self.M_parametrizations:
            n_clots = self.get_clots_number()
            self.M_MMclot_matrices = [np.zeros(0)] * n_clots
            self.M_AMclot_matrices = [np.zeros(0)] * n_clots
            self.M_MclotMclot_matrices = []
            self.M_MclotBdivT_matrices = [np.zeros(0)] * n_clots
            self.M_MclotBT_matrices = []

        return

    def build_rb_nonparametric_LHS(self):
        """
        MODIFY
        """

        dt = self.dt

        self.M_Blocks = [np.zeros(0)] * 9

        # ASSEMBLING SPATIAL QUANTITIES
        MM_matrix = np.expand_dims(self.M_MM_matrix, (2, 3))
        MA_matrix = np.expand_dims(self.M_MA_matrix, (2, 3))
        AA_matrix = np.expand_dims(self.M_AA_matrix, (2, 3))
        BdivBdiv_matrix = np.expand_dims(self.M_BdivBdiv_matrix, (2, 3))
        BdivTBdivT_matrix = np.expand_dims(self.M_BdivTBdivT_matrix, (2, 3))
        MBdivT_matrix = np.expand_dims(self.M_MBdivT_matrix, (2, 3))
        ABdivT_matrix = np.expand_dims(self.M_ABdivT_matrix, (2, 3))

        MBT_matrix = [np.zeros(0)] * self.M_n_coupling
        ABT_matrix = [np.zeros(0)] * self.M_n_coupling
        BdivTBT_matrix = [np.zeros(0)] * self.M_n_coupling
        BTBT_matrix = [np.zeros(0)] * self.M_n_coupling
        for n in range(self.M_n_coupling):
            MBT_matrix[n] = np.expand_dims(self.M_MBT_matrix[n], (2, 3))
            ABT_matrix[n] = np.expand_dims(self.M_ABT_matrix[n], (2, 3))
            BdivTBT_matrix[n] = np.expand_dims(self.M_BdivTBT_matrix[n], (2, 3))
            BTBT_matrix[n] = np.expand_dims(self.M_BTBT_matrix[n], (2, 3))

        # AUXILIARY MATRICES
        MAMA_matrix = (MM_matrix + self.M_bdf_rhs * dt * (MA_matrix + np.swapaxes(MA_matrix, 0, 1)) +
                       (self.M_bdf_rhs * dt) ** 2 * AA_matrix)
        MAM_matrix = MM_matrix + self.M_bdf_rhs * dt * np.swapaxes(MA_matrix, 0, 1)
        MABdivT_matrix = MBdivT_matrix + self.M_bdf_rhs * dt * ABdivT_matrix
        MABT_matrix = [np.zeros(0)] * self.M_n_coupling
        for n in range(self.M_n_coupling):
            MABT_matrix[n] = MBT_matrix[n] + self.M_bdf_rhs * dt * ABT_matrix[n]

        # ASSEMBLING SPATIAL CLOTH MATRICES
        if 'clot' in self.M_parametrizations:
            n_clots = self.get_clots_number()
            MMclot_matrices = [np.zeros(0)] * n_clots
            AMclot_matrices = [np.zeros(0)] * n_clots
            MclotMclot_matrices = []  # [[np.zeros(0)] * n_clots] * n_clots
            MclotBdivT_matrices = [np.zeros(0)] * n_clots
            MclotBT_matrices = []  # [[np.zeros(0)] * self.M_n_coupling] * n_clots
            for k in range(n_clots):
                MMclot_matrices[k] = np.expand_dims(self.M_MMclot_matrices[k], (2, 3))
                AMclot_matrices[k] = np.expand_dims(self.M_AMclot_matrices[k], (2, 3))
                MclotBdivT_matrices[k] = np.expand_dims(self.M_MclotBdivT_matrices[k], (2, 3))
                MclotMclot_matrices.append([])
                for kk in range(n_clots):
                    MclotMclot_matrices[-1].append(np.expand_dims(self.M_MclotMclot_matrices[k][kk], (2, 3)))
                MclotBT_matrices.append([])
                for n in range(self.M_n_coupling):
                    MclotBT_matrices[-1].append(np.expand_dims(self.M_MclotBT_matrices[k][n], (2, 3)))

        # ASSEMBLING TEMPORAL QUANTITIES
        psi_u_psi_u = np.expand_dims(np.eye(self.M_N_time['velocity']),
                                     (0, 1))
        psi_p_psi_p = np.expand_dims(np.eye(self.M_N_time['pressure']),
                                     (0, 1))
        psi_u_psi_u_leave1 = np.expand_dims(self.M_basis_time['velocity'][:-1].T.dot(self.M_basis_time['velocity'][:-1]),
                                            (0, 1))
        psi_u_psi_u_leave2 = np.expand_dims(self.M_basis_time['velocity'][:-2].T.dot(self.M_basis_time['velocity'][:-2]),

                                            (0, 1))
        psi_u_psi_u_shift1 = np.expand_dims(self.M_basis_time['velocity'][1:].T.dot(self.M_basis_time['velocity'][:-1]),

                                            (0, 1))
        psi_u_psi_u_shift2 = np.expand_dims(self.M_basis_time['velocity'][2:].T.dot(self.M_basis_time['velocity'][:-2]),

                                            (0, 1))
        psi_u_psi_u_leaveshift = np.expand_dims(self.M_basis_time['velocity'][1:-1, :].T.dot(self.M_basis_time['velocity'][:-2]),

                                                (0, 1))
        psi_u_psi_p = np.expand_dims(self.M_basis_time['velocity'].T.dot(self.M_basis_time['pressure']), (0, 1))
        psi_u_psi_p_shift1 = np.expand_dims(self.M_basis_time['velocity'][:-1].T.dot(self.M_basis_time['pressure'][1:]),

                                            (0, 1))
        psi_u_psi_p_shift2 = np.expand_dims(self.M_basis_time['velocity'][:-2].T.dot(self.M_basis_time['pressure'][2:]),

                                            (0, 1))

        psi_u_psi_lambda = [np.zeros(0)] * self.M_n_coupling
        psi_u_psi_lambda_shift1 = [np.zeros(0)] * self.M_n_coupling
        psi_u_psi_lambda_shift2 = [np.zeros(0)] * self.M_n_coupling
        psi_p_psi_lambda = [np.zeros(0)] * self.M_n_coupling
        psi_lambda_psi_lambda = [np.zeros(0)] * self.M_n_coupling
        sum_BnBn = np.zeros((self.M_N_space['velocity'], self.M_N_space['velocity']))
        for n in range(self.M_n_coupling):
            psi_u_psi_lambda[n] = np.expand_dims(self.M_basis_time['velocity'].T.dot(self.M_basis_time['lambda'][n]),
                                                 (0, 1))
            psi_u_psi_lambda_shift1[n] = np.expand_dims(self.M_basis_time['velocity'][:-1].T.dot(self.M_basis_time['lambda'][n][1:]),
                                                        (0, 1))
            psi_u_psi_lambda_shift2[n] = np.expand_dims(self.M_basis_time['velocity'][:-2].T.dot(self.M_basis_time['lambda'][n][2:]),
                                                        (0, 1))
            psi_p_psi_lambda[n] = np.expand_dims(self.M_basis_time['pressure'].T.dot(self.M_basis_time['lambda'][n]),
                                                 (0, 1))
            psi_lambda_psi_lambda[n] = np.expand_dims(self.M_basis_time['lambda'][n].T.dot(self.M_basis_time['lambda'][n]),
                                                      (0, 1))
            sum_BnBn += self.M_BB_matrix[n]

        sum_BnBn = np.expand_dims(sum_BnBn, (2, 3))

        logger.info('Constructing ST-PGRB blocks')

        block0 = (MAMA_matrix * psi_u_psi_u +
                  MM_matrix * (self.M_bdf[0] ** 2 * psi_u_psi_u_leave1 + self.M_bdf[1] ** 2 * psi_u_psi_u_leave2 +
                               self.M_bdf[0] * self.M_bdf[1] * psi_u_psi_u_leaveshift +
                               self.M_bdf[0] * self.M_bdf[1] * np.swapaxes(psi_u_psi_u_leaveshift, 2, 3)) +
                  MAM_matrix * (self.M_bdf[0] * psi_u_psi_u_shift1 + self.M_bdf[1] * psi_u_psi_u_shift2) +
                  np.swapaxes(MAM_matrix, 0, 1) * (self.M_bdf[0] * np.swapaxes(psi_u_psi_u_shift1, 2, 3) +
                                                   self.M_bdf[1] * np.swapaxes(psi_u_psi_u_shift2, 2, 3)) +
                  BdivBdiv_matrix * psi_u_psi_u +
                  sum_BnBn * psi_u_psi_u)
        self.M_Blocks[0] = np.reshape(np.transpose(block0, (0, 2, 1, 3)),
                                      (self.M_N['velocity'], self.M_N['velocity']))

        block1 = (MABdivT_matrix * psi_u_psi_p +
                  self.M_bdf[0] * MBdivT_matrix * psi_u_psi_p_shift1 +
                  self.M_bdf[1] * MBdivT_matrix * psi_u_psi_p_shift2) * (self.M_bdf_rhs * dt)
        self.M_Blocks[1] = np.reshape(np.transpose(block1, (0, 2, 1, 3)),
                                      (self.M_N['velocity'], self.M_N['pressure']))

        blocks2 = []
        for n in range(self.M_n_coupling):
            tmp = (MABT_matrix[n] * psi_u_psi_lambda[n] +
                   self.M_bdf[0] * MBT_matrix[n] * psi_u_psi_lambda_shift1[n] +
                   self.M_bdf[1] * MBT_matrix[n] * psi_u_psi_lambda_shift2[n]) * (self.M_bdf_rhs * dt)
            blocks2.append(np.reshape(np.transpose(tmp, (0, 2, 1, 3)),
                                      (self.M_N['velocity'], self.M_N['lambda'][n])))
        self.M_Blocks[2] = np.block([b2 for b2 in blocks2])

        block4 = (self.M_bdf_rhs * dt) ** 2 * BdivTBdivT_matrix * psi_p_psi_p
        self.M_Blocks[4] = np.reshape(np.transpose(block4, (0, 2, 1, 3)),
                                      (self.M_N['pressure'], self.M_N['pressure']))

        blocks5 = []
        for n in range(self.M_n_coupling):
            tmp = (self.M_bdf_rhs * dt) ** 2 * BdivTBT_matrix[n] * psi_p_psi_lambda[n]
            blocks5.append(np.reshape(np.transpose(tmp, (0, 2, 1, 3)),
                                      (self.M_N['pressure'], self.M_N['lambda'][n])))
        self.M_Blocks[5] = np.block([b5 for b5 in blocks5])

        blocks8 = [np.zeros(0)] * self.M_n_coupling
        for n in range(self.M_n_coupling):
            tmp = (self.M_bdf_rhs * dt) ** 2 * BTBT_matrix[n] * psi_lambda_psi_lambda[n]
            blocks8[n] = np.reshape(np.transpose(tmp, (0, 2, 1, 3)),
                                    (self.M_N['lambda'][n], self.M_N['lambda'][n]))
        self.M_Blocks[8] = scipy.linalg.block_diag(*blocks8)

        self.M_Blocks[3] = self.M_Blocks[1].T
        self.M_Blocks[6] = self.M_Blocks[2].T
        self.M_Blocks[7] = self.M_Blocks[5].T

        if 'clot' in self.M_parametrizations:
            self.M_Blocks_param_affine['clot'] = dict()
            self.M_Blocks_param_affine['clot'][0] = []
            self.M_Blocks_param_affine['clot'][1] = []
            self.M_Blocks_param_affine['clot'][2] = []
            self.M_Blocks_param_affine['clot'][3] = []
            self.M_Blocks_param_affine['clot'][6] = []

            # assembling linear terms
            for k in range(n_clots):
                block0 = (((np.swapaxes(MMclot_matrices[k], 0, 1) +
                            self.M_bdf_rhs * dt * np.swapaxes(AMclot_matrices[k], 0, 1) +
                            MMclot_matrices[k] + self.M_bdf_rhs * dt * AMclot_matrices[k]) *
                           self.M_bdf_rhs * dt * psi_u_psi_u) +
                          (MMclot_matrices[k] *
                           self.M_bdf_rhs * dt * (self.M_bdf[0] * np.swapaxes(psi_u_psi_u_shift1, 2, 3) +
                                                  self.M_bdf[1] * np.swapaxes(psi_u_psi_u_shift2, 2, 3))) +
                          (np.swapaxes(MMclot_matrices[k], 0, 1) *
                           self.M_bdf_rhs * dt * (self.M_bdf[0] * psi_u_psi_u_shift1 +
                                                  self.M_bdf[1] * psi_u_psi_u_shift2)))
                self.M_Blocks_param_affine['clot'][0].append(np.reshape(np.transpose(block0, (0, 2, 1, 3)),
                                                                         (self.M_N['velocity'], self.M_N['velocity'])))

                block1 = (self.M_bdf_rhs * dt) ** 2 * MclotBdivT_matrices[k] * psi_u_psi_p
                self.M_Blocks_param_affine['clot'][1].append(np.reshape(np.transpose(block1, (0, 2, 1, 3)),
                                                                         (self.M_N['velocity'], self.M_N['pressure'])))
                self.M_Blocks_param_affine['clot'][3].append(self.M_Blocks_param_affine['clot'][1][-1].T)

                blocks2 = []
                for n in range(self.M_n_coupling):
                    tmp = (self.M_bdf_rhs * dt) ** 2 * MclotBT_matrices[k][n] * psi_u_psi_lambda[n]
                    blocks2.append(np.reshape(np.transpose(tmp, (0, 2, 1, 3)),
                                              (self.M_N['velocity'], self.M_N['lambda'][n])))
                self.M_Blocks_param_affine['clot'][2].append(np.block([b2 for b2 in blocks2]))
                self.M_Blocks_param_affine['clot'][6].append(self.M_Blocks_param_affine['clot'][2][-1].T)

            # assembling mixed terms
            for k in range(n_clots):
                for kk in range(n_clots):
                    block0_mixed = (self.M_bdf_rhs * dt) ** 2 * MclotMclot_matrices[k][kk] * psi_u_psi_u
                    self.M_Blocks_param_affine['clot'][0].append(np.reshape(np.transpose(block0_mixed, (0, 2, 1, 3)),
                                                                             (self.M_N['velocity'], self.M_N['velocity'])))

        # self.set_param_functions()
        self.set_empty_blocks()

        return

    def set_empty_blocks(self):
        """
        MODIFY
        """

        if not self._has_IC():
            self.M_f_Blocks_no_param[0] = np.zeros(self.M_N['velocity'])
            self.M_f_Blocks_no_param[1] = np.zeros(self.M_N['pressure'])
            self.M_f_Blocks_no_param[2] = np.zeros(self.M_N_lambda_cumulative[-1])

        logger.debug(f"No empty blocks have to be set with the method {self.M_reduction_method}")

        return

    def save_rb_affine_decomposition(self, operators=None, blocks=None, f_blocks=None):
        """
        Saving RB blocks for the LHS matrix
        """

        if blocks is None:
            blocks = [0, 1, 2, 4, 5, 8]
        if f_blocks is None:
            f_blocks = [0, 1, 2]

        super().save_rb_affine_decomposition(operators=operators, blocks=blocks, f_blocks=f_blocks)

        return

    def import_rb_affine_components(self, operators=None, blocks=None, f_blocks=None):
        """
        Importing RB blocks for the LHS matrix
        """

        if operators is None:
            operators = {'Mat', 'f'}

        if blocks is None:
            blocks = [0, 1, 2, 4, 5, 8]
        if f_blocks is None:
            f_blocks = [0, 1, 2]

        import_failures_set = super().import_rb_affine_components(operators=operators, blocks=blocks, f_blocks=f_blocks)

        if 'Mat' in operators and 'Mat' not in import_failures_set:
            self.M_Blocks[3] = self.M_Blocks[1].T
            self.M_Blocks[6] = self.M_Blocks[2].T
            self.M_Blocks[7] = self.M_Blocks[5].T

        if not import_failures_set:
            self.set_param_functions()

        return import_failures_set

    def update_IC_terms(self, update_IC=False):
        """
        MODIFY
        """

        if update_IC:
            self.M_u0['velocity'] = np.dot(self.get_solution_field("velocity"), self.M_basis_time['velocity'][-2:].T).T

        dt = self.dt

        for i in range(3):
            self.M_f_Blocks_no_param[i] *= 0.0

        f_u_IC = (np.expand_dims((self.M_MM_matrix +
                                  self.M_bdf_rhs * dt * np.swapaxes(self.M_MA_matrix, 0, 1)).dot(self.M_u0['velocity'][0]), 1) *
                  np.expand_dims(-self.M_bdf[1] * self.M_basis_time['velocity'][0], 0) +
                  np.expand_dims(
                      (self.M_MM_matrix + self.M_bdf_rhs * dt * np.swapaxes(self.M_MA_matrix, 0, 1)).dot(self.M_u0['velocity'][1]), 1) *
                  np.expand_dims(- self.M_bdf[0] * self.M_basis_time['velocity'][0]
                                 - self.M_bdf[1] * self.M_basis_time['velocity'][1], 0) +
                  np.expand_dims(self.M_MM_matrix.dot(self.M_u0['velocity'][1]), 1) *
                  np.expand_dims(- self.M_bdf[0] * self.M_bdf[1] * self.M_basis_time['velocity'][0], 0)
                  ).flatten()
        self.M_f_Blocks_no_param[0] += f_u_IC

        f_p_IC = (np.expand_dims(self.M_MBdivT_matrix.T.dot(self.M_u0['velocity'][0]), 1) *
                  np.expand_dims(-self.M_bdf[1] * self.M_basis_time['pressure'][0], 0) +
                  np.expand_dims(self.M_MBdivT_matrix.T.dot(self.M_u0['velocity'][1]), 1) *
                  np.expand_dims(-self.M_bdf[0] * self.M_basis_time['pressure'][0]
                                 - self.M_bdf[1] * self.M_basis_time['pressure'][1], 0)
                  ).flatten()
        f_p_IC *= self.M_bdf_rhs * dt
        self.M_f_Blocks_no_param[1] += f_p_IC

        f_lambda_IC = [np.zeros(0)] * self.M_n_coupling
        for n in range(self.M_n_coupling):
            f_lambda_IC[n] = (np.expand_dims(self.M_MBT_matrix[n].T.dot(self.M_u0['velocity'][0]), 1) *
                              np.expand_dims(-self.M_bdf[1] * self.M_basis_time['lambda'][n][0], 0) +
                              np.expand_dims(self.M_MBT_matrix[n].T.dot(self.M_u0['velocity'][1]), 1) *
                              np.expand_dims(-self.M_bdf[0] * self.M_basis_time['lambda'][n][0]
                                             - self.M_bdf[1] * self.M_basis_time['lambda'][n][1], 0)
                              ).flatten()
            f_lambda_IC[n] *= self.M_bdf_rhs * dt
        self.M_f_Blocks_no_param[2] += np.hstack([elem for elem in f_lambda_IC])

        # add terms depending on the clot matrix
        if 'clot' in self.M_parametrizations:
            n_clots = self.get_clots_number()
            f_u_IC_clot = [np.zeros(0)] * n_clots

            self.M_f_Blocks_param_affine['clot'] = dict()
            self.M_f_Blocks_param_affine['clot'][0] = []

            for k in range(n_clots):
                f_u_IC_clot[k] = (
                        np.expand_dims(self.M_bdf_rhs * dt * np.swapaxes(self.M_MMclot_matrices[k], 0, 1).dot(self.M_u0['velocity'][0]), 1) *
                        np.expand_dims(-self.M_bdf[1] * self.M_basis_time[0], 0) +
                        np.expand_dims(self.M_bdf_rhs * dt * np.swapaxes(self.M_MMclot_matrices[k], 0, 1).dot(self.M_u0['velocity'][1]), 1) *
                        np.expand_dims(-self.M_bdf[0] * self.M_basis_time[0] - self.M_bdf[1] * self.M_basis_time[1], 0))
                self.M_f_Blocks_param_affine['clot'][0].append(f_u_IC_clot[k])

        self.M_f_Block = np.hstack([self._get_f_block(0), self._get_f_block(1), self._get_f_block(2)])

        return

    def build_rb_parametric_RHS(self, param):
        """
        MODIFY
        """

        super().build_rb_parametric_RHS(param)

        flow_rates = self.get_flow_rates(param)

        if 0 not in self.M_f_Blocks_param.keys():
            self.M_f_Blocks_param[0] = np.zeros(self.M_N['velocity'])

        for n in range(self.M_n_coupling):
            time_basis_flowrates = self.M_basis_time['velocity'].T.dot(flow_rates[:, n])
            time_basis_flowrates = np.expand_dims(time_basis_flowrates, 0)

            self.M_f_Blocks_param[0] += (np.expand_dims(self.M_RHS_vector[n], 1) * time_basis_flowrates).flatten()

        return

    def set_param_functions(self):
        """
        MODIFY
        """

        super().set_param_functions()

        n_clots = self.get_clots_number()

        if 'clot' in self.M_parametrizations:
            self.M_Blocks_param_affine_fun['clot'][0].extend([lambda mu, k=k, kk=kk: mu[k]*mu[kk]
                                                               for k in range(n_clots) for kk in range(n_clots)])

        return

    def set_paths(self, _snapshot_matrix=None, _basis_matrix=None,
                  _affine_components=None,
                  _fom_structures=None, _reduced_structures=None,
                  _generalized_coords=None, _results=None, _used_norm='l2'):
        """
        Set data paths
        """

        self.M_snapshots_path = _snapshot_matrix
        self.M_fom_structures_path = _fom_structures
        self.M_basis_path = _basis_matrix

        self.M_affine_components_path = _affine_components + '_' + _used_norm
        self.M_reduced_structures_path = _reduced_structures + '_' + _used_norm
        self.M_generalized_coords_path = _generalized_coords + '_' + _used_norm
        self.M_results_path = _results

        return

    def build_rb_approximation(self, _ns, _n_weak_io, _mesh_name, _tolerances,
                               _space_projection='standard', prob=None, ss_ratio=1, _used_norm='l2'):
        """
        MODIFY
        """

        self.M_used_norm = _used_norm

        super().build_rb_approximation(_ns, _n_weak_io, _mesh_name, _tolerances,
                                       _space_projection=_space_projection, prob=prob, ss_ratio=ss_ratio)

        return

    def time_marching(self, *args, **kwargs):
        raise NotImplementedError("Time marching method available only with SRB-TFO method")

    def test_time_marching(self, *args, **kwargs):
        raise NotImplementedError("Time marching method available only with SRB-TFO method")


__all__ = [
    "RbManagerSTPGRBStokes"
]
