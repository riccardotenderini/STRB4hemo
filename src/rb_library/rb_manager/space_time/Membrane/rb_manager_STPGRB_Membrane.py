#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 5 14:26:50 2023
@author: Riccardo Tenderini
@mail: riccardo.tenderini@epfl.ch
"""

import os
import numpy as np
from scipy.sparse import diags

import src.utils.array_utils as arr_utils

import src.rb_library.rb_manager.space_time.Navier_Stokes.rb_manager_STPGRB_Navier_Stokes as rbmstpgrbNS
import src.rb_library.rb_manager.space_time.Membrane.rb_manager_space_time_Membrane as rbmstM

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../../../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class RbManagerSTPGRBMembrane(rbmstpgrbNS.RbManagerSTPGRBNavierStokes,
                              rbmstM.RbManagerSpaceTimeMembrane):
    """
    MODIFY
    """

    def __init__(self, _fom_problem, _affine_decomposition=None):
        """
        MODIFY
        """

        super().__init__(_fom_problem, _affine_decomposition=_affine_decomposition)

        self.M_MbdM_matrix = np.zeros(0)
        self.M_MbdA_matrix = np.zeros(0)
        self.M_MbdMbd_matrix = np.zeros(0)
        self.M_MbdBdivT_matrix = np.zeros(0)
        # self.M_MbdBT_matrices = [np.zeros(0)] * self.M_n_coupling

        self.M_MbdAbd_matrices = np.zeros(0)
        self.M_AbdM_matrices = np.zeros(0)
        self.M_AbdA_matrices = np.zeros(0)
        self.M_AbdBdivT_matrices = np.zeros(0)
        self.M_AbdAbd_matrices = np.zeros(0)
        # self.M_AbdBT_matrices = [np.zeros(0)] * self.M_n_coupling

        self.M_MbdK_vectors = np.zeros(0)
        self.M_AbdK_vectors = np.zeros(0)

        self.M_NLterm_offline_time_tensor_uud = np.zeros(0)

        logger.warning("ST-PGRB method for the coupled momentum model is likely to suffer from ill-conditioning!")
        logger.warning("Resistance BCs are not yet supported for ST-PGRB!")
        logger.warning("Wall elasticity is not yet supported for ST-PGRB!")
        logger.warning("Non-zero displacement initial conditions are not yet supported for ST-PGRB!")

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

        rbmstpgrbNS.RbManagerSTPGRBNavierStokes.build_ST_basis(self, _tolerances, which=which)

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

    def __update_velocity_norm(self, mat, inverse=False):
        """
        Update velocity norm matrix to improve conditioning
        """

        factor = -1.0 if inverse else 1.0

        if self.M_used_norm == 'X':
            self.M_norm_matrices['velocity'] += factor * mat
        elif self.M_used_norm == 'P':
            self.M_inv_norm['velocity'] += diags(factor / mat.diagonal(),  format='csc')
        else:
            pass

        return

    def _assemble_right_reduced_structures(self):
        """
        MODIFY
        """

        if not self.check_norm_matrices():
            self.get_norm_matrices()
        self.set_projection_norm()

        structures = {'A', 'M', 'Bdiv', 'B', 'RHS', 'Abd', 'Mbd'}
        FEM_matrices = rbmstM.RbManagerSpaceTimeMembrane.import_FEM_structures(self, structures=structures)

        logger.info("Projecting FEM structures onto the reduced subspace in space; this may take some time...")

        # temporary updating velocity norm to get a well conditioned system
        # avg_bd_matrix = FEM_matrices['Mbd'] + sum(FEM_matrices['Abd'])  # TODO: remove BCs from these matrices !!
        # self.__update_velocity_norm(avg_bd_matrix)

        matrices = rbmstpgrbNS.RbManagerSTPGRBNavierStokes._assemble_right_reduced_structures(self)

        matrices['Abd'], matrices['XAbd'] = [], []
        for Abd_matrix in FEM_matrices['Abd']:
            matrices['Abd'].append(arr_utils.sparse_matrix_matrix_mul(Abd_matrix, self.M_basis_space['velocity']))
            matrices['XAbd'].append(self.inverse_norm_premultiplication(matrices['Abd'][-1], 'velocity'))

        matrices['Mbd'] = arr_utils.sparse_matrix_matrix_mul(FEM_matrices['Mbd'], self.M_basis_space['velocity'])
        matrices['XMbd'] = self.inverse_norm_premultiplication(matrices['Mbd'], 'velocity')

        # reset velocity norm to original value
        # self.__update_velocity_norm(avg_bd_matrix, inverse=True)

        return matrices

    def assemble_reduced_structures(self, _space_projection='standard', matrices=None,
                                    _tolerances=None, N_components=None):
        """
        MODIFY
        """

        if matrices is None:
            matrices = self._assemble_right_reduced_structures()

        rbmstpgrbNS.RbManagerSTPGRBNavierStokes.assemble_reduced_structures(self, _space_projection=_space_projection,
                                                                            matrices=matrices, _tolerances=_tolerances,
                                                                            N_components=N_components)

        self.M_MbdM_matrix = matrices['Mbd'].T.dot(matrices['XM'])
        self.M_MbdA_matrix = matrices['Mbd'].T.dot(matrices['XA'])
        self.M_MbdMbd_matrix = matrices['Mbd'].T.dot(matrices['XMbd'])
        self.M_MbdBdivT_matrix = matrices['Mbd'].T.dot(matrices['XBdivT'])
        # self.M_MbdBT_matrices = [np.zeros(self.M_N_space['velocity'], self.M_N_space['lambda'][n])
        #                          for n in range(self.M_n_coupling)]
        # for n in range(self.M_n_coupling):
        #     self.M_MbdBT_matrices[n] = matrices['Mbd'].T.dot(matrices['XBT'][n])

        K = len(matrices['Abd'])
        self.M_MbdAbd_matrices = np.zeros((K, self.M_N_space['velocity'], self.M_N_space['velocity']))
        self.M_AbdM_matrices = np.zeros((K, self.M_N_space['velocity'], self.M_N_space['velocity']))
        self.M_AbdA_matrices = np.zeros((K, self.M_N_space['velocity'], self.M_N_space['velocity']))
        self.M_AbdBdivT_matrices = np.zeros((K, self.M_N_space['velocity'], self.M_N_space['pressure']))
        self.M_AbdAbd_matrices = np.zeros((K, K, self.M_N_space['velocity'], self.M_N_space['velocity']))
        # self.M_AbdBT_matrices = [np.zeros((K, self.M_N_space['velocity'], self.M_N_space['lambda'][n]))
        #                          for n in range(self.M_n_coupling)]

        for k in range(K):
            self.M_MbdAbd_matrices[k] = matrices['Mbd'].T.dot(matrices['XAbd'][k])
            self.M_AbdM_matrices[k] = matrices['Abd'][k].T.dot(matrices['XMbd'])
            self.M_AbdA_matrices[k] = matrices['Abd'][k].T.dot(matrices['XA'])
            self.M_AbdBdivT_matrices[k] = matrices['Abd'][k].T.dot(matrices['XBdivT'])
            for k2 in range(len(matrices['Abd'])):
                self.M_AbdAbd_matrices[k][k2] = matrices['Abd'][k].T.dot(matrices['XAbd'][k2])
            # for n in range(self.M_n_coupling):
            #     self.M_AbdBT_matrices[n][k] = matrices['Abd'][k].T.dot(matrices['XBT'][n])

        if self.M_save_offline_structures:
            self.save_reduced_structures()

        return

    def save_reduced_structures(self):
        """
        MODIFY
        """

        rbmstpgrbNS.RbManagerSTPGRBNavierStokes.save_reduced_structures(self)

        np.save(os.path.join(self.M_reduced_structures_path, 'MbdM_rb.npy'), self.M_MbdM_matrix)
        np.save(os.path.join(self.M_reduced_structures_path, 'MbdA_rb.npy'), self.M_MbdA_matrix)
        np.save(os.path.join(self.M_reduced_structures_path, 'MbdMbd_rb.npy'), self.M_MbdMbd_matrix)
        np.save(os.path.join(self.M_reduced_structures_path, 'MbdBdivT_rb.npy'), self.M_MbdBdivT_matrix)
        # for n in range(self.M_n_coupling):
        #     np.save(os.path.join(self.M_reduced_structures_path, f'MbdBT{n}_rb.npy'), self.M_MbdBT_matrices[n])

        K = len(self.M_MbdAbd_matrices)
        for k in range(K):
            np.save(os.path.join(self.M_reduced_structures_path, f'MbdAbd{k}_rb.npy'), self.M_MbdAbd_matrices[k])
            np.save(os.path.join(self.M_reduced_structures_path, f'Abd{k}M_rb.npy'), self.M_AbdM_matrices[k])
            np.save(os.path.join(self.M_reduced_structures_path, f'Abd{k}A_rb.npy'), self.M_AbdA_matrices[k])
            np.save(os.path.join(self.M_reduced_structures_path, f'Abd{k}BdivT_rb.npy'), self.M_AbdBdivT_matrices[k])
            for k2 in range(K):
                np.save(os.path.join(self.M_reduced_structures_path, f'Abd{k}Abd{k2}_rb.npy'), self.M_AbdAbd_matrices[k][k2])
            # for n in range(self.M_n_coupling):
            #     np.save(os.path.join(self.M_reduced_structures_path, f'Abd{k}BT{n}_rb.npy'), self.M_AbdBT_matrices[k][n])

        return

    def import_reduced_structures(self, _tolerances=None, N_components=None, _space_projection='standard'):
        """
        MODIFY
        """

        self.reset_reduced_structures()

        try:
            rbmstpgrbNS.RbManagerSTPGRBNavierStokes.import_reduced_structures(self, _tolerances=_tolerances,
                                                                              N_components=N_components,
                                                                              _space_projection=_space_projection)

            self.M_MbdM_matrix = np.load(os.path.join(self.M_reduced_structures_path, 'MbdM_rb.npy'))
            self.M_MbdA_matrix = np.load(os.path.join(self.M_reduced_structures_path, 'MbdA_rb.npy'))
            self.M_MbdMbd_matrix = np.load(os.path.join(self.M_reduced_structures_path, 'MbdMbd_rb.npy'))
            self.M_MbdBdivT_matrix = np.load(os.path.join(self.M_reduced_structures_path, 'MbdBdivT_rb.npy'))
            # for n in range(self.M_n_coupling):
            #     self.M_MbdBT_matrices[n] = np.load(os.path.join(self.M_reduced_structures_path, f'MbdBT{n}_rb.npy'))

            K = 3
            pathMbd = lambda cnt: os.path.join(self.M_reduced_structures_path, f"MbdAbd{cnt}_rb.npy")
            pathM = lambda cnt: os.path.join(self.M_reduced_structures_path, f"Abd{cnt}M_rb.npy")
            pathA = lambda cnt: os.path.join(self.M_reduced_structures_path, f"Abd{cnt}A_rb.npy")
            pathBdivT = lambda cnt: os.path.join(self.M_reduced_structures_path, f"Abd{cnt}BdivT_rb.npy")
            pathAbd = lambda cnt1, cnt2: os.path.join(self.M_reduced_structures_path, f"Abd{cnt1}Abd{cnt2}_rb.npy")
            # pathBT = lambda cnt, m: os.path.join(self.M_reduced_structures_path, f"Abd{cnt}BT{m}_rb.npy")

            self.M_MbdAbd_matrices = np.zeros((K, self.M_N_space['velocity'], self.M_N_space['velocity']))
            self.M_AbdM_matrices = np.zeros((K, self.M_N_space['velocity'], self.M_N_space['velocity']))
            self.M_AbdA_matrices = np.zeros((K, self.M_N_space['velocity'], self.M_N_space['velocity']))
            self.M_AbdBdivT_matrices = np.zeros((K, self.M_N_space['velocity'], self.M_N_space['pressure']))
            self.M_AbdAbd_matrices = np.zeros((K, K, self.M_N_space['velocity'], self.M_N_space['velocity']))
            # self.M_AbdBT_matrices = [np.zeros((K, self.M_N_space['velocity'], self.M_N_space['lambda'][n]))
            #                          for n in range(self.M_n_coupling)]

            for k in range(K):
                self.M_MbdAbd_matrices[k] = np.load(pathMbd(k))
                self.M_AbdM_matrices[k] = np.load(pathM(k))
                self.M_AbdA_matrices[k] = np.load(pathA(k))
                self.M_AbdBdivT_matrices[k] = np.load(pathBdivT(k))

                for kk in range(K):
                    self.M_AbdAbd_matrices[k][kk] = np.load(pathAbd(k, kk))

                # for n in range(self.M_n_coupling):
                #     self.M_AbdBT_matrices[n][k] = np.load(pathBT(k, n))

            import_success = True

        except (OSError, FileNotFoundError, AssertionError) as e:
            logger.error(f"Error {e}: failed to import the reduced structures!")
            import_success = False

        return import_success

    def reset_reduced_structures(self):
        """
        MODIFY
        """

        rbmstpgrbNS.RbManagerSTPGRBNavierStokes.reset_reduced_structures(self)

        self.M_MbdM_matrix = np.zeros(0)
        self.M_MbdA_matrix = np.zeros(0)
        self.M_MbdMbd_matrix = np.zeros(0)
        self.M_MbdBdivT_matrix = np.zeros(0)
        # self.M_MbdBT_matrices = [np.zeros(0)] * self.M_n_coupling

        self.M_MbdAbd_matrices = np.zeros(0)
        self.M_AbdM_matrices = np.zeros(0)
        self.M_AbdA_matrices = np.zeros(0)
        self.M_AbdBdivT_matrices = np.zeros(0)
        self.M_AbdAbd_matrices = np.zeros(0)
        # self.M_AbdBT_matrices = [np.zeros(0)] * self.M_n_coupling

        self.M_MbdK_vectors = np.zeros(0)
        self.M_AbdK_vectors = np.zeros(0)

        self.M_NLterm_offline_time_tensor_uud = np.zeros(0)

        return

    def build_rb_nonparametric_LHS(self):
        """
        MODIFY
        """

        rbmstpgrbNS.RbManagerSTPGRBNavierStokes.build_rb_nonparametric_LHS(self)

        dt = self.dt

        expand_space = lambda X: np.expand_dims(X, (2, 3))
        expand_time = lambda X: np.expand_dims(X, (0, 1))
        transpose_space = lambda X: np.swapaxes(X, 0, 1)
        transpose_time = lambda X: np.swapaxes(X, 2, 3)
        reshape_space_time = lambda X, N1, N2: np.reshape(np.transpose(X, (0, 2, 1, 3)), (N1, N2))

        MbdM_matrix = expand_space(self.M_MbdM_matrix)
        MbdA_matrix = expand_space(self.M_MbdA_matrix)
        MbdMbd_matrix = expand_space(self.M_MbdMbd_matrix)
        MbdBdivT_matrix = expand_space(self.M_MbdBdivT_matrix)
        # MbdBT_matrices = [np.zeros(0)] * self.M_n_coupling
        # for n in range(self.M_n_coupling):
        #     MbdBT_matrices[n] = expand_space(self.M_MbdBT_matrices[n])

        K = len(self.M_AbdM_matrices)
        MbdAbd_matrices = np.zeros((K, self.M_N_space['velocity'], self.M_N_space['velocity'], 1, 1))
        AbdM_matrices = np.zeros((K, self.M_N_space['velocity'], self.M_N_space['velocity'], 1, 1))
        AbdA_matrices = np.zeros((K, self.M_N_space['velocity'], self.M_N_space['velocity'], 1, 1))
        AbdBdivT_matrices = np.zeros((K, self.M_N_space['velocity'], self.M_N_space['pressure'], 1, 1))
        AbdAbd_matrices = np.zeros((K, K, self.M_N_space['velocity'], self.M_N_space['velocity'], 1, 1))
        # AbdBT_matrices = [np.zeros((K, self.M_N_space['velocity'], self.M_N_space['lambda'][n], 1, 1))
        #                   for n in range(self.M_n_coupling)]
        for k in range(K):
            MbdAbd_matrices[k] = expand_space(self.M_MbdAbd_matrices[k])
            AbdA_matrices[k] = expand_space(self.M_AbdA_matrices[k])
            AbdM_matrices[k] = expand_space(self.M_AbdM_matrices[k])
            AbdBdivT_matrices[k] = expand_space(self.M_AbdBdivT_matrices[k])
            for k2 in range(K):
                AbdAbd_matrices[k, k2] = expand_space(self.M_AbdAbd_matrices[k][k2])
            # for n in range(self.M_n_coupling):
            #     AbdBT_matrices[n][k] = expand_space(self.M_AbdBT_matrices[n][k])

        # AUXILIARY MATRICES
        MbdMA_matrix = MbdM_matrix + (self.M_bdf_rhs * dt) * MbdA_matrix
        AbdMA_matrices = [AbdM_matrices[k] + (self.M_bdf_rhs * dt) * AbdA_matrices[k] for k in range(K)]

        # ASSEMBLING TEMPORAL QUANTITIES
        psi_u_psi_u = expand_time(np.eye(self.M_N_time['velocity']))
        psi_u_psi_u_leave1 = expand_time(self.M_basis_time['velocity'][:-1].T.dot(self.M_basis_time['velocity'][:-1]))
        psi_u_psi_u_leave2 = expand_time(self.M_basis_time['velocity'][:-2].T.dot(self.M_basis_time['velocity'][:-2]))
        psi_u_psi_u_shift1 = expand_time(self.M_basis_time['velocity'][1:].T.dot(self.M_basis_time['velocity'][:-1]))
        psi_u_psi_u_shift2 = expand_time(self.M_basis_time['velocity'][2:].T.dot(self.M_basis_time['velocity'][:-2]))
        psi_u_psi_u_leaveshift = expand_time(self.M_basis_time['velocity'][1:-1].T.dot(self.M_basis_time['velocity'][:-2]))

        psi_u_psi_p = expand_time(self.M_basis_time['velocity'].T.dot(self.M_basis_time['pressure']))
        psi_u_psi_p_shift1 = expand_time(self.M_basis_time['velocity'][:-1].T.dot(self.M_basis_time['pressure'][1:]))
        psi_u_psi_p_shift2 = expand_time(self.M_basis_time['velocity'][:-2].T.dot(self.M_basis_time['pressure'][2:]))

        psi_u_psi_lambda = [np.zeros(0)] * self.M_n_coupling
        psi_u_psi_lambda_shift1 = [np.zeros(0)] * self.M_n_coupling
        psi_u_psi_lambda_shift2 = [np.zeros(0)] * self.M_n_coupling
        for n in range(self.M_n_coupling):
            psi_u_psi_lambda[n] = expand_time(self.M_basis_time['velocity'].T.dot(self.M_basis_time['lambda'][n]))
            psi_u_psi_lambda_shift1[n] = expand_time(self.M_basis_time['velocity'][:-1].T.dot(self.M_basis_time['lambda'][n][1:]))
            psi_u_psi_lambda_shift2[n] = expand_time(self.M_basis_time['velocity'][:-2].T.dot(self.M_basis_time['lambda'][n][2:]))

        psi_d_psi_u = expand_time(self.M_basis_time['displacement'].T.dot(self.M_basis_time['velocity']))
        psi_d_psi_u_shift1 = expand_time(self.M_basis_time['displacement'][1:].T.dot(self.M_basis_time['velocity'][:-1]))
        psi_d_psi_u_shift2 = expand_time(self.M_basis_time['displacement'][2:].T.dot(self.M_basis_time['velocity'][:-2]))
        psi_d_psi_d = expand_time(self.M_basis_time['displacement'].T.dot(self.M_basis_time['displacement']))
        psi_d_psi_p = expand_time(self.M_basis_time['displacement'].T.dot(self.M_basis_time['pressure']))

        # ASSEMBLING ST-PGRB structure blocks
        self.M_Blocks_param_affine['structure'] = dict()
        self.M_Blocks_param_affine['structure'][0] = []
        self.M_Blocks_param_affine['structure'][1] = []
        self.M_Blocks_param_affine['structure'][3] = []

        block1 = reshape_space_time(MbdBdivT_matrix * (psi_u_psi_p +
                                                       self.M_bdf[0] * psi_u_psi_p_shift1 +
                                                       self.M_bdf[1] * psi_u_psi_p_shift2),
                                    self.M_N['velocity'], self.M_N['pressure']) * (self.M_bdf_rhs * dt)
        self.M_Blocks_param_affine['structure'][1].append(block1 * 1.0)
        self.M_Blocks_param_affine['structure'][3].append(block1.T * 1.0)
        for k in range(K):
            block1_k = reshape_space_time(AbdBdivT_matrices[k] * psi_d_psi_p, self.M_N['velocity'], self.M_N['pressure']) * (self.M_bdf_rhs * dt) ** 2
            # block1_k = reshape_space_time(AbdBdivT_matrices[k] * psi_u_psi_p, self.M_N['velocity'], self.M_N['pressure']) * (self.M_bdf_rhs_coeff*dt)**2
            self.M_Blocks_param_affine['structure'][1].append(block1_k * 1.0)
            self.M_Blocks_param_affine['structure'][3].append(block1_k.T * 1.0)

        block0_0 = (transpose_space(MbdMA_matrix) * (psi_u_psi_u + self.M_bdf[0] * psi_u_psi_u_shift1 + self.M_bdf[1] * psi_u_psi_u_shift2) +
                    MbdMA_matrix * (psi_u_psi_u + self.M_bdf[0] * transpose_time(psi_u_psi_u_shift1) +
                                    self.M_bdf[1] * transpose_time(psi_u_psi_u_shift2)) +
                    MbdM_matrix * (self.M_bdf[0] * psi_u_psi_u_shift1 + self.M_bdf[1] * psi_u_psi_u_shift2 + self.M_bdf[0] ** 2 * psi_u_psi_u_leave1 +
                                   self.M_bdf[1] ** 2 * psi_u_psi_u_leave2 + self.M_bdf[0] * self.M_bdf[1] * psi_u_psi_u_leaveshift +
                                   self.M_bdf[0] * self.M_bdf[1] * transpose_time(psi_u_psi_u_leaveshift)) +
                    transpose_space(MbdM_matrix) * (self.M_bdf[0] * transpose_time(psi_u_psi_u_shift1) +
                                                    self.M_bdf[1] * transpose_time(psi_u_psi_u_shift2) + self.M_bdf[0] ** 2 * psi_u_psi_u_leave1 +
                                                    self.M_bdf[1] ** 2 * psi_u_psi_u_leave2 + self.M_bdf[0] * self.M_bdf[1] * psi_u_psi_u_leaveshift +
                                                    self.M_bdf[0] * self.M_bdf[1] * transpose_time(psi_u_psi_u_leaveshift)))
        self.M_Blocks_param_affine['structure'][0].append(reshape_space_time(block0_0 * 1.0, self.M_N['velocity'], self.M_N['velocity']))

        for k in range(K):
            block0_1k = (transpose_space(AbdMA_matrices[k]) * transpose_time(psi_d_psi_u) +
                         AbdMA_matrices[k] * psi_d_psi_u +
                         AbdM_matrices[k] * (self.M_bdf[0] * psi_d_psi_u_shift1 + self.M_bdf[1] * psi_d_psi_u_shift2) +
                         # AbdMA_matrices[k] * psi_u_psi_u +
                         # AbdM_matrices[k] * (self.M_bdf_coeffs[0] * psi_u_psi_u_shift1 + self.M_bdf_coeffs[1] * psi_u_psi_u_shift2) +
                         transpose_space(AbdM_matrices[k]) * (self.M_bdf[0] * transpose_time(psi_d_psi_u_shift1) +
                                                              self.M_bdf[1] * transpose_time(psi_d_psi_u_shift2))) * (self.M_bdf_rhs * dt)
            self.M_Blocks_param_affine['structure'][0].append(reshape_space_time(block0_1k * 1.0, self.M_N['velocity'], self.M_N['velocity']))

        block0_2 = MbdMbd_matrix * (psi_u_psi_u + self.M_bdf[0] ** 2 * psi_u_psi_u_leave1 + self.M_bdf[1] ** 2 * psi_u_psi_u_leave2 +
                                    self.M_bdf[0] * self.M_bdf[1] * psi_u_psi_u_leaveshift + self.M_bdf[0] * self.M_bdf[1] * transpose_time(psi_u_psi_u_leaveshift) +
                                    self.M_bdf[0] * transpose_time(psi_u_psi_u_shift1) + self.M_bdf[1] * transpose_time(psi_u_psi_u_shift2) +
                                    self.M_bdf[0] * psi_u_psi_u_shift1 + self.M_bdf[1] * psi_u_psi_u_shift2)
        self.M_Blocks_param_affine['structure'][0].append(reshape_space_time(block0_2 * 1.0, self.M_N['velocity'], self.M_N['velocity']))

        for k in range(K):
            block0_3k = (MbdAbd_matrices[k] * (transpose_time(psi_d_psi_u) + self.M_bdf[0] * transpose_time(psi_d_psi_u_shift1) +
                                               self.M_bdf[1] * transpose_time(psi_d_psi_u_shift2)) +
                         transpose_space(MbdAbd_matrices[k]) * (psi_d_psi_u + self.M_bdf[0] * psi_d_psi_u_shift1 +
                                                                self.M_bdf[1] * psi_d_psi_u_shift2)
                         # transpose_space(MbdAbd_matrices[k]) * (psi_u_psi_u + self.M_bdf_coeffs[0] * psi_u_psi_u_shift1 +
                         #                                        self.M_bdf_coeffs[1] * psi_u_psi_u_shift2)
                         ) * (self.M_bdf_rhs * dt)
            self.M_Blocks_param_affine['structure'][0].append(reshape_space_time(block0_3k * 1.0, self.M_N['velocity'], self.M_N['velocity']))

        for k in range(K):
            for kk in range(K):
                block0_4k = AbdAbd_matrices[k, kk] * psi_d_psi_d * (self.M_bdf_rhs * dt) ** 2
                # block0_4k = AbdAbd_matrices[k, kk] * transpose_time(psi_d_psi_u) * (self.M_bdf_rhs_coeff * dt)**2
                self.M_Blocks_param_affine['structure'][0].append(reshape_space_time(block0_4k * 1.0, self.M_N['velocity'], self.M_N['velocity']))

        return

    def set_param_functions(self):
        """
        MODIFY
        """

        rbmstM.RbManagerSpaceTimeMembrane.set_param_functions(self)

        if 'structure' in self.M_parametrizations:

            self.M_Blocks_param_affine_fun['structure'][0].extend([lambda mu: mu[0]**2])
            self.M_Blocks_param_affine_fun['structure'][0].extend([lambda mu, k=k: mu[0]*mu[k+1] for k in range(3)])
            self.M_Blocks_param_affine_fun['structure'][0].extend([lambda mu, k=k, kk=kk: mu[k+1]*mu[kk+1]
                                                                   for k in range(3) for kk in range(3)])

            if self._has_IC():  # TODO: to remove !!
                self.M_f_Blocks_param_affine_fun['structure'][0].extend([lambda mu: mu[0]**2])
                self.M_f_Blocks_param_affine_fun['structure'][0].extend([lambda mu, k=k: mu[0]*mu[k+1]
                                                                         for k in range(3)])

        return

    def compute_NLterm_offline_quantities_space(self):
        """
        MODIFY
        """

        if self.M_newton_specifics['use convective jacobian']:
            raise NotImplementedError("Convective Jacobian terms not implemented for ST-PGRB with membrane.")

        if not self.check_norm_matrices():
            self.get_norm_matrices()
        self.set_projection_norm()

        structures = {'Abd', 'Mbd'}
        FEM_matrices = self.import_FEM_structures(structures=structures)
        K = len(FEM_matrices['Abd'])

        # temporary updating velocity norm to get a well conditioned system
        avg_bd_matrix = FEM_matrices['Mbd'] + sum(FEM_matrices['Abd'])
        self.__update_velocity_norm(avg_bd_matrix)

        rbmstpgrbNS.RbManagerSTPGRBNavierStokes.compute_NLterm_offline_quantities_space(self)

        M_Mbd_red = arr_utils.sparse_matrix_matrix_mul(FEM_matrices['Mbd'], self.M_basis_space['velocity'])
        M_Abd_red = [arr_utils.sparse_matrix_matrix_mul(Abd, self.M_basis_space['velocity']) for Abd in FEM_matrices['Abd']]

        # TODO: these matrices have already been assembled in call to NS method !!
        M_K_red = np.array([[self.M_NLTerm_affine_components[i][j] for i in range(self.M_N_components_NLTerm)]
                            for j in range(self.M_N_components_NLTerm)])
        XK_red = np.zeros((self.M_N_components_NLTerm, self.M_N_components_NLTerm, self.M_Nh['velocity']))
        for i1 in range(self.M_N_components_NLTerm):
            for i2 in range(self.M_N_components_NLTerm):
                XK_red[i1, i2] = self.inverse_norm_premultiplication(M_K_red[i1, i2], 'velocity')

        self.M_MbdK_vectors = np.zeros((self.M_N_components_NLTerm, self.M_N_components_NLTerm, self.M_N_space['velocity']))
        self.M_AbdK_vectors = np.zeros((K, self.M_N_components_NLTerm, self.M_N_components_NLTerm, self.M_N_space['velocity']))

        for i1 in range(self.M_N_components_NLTerm):
            for i2 in range(self.M_N_components_NLTerm):
                self.M_MbdK_vectors[i1, i2] = M_Mbd_red.T.dot(XK_red[i1, i2])
                for k in range(K):
                    self.M_AbdK_vectors[k, i1, i2] = M_Abd_red[k].T.dot(XK_red[i1, i2])

        # reset velocity norm to original value
        self.__update_velocity_norm(avg_bd_matrix, inverse=True)

        return

    def compute_NLterm_offline_quantities_time(self):
        """
        MODIFY
        """

        rbmstpgrbNS.RbManagerSTPGRBNavierStokes.compute_NLterm_offline_quantities_time(self)

        self.M_NLterm_offline_time_tensor_uud = np.zeros((self.M_N_time['velocity'],
                                                          self.M_N_time['velocity'],
                                                          self.M_N_time['displacement']))
        for i in range(self.M_N_time['velocity']):
            for j in range(self.M_N_time['velocity']):
                for k in range(self.M_N_time['displacement']):
                    self.M_NLterm_offline_time_tensor_uud[i, j, k] = np.sum(self.M_basis_time['velocity'][:, i] *
                                                                            self.M_basis_time['velocity'][:, j] *
                                                                            self.M_basis_time['displacement'][:, k])

        return

    def __update_K2(self, u_hat, param):
        """
        Update the convective term with membrane contributions
        """

        logger.debug("Updating K2 with membrane contributions")

        expand_space = lambda X: np.expand_dims(X, 3)
        expand_time = lambda X: np.expand_dims(X, (0, 1))

        dt = self.dt
        K = len(self.M_AbdK_vectors)

        u_hat2 = np.expand_dims(np.swapaxes(np.multiply.outer(u_hat, u_hat), 1, 2), 4)  # (n_c, n_c, n_t, n_t, 1)

        param_map = self.differentiate_parameters(param)

        time_contrib_uuu = expand_time(np.array(self.M_NLterm_offline_time_tensor_uuu))  # (1, 1, n_t, n_t, n_t)
        time_contrib_u1u1u = expand_time(np.array(self.M_NLterm_offline_time_tensor_u1u1u))  # (1, 1, n_t, n_t, n_t)
        time_contrib_u2u2u = expand_time(np.array(self.M_NLterm_offline_time_tensor_u2u2u))  # (1, 1, n_t, n_t, n_t)
        time_contrib_uud = expand_time(np.array(self.M_NLterm_offline_time_tensor_uud))  # (1, 1, n_t, n_t, n_t)

        MbdK = np.expand_dims(np.array(self.M_MbdK_vectors), 3)  # (n_c, n_c, n_s, 1)
        AbdK = [np.zeros(0)] * K
        for k in range(K):
            AbdK[k] = expand_space(np.array(self.M_AbdK_vectors[k]))  # (n_c, n_c, n_s, 1)

        tmp_uuu = np.expand_dims(np.sum(u_hat2 * time_contrib_uuu, axis=(2, 3)), 2)  # (n_c, n_c, 1, n_t)
        tmp_u1u1u = np.expand_dims(np.sum(u_hat2 * time_contrib_u1u1u, axis=(2, 3)), 2)  # (n_c, n_c, 1, n_t)
        tmp_u2u2u = np.expand_dims(np.sum(u_hat2 * time_contrib_u2u2u, axis=(2, 3)), 2)  # (n_c, n_c, 1, n_t)
        tmp_uud = np.expand_dims(np.sum(u_hat2 * time_contrib_uud, axis=(2, 3)), 2)  # (n_c, n_c, 1, n_t)

        bd_K_uuu = [np.zeros(self.M_N['velocity'])] * (K+1)
        bd_K_uuu[0] = (np.sum(MbdK * tmp_uuu, axis=(0, 1)) +
                       np.sum(MbdK * tmp_u1u1u, axis=(0, 1)) +
                       np.sum(MbdK * tmp_u2u2u, axis=(0, 1))).flatten() * (self.M_bdf_rhs * dt)
        for k in range(K):
            bd_K_uuu[k+1] = np.sum(AbdK[k] * tmp_uud, axis=(0, 1)).flatten() * (self.M_bdf_rhs * dt) ** 2
            # bd_K_uuu[k+1] = np.sum(AbdK[k] * tmp_uuu, axis=(0, 1)).flatten() * (self.M_bdf_rhs_coeff * dt)**2

        self.M_K2_Blocks[0] += sum([_fun(param_map['structure']) * _vector
                                    for _vector, _fun in zip(bd_K_uuu,
                                                             self.M_Blocks_param_affine_fun['structure'][0][:K+1])])

        return

    def update_IC_terms(self, update_IC=False):
        """
        MODIFY
        """

        rbmstpgrbNS.RbManagerSTPGRBNavierStokes.update_IC_terms(self, update_IC=update_IC)

        dt = self.dt
        K = 3

        if 'structure' in self.M_parametrizations:
            self.M_f_Blocks_param_affine['structure'] = dict()
            self.M_f_Blocks_param_affine['structure'][0] = []

            fblock0_0 = (np.expand_dims((self.M_MbdM_matrix + self.M_bdf_rhs * dt * self.M_MbdA_matrix).T.dot(self.M_u0['velocity'][0]), 1) *
                         np.expand_dims(-self.M_bdf[1] * self.M_basis_time['velocity'][0], 0) +
                         np.expand_dims((self.M_MbdM_matrix + self.M_bdf_rhs * dt * self.M_MbdA_matrix).T.dot(self.M_u0['velocity'][1]), 1) *
                         np.expand_dims(-self.M_bdf[0] * self.M_basis_time['velocity'][0] - self.M_bdf[1] * self.M_basis_time['velocity'][1], 0) +
                         np.expand_dims(self.M_MbdM_matrix.T.dot(self.M_u0['velocity'][1]), 1) *
                         np.expand_dims(-self.M_bdf[0] * self.M_bdf[1] * self.M_basis_time['velocity'][0], 0) +
                         np.expand_dims(self.M_MbdM_matrix.dot(self.M_u0['velocity'][0]), 1) *
                         np.expand_dims(-self.M_bdf[1] * self.M_basis_time['velocity'][0], 0) +
                         np.expand_dims(self.M_MbdM_matrix.dot(self.M_u0['velocity'][1]), 1) *
                         np.expand_dims(-self.M_bdf[0] * self.M_basis_time['velocity'][0] - self.M_bdf[1] * self.M_basis_time['velocity'][1]
                                        - self.M_bdf[0] * self.M_bdf[1] * self.M_basis_time['velocity'][0], 0))
            self.M_f_Blocks_param_affine['structure'][0].append(fblock0_0)

            for k in range(K):
                fblock0_1k = (np.expand_dims(self.M_AbdM_matrices[k].dot(self.M_u0['velocity'][0]), 1) *
                              # np.expand_dims(-self.M_bdf_coeffs[1] * self.M_basis_time['velocity'][0], 0) +
                              np.expand_dims(-self.M_bdf[1] * self.M_basis_time['displacement'][0], 0) +
                              np.expand_dims(self.M_AbdM_matrices[k].dot(self.M_u0['velocity'][1]), 1) *
                              np.expand_dims(-self.M_bdf[0] * self.M_basis_time['displacement'][0] - self.M_bdf[1] * self.M_basis_time['displacement'][1], 0)
                              ) * (self.M_bdf_rhs * dt)
                self.M_f_Blocks_param_affine['structure'][0].append(fblock0_1k)

            fblock0_2 = (np.expand_dims(self.M_MbdMbd_matrix.dot(self.M_u0['velocity'][0]), 1) *
                         np.expand_dims(-self.M_bdf[1] * self.M_basis_time['velocity'][0], 0) +
                         np.expand_dims(self.M_MbdMbd_matrix.dot(self.M_u0['velocity'][1]), 1) *
                         np.expand_dims(-self.M_bdf[0] * self.M_basis_time['velocity'][0] - self.M_bdf[1] * self.M_basis_time['velocity'][1] -
                                        self.M_bdf[0] * self.M_bdf[1] * self.M_basis_time['velocity'][0], 0))
            self.M_f_Blocks_param_affine['structure'][0].append(fblock0_2)

            for k in range(K):
                fblock0_3k = (np.expand_dims(self.M_MbdAbd_matrices[k].T.dot(self.M_u0['velocity'][0]), 1) *
                              # np.expand_dims(-self.M_bdf_coeffs[1] * self.M_basis_time['velocity'][0], 0) +
                              np.expand_dims(-self.M_bdf[1] * self.M_basis_time['displacement'][0], 0) +
                              np.expand_dims(self.M_MbdAbd_matrices[k].T.dot(self.M_u0['velocity'][1]), 1) *
                              # np.expand_dims(-self.M_bdf_coeffs[0] * self.M_basis_time['velocity'][0] - self.M_bdf_coeffs[1] * self.M_basis_time['velocity'][1])
                              np.expand_dims(-self.M_bdf[0] * self.M_basis_time['displacement'][0] - self.M_bdf[1] * self.M_basis_time['displacement'][1], 0)
                              ) * (self.M_bdf_rhs * dt)
                self.M_f_Blocks_param_affine['structure'][0].append(fblock0_3k)

        self.M_f_Block = np.hstack([self._get_f_block(0), self._get_f_block(1), self._get_f_block(2)])

        # TODO: add terms related to the initial condition for the displacement

        return

    def build_rb_parametric_RHS(self, param):
        """
        MODIFY
        """

        rbmstpgrbNS.RbManagerSTPGRBNavierStokes.build_rb_parametric_RHS(self, param)

        return

    def assemble_reduced_structures_nlterm(self, x_hat, param=None):
        """
        Assemble the reduced structures needed to assemble the reduced non-linear term and its jacobian
        """

        rbmstpgrbNS.RbManagerSTPGRBNavierStokes.assemble_reduced_structures_nlterm(self, x_hat, param=param)

        u_hat = np.reshape(x_hat[:self.M_N['velocity']], (self.M_N_space['velocity'], self.M_N_time['velocity']))

        N_comp_nlterm = len(self.M_NLTerm_affine_components)
        if N_comp_nlterm == 0:
            logger.warning("No affine components for the non-linear convective term have been loaded!")
        else:
            self.__update_K2(u_hat[:N_comp_nlterm], param)

        return

    def reconstruct_fem_solution(self, _w, fields=None, indices_space=None, indices_time=None):
        """
        MODIFY
        """
        rbmstM.RbManagerSpaceTimeMembrane.reconstruct_fem_solution(self, _w, fields=fields,
                                                                   indices_space=indices_space,
                                                                   indices_time=indices_time)
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
