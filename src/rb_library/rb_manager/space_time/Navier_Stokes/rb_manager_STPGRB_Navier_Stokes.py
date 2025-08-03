#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 15:32:02 2022
@author: Riccardo Tenderini
@mail: riccardo.tenderini@epfl.ch
"""

import numpy as np
import os

import src.rb_library.rb_manager.space_time.Stokes.rb_manager_STPGRB_Stokes as rbmstpgrbS
import src.rb_library.rb_manager.space_time.Navier_Stokes.rb_manager_space_time_Navier_Stokes as rbmstNS

import src.utils.array_utils as arr_utils

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../../../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class RbManagerSTPGRBNavierStokes(rbmstNS.RbManagerSpaceTimeNavierStokes,
                                  rbmstpgrbS.RbManagerSTPGRBStokes):
    """MODIFY
    """

    def __init__(self, _fom_problem, _affine_decomposition=None):
        """ MODIFY
        """

        super().__init__(_fom_problem, _affine_decomposition=_affine_decomposition)

        self.M_NLterm_offline_time_tensor_uup = np.zeros(0)
        self.M_NLterm_offline_time_tensor_uul = [np.zeros(0)] * self.M_n_coupling
        self.M_NLterm_offline_time_tensor_uuuu = np.zeros(0)
        self.M_NLterm_offline_time_tensor_u1uu = np.zeros(0)
        self.M_NLterm_offline_time_tensor_u2uu = np.zeros(0)
        self.M_NLterm_offline_time_tensor_u1u1u = np.zeros(0)
        self.M_NLterm_offline_time_tensor_u2u2u = np.zeros(0)

        self.M_AJ_matrices = np.zeros(0)
        self.M_MJ_matrices = np.zeros(0)
        self.M_BdivTJ_matrices = np.zeros(0)
        self.M_BTJ_matrices = [np.zeros(0)] * self.M_n_coupling
        self.M_JJ_matrices = np.zeros(0)
        self.M_MclotJ_matrices = np.zeros(0)

        self.M_AK_vectors = np.zeros(0)
        self.M_MK_vectors = np.zeros(0)
        self.M_BdivTK_vectors = np.zeros(0)
        self.M_BTK_vectors = [np.zeros(0)] * self.M_n_coupling
        self.M_JK_vectors = np.zeros(0)
        self.M_MclotK_vectors = np.zeros(0)

        self.M_J1_Blocks = [np.zeros(0)] * 9
        self.M_J2_Blocks = [np.zeros(0)] * 9
        self.M_K1_Blocks = [np.zeros(0)] * 3
        self.M_K2_Blocks = [np.zeros(0)] * 3
        self.M_K3_Blocks = [np.zeros(0)] * 3
        self.M_K4_Blocks = [np.zeros(0)] * 3

        logger.warning("Resistance BCs are not supported in the current implementation of ST-PGRB!")

        return

    def import_NLTerm_affine_components(self, _tolerances,  N_components=None,
                                        _space_projection='standard'):
        """
        MODIFY
        """

        if not self.check_norm_matrices():
            self.get_norm_matrices()

        if len(self.M_NLTerm_affine_components) > 0:
            return

        logger.info("Importing non-linear term affine components")

        path = os.path.join(self.M_fom_structures_path,
                            os.path.normpath(f"NLterm/"
                                             f"POD_tol_{_tolerances['velocity-space']:.0e}_"
                                             f"{_tolerances['pressure-space']:.0e}/"
                                             f"{_space_projection}/Vector_PG"))

        Nmax = N_components if N_components is not None else self.M_N_space
        if not os.path.isdir(path) and Nmax > 0:
            raise ValueError(f"Invalid path! No affine components for the NL term available at {path} !")
        if Nmax > self.M_N_space['velocity']:
            logger.warning(f"Setting the number of affine components for the NL term to {self.M_N_space['velocity']}, since the "
                           f"prescribed number {Nmax} is bigger than the basis dimension {self.M_N_space['velocity']}")
            Nmax = self.M_N_space['velocity']

        count_i = 0
        while os.path.isfile(os.path.join(path, f"Vec_{count_i}_0.m")) and count_i < Nmax:
            self.M_NLTerm_affine_components.append([])
            count_j = 0
            while os.path.isfile(os.path.join(path, f"Vec_{count_i}_{count_j}.m")) and count_j < Nmax:
                self.M_NLTerm_affine_components[count_i].append(np.loadtxt(
                    os.path.join(path, f"Vec_{count_i}_{count_j}.m"), delimiter=None))
                count_j += 1
            count_i += 1

        self.M_N_components_NLTerm = count_i

        logger.info(f"Loaded {self.M_N_components_NLTerm**2} non-linear term affine components")

        return

    def import_NLJacobian_affine_components(self, _tolerances,  N_components=None,
                                            _space_projection='standard'):
        """
        MODIFY
        """

        assert 'use convective jacobian' in self.M_newton_specifics.keys()

        if len(self.M_NLJacobian_affine_components) > 0:
            return

        logger.info("Importing non-linear jacobian affine components")

        path = os.path.join(self.M_fom_structures_path,
                            os.path.normpath(f"NLterm/"
                                             f"POD_tol_{_tolerances['velocity-space']:.0e}_"
                                             f"{_tolerances['pressure-space']:.0e}/"
                                             f"{_space_projection}/Matrix_PG"))

        Nmax = N_components if N_components is not None else self.M_N_space
        Nmax *= int(self.M_newton_specifics['use convective jacobian'])
        if not os.path.isdir(path) and Nmax > 0:
            raise ValueError(f"Invalid path! No affine components for the NL term jacobian available at {path} !")
        if Nmax > self.M_N_space['velocity']:
            logger.warning(f"Setting the number of affine components for the NL term to {self.M_N_space['velocity']}, since the "
                           f"prescribed number {Nmax} is bigger than the basis dimension {self.M_N_space['velocity']}")
            Nmax = self.M_N_space['velocity']

        count = 0
        while os.path.isfile(os.path.join(path, f"Mat_{count}.m")) and count < Nmax:
            self.M_NLJacobian_affine_components.append(np.loadtxt(
                os.path.join(path, f"Mat_{count}.m"), delimiter=',')[:, :self.M_N_space['velocity']])
            count += 1

        self.M_N_components_NLJacobian = count

        logger.info(f"Loaded {self.M_N_components_NLJacobian} non-linear jacobian affine components")

        return

    def build_ST_basis(self, _tolerances, which=None):
        """MODIFY
        """

        rbmstpgrbS.RbManagerSTPGRBStokes.build_ST_basis(self, _tolerances, which=which)

        return

    def compute_NLterm_offline_quantities_time(self):
        """
        MODIFY
        """

        super().compute_NLterm_offline_quantities_time()

        self.M_NLterm_offline_time_tensor_uup = np.zeros((self.M_N_time['velocity'], self.M_N_time['velocity'], self.M_N_time['pressure']))
        for i in range(self.M_N_time['velocity']):
            for j in range(self.M_N_time['velocity']):
                for k in range(self.M_N_time['pressure']):
                    self.M_NLterm_offline_time_tensor_uup[i, j, k] = np.sum(self.M_basis_time['velocity'][:, i] *
                                                                            self.M_basis_time['velocity'][:, j] *
                                                                            self.M_basis_time['pressure'][:, k])

        self.M_NLterm_offline_time_tensor_uul = [np.zeros(0)] * self.M_n_coupling
        for n in range(self.M_n_coupling):
            self.M_NLterm_offline_time_tensor_uul[n] = np.zeros((self.M_N_time['velocity'], self.M_N_time['velocity'], self.M_N_time['lambda'][n]))
            for i in range(self.M_N_time['velocity']):
                for j in range(self.M_N_time['velocity']):
                    for k in range(self.M_N_time['lambda'][n]):
                        self.M_NLterm_offline_time_tensor_uul[n][i, j, k] = np.sum(self.M_basis_time['velocity'][:, i] *
                                                                                   self.M_basis_time['velocity'][:, j] *
                                                                                   self.M_basis_time['lambda'][n][:, k])

        self.M_NLterm_offline_time_tensor_uuuu = np.zeros((self.M_N_time['velocity'], self.M_N_time['velocity'], self.M_N_time['velocity'], self.M_N_time['velocity']))
        for i in range(self.M_N_time['velocity']):
            for j in range(self.M_N_time['velocity']):
                for k in range(self.M_N_time['velocity']):
                    for l in range(self.M_N_time['velocity']):
                        self.M_NLterm_offline_time_tensor_uuuu[i, j, k, l] = np.sum(self.M_basis_time['velocity'][:, i] *
                                                                                    self.M_basis_time['velocity'][:, j] *
                                                                                    self.M_basis_time['velocity'][:, k] *
                                                                                    self.M_basis_time['velocity'][:, l])

        # self.M_NLterm_offline_time_tensor_u1uu = np.zeros((self.M_N_time['velocity'], self.M_N_time['velocity'], self.M_N_time['velocity']))
        # for i in range(self.M_N_time['velocity']):
        #     for j in range(self.M_N_time['velocity']):
        #         for k in range(self.M_N_time['velocity']):
        #             self.M_NLterm_offline_time_tensor_u1uu[i, j, k] = np.sum(self.M_basis_time['velocity'][1:, i] *
        #                                                                      self.M_basis_time['velocity'][:-1, j] *
        #                                                                      self.M_basis_time['velocity'][:-1, k])
        #
        # self.M_NLterm_offline_time_tensor_u2uu = np.zeros((self.M_N_time['velocity'], self.M_N_time['velocity'], self.M_N_time['velocity']))
        # for i in range(self.M_N_time['velocity']):
        #     for j in range(self.M_N_time['velocity']):
        #         for k in range(self.M_N_time['velocity']):
        #             self.M_NLterm_offline_time_tensor_u2uu[i, j, k] = np.sum(self.M_basis_time['velocity'][2:, i] *
        #                                                                      self.M_basis_time['velocity'][:-2, j] *
        #                                                                      self.M_basis_time['velocity'][:-2, k])

        self.M_NLterm_offline_time_tensor_u1u1u = np.zeros((self.M_N_time['velocity'], self.M_N_time['velocity'], self.M_N_time['velocity']))
        for i in range(self.M_N_time['velocity']):
            for j in range(self.M_N_time['velocity']):
                for k in range(self.M_N_time['velocity']):
                    self.M_NLterm_offline_time_tensor_u1u1u[i, j, k] = np.sum(self.M_basis_time['velocity'][1:, i] *
                                                                              self.M_basis_time['velocity'][1:, j] *
                                                                              self.M_basis_time['velocity'][:-1, k])

        self.M_NLterm_offline_time_tensor_u2u2u = np.zeros((self.M_N_time['velocity'], self.M_N_time['velocity'], self.M_N_time['velocity']))
        for i in range(self.M_N_time['velocity']):
            for j in range(self.M_N_time['velocity']):
                for k in range(self.M_N_time['velocity']):
                    self.M_NLterm_offline_time_tensor_u2u2u[i, j, k] = np.sum(self.M_basis_time['velocity'][2:, i] *
                                                                              self.M_basis_time['velocity'][2:, j] *
                                                                              self.M_basis_time['velocity'][:-2, k])

        return

    def compute_NLterm_offline_quantities_space(self):
        """
        MODIFY
        """

        assert self.M_used_norm in {'P', 'l2'}

        if not self.check_norm_matrices():
            self.get_norm_matrices()
        self.set_projection_norm()

        structures = {'A', 'M', 'Bdiv', 'B', 'R'}
        if 'clot' in self.M_parametrizations:
            structures.add('Mclot')
        FEM_matrices = self.import_FEM_structures(structures=structures)

        if 'R' in FEM_matrices:
            raise ValueError("Resistance BCs are not supported in the current implementation of ST-PGRB!")

        logger.info("Projecting NL term FEM structures onto the reduced subspace in space; this may take some time...")

        M_M_red = arr_utils.sparse_matrix_matrix_mul(FEM_matrices['M'], self.M_basis_space['velocity'])
        M_A_red = arr_utils.sparse_matrix_matrix_mul(FEM_matrices['A'], self.M_basis_space['velocity'])
        M_BdivT_red = arr_utils.sparse_matrix_matrix_mul(FEM_matrices['BdivT'], self.M_basis_space['pressure'])
        M_BT_red = [arr_utils.sparse_matrix_matrix_mul(FEM_matrices['BT'][n], self.M_basis_space['lambda'][n])
                    for n in range(self.M_n_coupling)]
        if 'clot' in self.M_parametrizations:
            M_Mclot_red = [arr_utils.sparse_matrix_matrix_mul(Mclot, self.M_basis_space['velocity'])
                            for Mclot in FEM_matrices['Mclot']]
            n_clots = len(M_Mclot_red)

        # 1. ASSEMBLING MATRICES

        if self.M_newton_specifics['use convective jacobian']:

            M_J_red = np.array([self.M_NLJacobian_affine_components[j]
                                for j in range(self.M_N_components_NLJacobian)])
            XJ_red = np.zeros((self.M_N_components_NLJacobian, self.M_Nh['velocity'], self.M_Nh['velocity']))
            for j in range(self.M_N_components_NLJacobian):
                XJ_red[j] = self.inverse_norm_premultiplication(M_J_red[j],  'velocity')

            self.M_AJ_matrices = np.zeros((self.M_N_components_NLJacobian, self.M_N_space['velocity'], self.M_N_space['velocity']))
            self.M_MJ_matrices = np.zeros((self.M_N_components_NLJacobian, self.M_N_space['velocity'], self.M_N_space['velocity']))
            self.M_BdivTJ_matrices = np.zeros((self.M_N_components_NLJacobian, self.M_N_space['pressure'], self.M_N_space['velocity']))
            self.M_BTJ_matrices = [np.zeros((self.M_N_components_NLJacobian, self.M_N['lambda'][n], self.M_N_space['velocity']))
                                   for n in range(self.M_n_coupling)]
            self.M_JJ_matrices = np.zeros((self.M_N_components_NLJacobian, self.M_N_components_NLJacobian,
                                           self.M_N_space['velocity'], self.M_N_space['velocity']))
            if 'clot' in self.M_parametrizations:
                self.M_MclotJ_matrices = np.zeros((n_clots, self.M_N_components_NLJacobian,
                                                    self.M_N_space['velocity'], self.M_N_space['velocity']))

            for j in range(self.M_N_components_NLJacobian):
                self.M_AJ_matrices[j] = M_A_red.T.dot(XJ_red[j])
                self.M_MJ_matrices[j] = M_M_red.T.dot(XJ_red[j])
                self.M_BdivTJ_matrices[j] = M_BdivT_red.T.dot(XJ_red[j])

                for n in range(self.M_n_coupling):
                    self.M_BTJ_matrices[n][j] = M_BT_red[n].T.dot(XJ_red[j])

                for j2 in range(self.M_N_components_NLJacobian):
                    self.M_JJ_matrices[j, j2] = M_J_red[j].T.dot(XJ_red[j2])

                if 'clot' in self.M_parametrizations:
                    for k in range(n_clots):
                        self.M_MclotJ_matrices[k][j] = M_Mclot_red[k].T.dot(XJ_red[j])

        # 2. ASSEMBLING VECTORS
        M_K_red = np.array([[self.M_NLTerm_affine_components[i][j] for i in range(self.M_N_components_NLTerm)]
                            for j in range(self.M_N_components_NLTerm)])
        XK_red = np.zeros((self.M_N_components_NLTerm, self.M_N_components_NLTerm, self.M_Nh['velocity']))
        for i1 in range(self.M_N_components_NLTerm):
            for i2 in range(self.M_N_components_NLTerm):
                XK_red[i1, i2] = self.inverse_norm_premultiplication(M_K_red[i1, i2], 'velocity')

        self.M_AK_vectors = np.zeros((self.M_N_components_NLTerm, self.M_N_components_NLTerm, self.M_N_space['velocity']))
        self.M_MK_vectors = np.zeros((self.M_N_components_NLTerm, self.M_N_components_NLTerm, self.M_N_space['velocity']))
        self.M_BdivTK_vectors = np.zeros((self.M_N_components_NLTerm, self.M_N_components_NLTerm, self.M_N_space['pressure']))
        self.M_BTK_vectors = [np.zeros((self.M_N_components_NLTerm, self.M_N_components_NLTerm, self.M_N_space['lambda'][n]))
                              for n in range(self.M_n_coupling)]
        if self.M_newton_specifics['use convective jacobian']:
            M_J_red = np.array([self.M_NLJacobian_affine_components[j]
                                for j in range(self.M_N_components_NLJacobian)])
            self.M_JK_vectors = np.zeros((self.M_N_components_NLJacobian, self.M_N_components_NLTerm,
                                          self.M_N_components_NLTerm, self.M_N_space['velocity']))
        if 'clot' in self.M_parametrizations:
            self.M_MclotK_vectors = np.zeros((n_clots, self.M_N_components_NLTerm,
                                               self.M_N_components_NLTerm, self.M_N_space['velocity']))

        for i1 in range(self.M_N_components_NLTerm):

            for i2 in range(self.M_N_components_NLTerm):
                self.M_AK_vectors[i1, i2] = M_A_red.T.dot(XK_red[i1, i2])
                self.M_MK_vectors[i1, i2] = M_M_red.T.dot(XK_red[i1, i2])
                self.M_BdivTK_vectors[i1, i2] = M_BdivT_red.T.dot(XK_red[i1, i2])

                if self.M_newton_specifics['use convective jacobian']:
                    for j in range(self.M_N_components_NLJacobian):
                        self.M_JK_vectors[j, i1, i2] = M_J_red[j].T.dot(XK_red[i1, i2])

                for n in range(self.M_n_coupling):
                    self.M_BTK_vectors[n][i1, i2] = M_BT_red[n].T.dot(XK_red[i1, i2])

                if 'clot' in self.M_parametrizations:
                    for k in range(n_clots):
                        self.M_MclotK_vectors[k, i1, i2] = M_Mclot_red[k].T.dot(XK_red[i1, i2])

        # TODO: save to file
        # if self.M_save_offline_structures:
        #     logger.debug("Dumping space-reduced NL-term structures to file ...")

        return

    def import_NLterm_offline_quantities_space(self):
        """
        MODIFY
        """

        # TODO: import the NL-term reduced structures from file

        return

    def build_reduced_convective_term(self, x_hat):
        """
        MODIFY
        """
        N_comp = len(self.M_NLTerm_affine_components)
        if N_comp == 0:
            logger.warning("No affine components for the non-linear convective term have been loaded!")
            return 0

        K_Blocks = [np.zeros(0)] * 3
        K_Blocks[0] = self.M_K2_Blocks[0]
        K_Blocks[1] = self.M_K2_Blocks[1]
        K_Blocks[2] = self.M_K2_Blocks[2]

        if self.M_newton_specifics['use convective jacobian']:
            K_Blocks[0] += (self.M_K1_blocks[0] + self.M_K3_Blocks[0] + self.M_K4_Blocks[0])

        K = np.block([K_Blocks[0], K_Blocks[1], K_Blocks[2]])

        return K

    def __build_K1(self, x_hat):
        """
        Build the non-linear rhs term K1 = J1^T x_{st}, with J1 = Pi^T A^T (P_x)^(-1) J(u_{st}) Pi
        """

        assert self.M_newton_specifics['use convective jacobian']

        logger.debug("Building K1")

        u_hat, p_hat, l_hat = x_hat[:self.M_N['velocity']], x_hat[self.M_N['velocity']:self.M_N['velocity']+self.M_N['pressure']], x_hat[self.M_N['velocity']+self.M_N['pressure']:]

        self.M_K1_blocks = [np.zeros(0)] * 3
        self.M_K1_blocks[0] = (self.M_J1_Blocks[0].T.dot(u_hat) +
                               self.M_J1_Blocks[3].T.dot(p_hat) +
                               self.M_J1_Blocks[6].T.dot(l_hat))

        return

    def __build_K2(self, u_hat, param):
        """
        Build the non-linear rhs term K2 = Pi^T A^T (P_x)^(-1) c(u_{st})
        """

        logger.debug("Building K2")

        dt = self.dt
        n_clots = self.get_clots_number()

        u_hat2 = np.expand_dims(np.swapaxes(np.multiply.outer(u_hat, u_hat), 1, 2), 4)  # (n_c, n_c, n_t, n_t, 1)

        param_map = self.differentiate_parameters(param)

        time_contrib_uuu = np.expand_dims(np.array(self.M_NLterm_offline_time_tensor_uuu), (0, 1))  # (1, 1, n_t, n_t, n_t)
        time_contrib_u1u1u = np.expand_dims(np.array(self.M_NLterm_offline_time_tensor_u1u1u), (0, 1))  # (1, 1, n_t, n_t, n_t)
        time_contrib_u2u2u = np.expand_dims(np.array(self.M_NLterm_offline_time_tensor_u2u2u), (0, 1))  # (1, 1, n_t, n_t, n_t)
        time_contrib_uup = np.expand_dims(np.array(self.M_NLterm_offline_time_tensor_uup), (0, 1))  # (1, 1, n_t, n_t, n_tp)
        time_contrib_uul = [np.zeros(0)] * self.M_n_coupling
        for n in range(self.M_n_coupling):
            time_contrib_uul[n] = np.expand_dims(np.array(self.M_NLterm_offline_time_tensor_uul[n]), (0, 1))  # (1, 1, n_t, n_t, n_tl)

        AK = np.expand_dims(np.array(self.M_AK_vectors), 3)  # (n_c, n_c, n_s, 1)
        MK = np.expand_dims(np.array(self.M_MK_vectors), 3)  # (n_c, n_c, n_s, 1)
        BdivTK = np.expand_dims(np.array(self.M_BdivTK_vectors), 3)  # (n_c, n_c, n_sp, 1)
        BTK = [np.zeros(0)] * self.M_n_coupling
        for n in range(self.M_n_coupling):
            BTK[n] = np.expand_dims(np.array(self.M_BTK_vectors[n]), 3)  # (n_c, n_c, n_sl, 1)
        if 'clot' in self.M_parametrizations:
            MclotK = [np.zeros(0)] * n_clots
            for k in range(n_clots):
                MclotK[k] = np.expand_dims(np.array(self.M_MclotK_vectors[k]), 3)  # (n_c, n_c, n_s, 1)

        tmp_uuu = np.expand_dims(np.sum(u_hat2 * time_contrib_uuu, axis=(2, 3)), 2)  # (n_c, n_c, 1, n_t)
        tmp_u1u1u = np.expand_dims(np.sum(u_hat2 * time_contrib_u1u1u, axis=(2, 3)), 2)  # (n_c, n_c, 1, n_t)
        tmp_u2u2u = np.expand_dims(np.sum(u_hat2 * time_contrib_u2u2u, axis=(2, 3)), 2)  # (n_c, n_c, 1, n_t)
        tmp_uup = np.expand_dims(np.sum(u_hat2 * time_contrib_uup, axis=(2, 3)), 2)  # (n_c, n_c, 1, n_tp)
        tmp_uul = [np.zeros(0)] * self.M_n_coupling
        for n in range(self.M_n_coupling):
            tmp_uul[n] = np.expand_dims(np.sum(u_hat2 * time_contrib_uul[n], axis=(2, 3)), 2)  # (n_c, n_c, 1, n_tl)

        AK_uuu = np.sum(AK * tmp_uuu, axis=(0, 1)).flatten()
        MK_uuu = np.sum(MK * tmp_uuu, axis=(0, 1)).flatten()
        MK_u1u1u = np.sum(MK * tmp_u1u1u, axis=(0, 1)).flatten()
        MK_u2u2u = np.sum(MK * tmp_u2u2u, axis=(0, 1)).flatten()
        BdivTK_uup = np.sum(BdivTK * tmp_uup, axis=(0, 1)).flatten()
        BTK_uul = [np.zeros(0)] * self.M_n_coupling
        for n in range(self.M_n_coupling):
            BTK_uul[n] = np.sum(BTK[n] * tmp_uul[n], axis=(0, 1)).flatten()
        if 'clot' in self.M_parametrizations:
            MclotK_uuu = [np.zeros(0)] * n_clots
            for k in range(n_clots):
                MclotK_uuu[k] = np.sum(MclotK[k] * tmp_uuu, axis=(0, 1)).flatten()

        self.M_K2_Blocks = [np.zeros(0)] * 3
        self.M_K2_Blocks[0] = MK_uuu + self.M_bdf_rhs * dt * AK_uuu + self.M_bdf[0] * MK_u1u1u + self.M_bdf[1] * MK_u2u2u
        self.M_K2_Blocks[1] = self.M_bdf_rhs * dt * BdivTK_uup
        self.M_K2_Blocks[2] = np.zeros(self.M_N_lambda_cumulative[-1])
        for n in range(self.M_n_coupling):
            self.M_K2_Blocks[2][self.M_N_lambda_cumulative[n]:self.M_N_lambda_cumulative[n+1]] = self.M_bdf_rhs * dt * BTK_uul[n]

        if 'clot' in self.M_parametrizations:
            self.M_K2_Blocks[0] += np.sum(_fun(param_map['clot']) * _vector * self.M_bdf_rhs * dt
                                          for _vector, _fun in zip(MclotK_uuu,
                                                                   self.M_Blocks_param_affine_fun['clot'][0]))

        self.M_K2_Blocks = [elem * (self.M_bdf_rhs * dt) for elem in self.M_K2_Blocks]

        return

    def __build_K3(self, u_hat):
        """
        Build the non-linear rhs term K3 = Pi^T J^T(u_{st}) (P_X)^(-1) c(u_{st})
        """

        assert self.M_newton_specifics['use convective jacobian']

        logger.debug("Building K3")

        dt = self.dt

        u_hat2 = np.multiply.outer(u_hat, u_hat)  # (n_c, n_t, n_c, n_t)
        u_hat3 = np.multiply.outer(u_hat2, u_hat)  # (n_c, n_t, n_c, n_t, n_c, n_t)
        u_hat3 = np.expand_dims(np.transpose(u_hat3, (0, 2, 4, 1, 3, 5)), 6)  # (n_c, n_c, n_c, n_t, n_t, n_t, 1)

        time_contrib_uuuu = np.expand_dims(np.array(self.M_NLterm_offline_time_tensor_uuuu), (0, 1, 2))  # (1, 1, 1, n_t, n_t, n_t, n_t)

        JK = np.expand_dims(np.array(self.M_JK_vectors), 4)  # (n_c, n_c, n_c, n_s, 1)

        tmp_uuuu = np.expand_dims(np.sum(u_hat3 * time_contrib_uuuu, axis=(3, 4, 5)), 3)  # (n_c, n_c, n_c, 1, n_t)

        JK_uuuu = np.sum(JK * tmp_uuuu, axis=(0, 1, 2)).flatten()

        self.M_K3_Blocks = [np.zeros(0)] * 3
        self.M_K3_Blocks[0] = - self.M_bdf[0] * self.M_bdf[1] * dt ** 2 * JK_uuuu

        return

    def __build_K4(self, u_hat):
        """
        Build the non-linear term K4 = Pi^T J^T(u_{st}) (P_X)^(-1) F^{st}  (non-zero only if the IC is non-zero)
        """

        assert self.M_newton_specifics['use convective jacobian']

        logger.debug("Building K4")

        dt = self.dt

        self.M_K4_Blocks = [np.zeros(0)] * 3

        if self._has_IC():
            JM = np.swapaxes(np.array(self.M_MJ_matrices), 1, 2)  # (n_c, n_s, n_s)
            u0 = np.expand_dims(self.M_u0['velocity'].T, 0)  # (1, ns, 2)
            JMu0 = np.dot(JM, u0)  # (nc, ns, 1, 2)

            time_contrib_u1 = np.expand_dims(np.outer(self.M_basis_time['velocity'][0], self.M_basis_time['velocity'][0]), 0)  # (1, nt_u, nt_u)
            tmp_u1 = np.expand_dims(np.sum(np.expand_dims(u_hat, (1, 2)) * time_contrib_u1, 2), 1)  # (nc, 1, nt_u)
            time_contrib_u2 = np.expand_dims(np.outer(self.M_basis_time['velocity'][1], self.M_basis_time['velocity'][1]), 0)  # (1, nt_u, nt_u)
            tmp_u2 = np.expand_dims(np.sum(np.expand_dims(u_hat, (1, 2)) * time_contrib_u2, 2), 1)  # (nc, 1, nt_u)

            K4_1 = np.sum(JMu0[..., 0] * tmp_u1, 0).flatten()
            K4_1 *= -self.M_bdf[1] * self.M_bdf_rhs * dt

            K4_2 = np.sum(JMu0[..., 1] * tmp_u1, 0).flatten()
            K4_2 *= -self.M_bdf[0] * self.M_bdf_rhs * dt

            K4_3 = np.sum(JMu0[..., 1] * tmp_u2, 0).flatten()
            K4_3 *= -self.M_bdf[1] * self.M_bdf_rhs * dt

            self.M_K4_Blocks[0] = K4_1 + K4_2 + K4_3
        else:
            self.M_K4_Blocks[0] = np.zeros(self.M_N['velocity'])

        return

    def build_reduced_convective_jacobian(self, x_hat):
        """
        MODIFY
        """

        assert self.M_newton_specifics['use convective jacobian']

        N_comp = len(self.M_NLJacobian_affine_components)
        if N_comp == 0:
            logger.warning("No affine components for the non-linear jacobian matrix have been loaded!")
            return 0

        J_Blocks = [np.zeros(0)] * 9
        J_Blocks[0] = self.M_J1_Blocks[0] + self.M_J1_Blocks[0].T + self.M_J2_Blocks[0]
        J_Blocks[3] = self.M_J1_Blocks[3]
        J_Blocks[6] = self.M_J1_Blocks[6]
        J_Blocks[1] = self.M_J1_Blocks[3].T
        J_Blocks[2] = self.M_J1_Blocks[6].T

        J_Blocks[4] = np.zeros((self.M_N['pressure'], self.M_N['pressure']))
        J_Blocks[5] = np.zeros((self.M_N['pressure'], self.M_N_lambda_cumulative[-1]))
        J_Blocks[7] = np.zeros((self.M_N_lambda_cumulative[-1], self.M_N['pressure']))
        J_Blocks[8] = np.zeros((self.M_N_lambda_cumulative[-1], self.M_N_lambda_cumulative[-1]))

        J = np.block([[J_Blocks[0], J_Blocks[1], J_Blocks[2]],
                      [J_Blocks[3], J_Blocks[4], J_Blocks[5]],
                      [J_Blocks[6], J_Blocks[7], J_Blocks[8]]])

        return J

    def __build_J1(self, u_hat, param):
        """
        Build the non-linear jacobian term Pi^T A^T (P_x)^(-1) J Pi
        """

        assert self.M_newton_specifics['use convective jacobian']

        logger.debug("Building J1")

        dt = self.dt
        n_clots = self.get_clots_number()

        u_hat = np.expand_dims(u_hat, (2, 3))  # (n_c, n_t, 1, 1)

        param_map = self.differentiate_parameters(param)

        time_contrib_uuu = np.expand_dims(np.array(self.M_NLterm_offline_time_tensor_uuu), 0)  # (1, n_t, n_t, n_t)
        time_contrib_u1uu1 = np.expand_dims(np.swapaxes(np.array(self.M_NLterm_offline_time_tensor_u1u1u), 1, 2), 0)  # (1, n_t, n_t, n_t)
        time_contrib_u2uu2 = np.expand_dims(np.swapaxes(np.array(self.M_NLterm_offline_time_tensor_u2u2u), 1, 2), 0)  # (1, n_t, n_t, n_t)
        time_contrib_uup = np.expand_dims(np.array(self.M_NLterm_offline_time_tensor_uup), 0)  # (1, n_t, n_t, n_tp)
        time_contrib_uul = [np.zeros(0)] * self.M_n_coupling
        for n in range(self.M_n_coupling):
            time_contrib_uul[n] = np.expand_dims(np.array(self.M_NLterm_offline_time_tensor_uul[n]), 0)  # (1, n_t, n_t, n_tl)

        AJ = np.expand_dims(np.array(self.M_AJ_matrices), (3, 4))  # (n_c, n_s, n_s, 1, 1)
        MJ = np.expand_dims(np.array(self.M_MJ_matrices), (3, 4))  # (n_c, n_s, n_s, 1, 1)
        BdivTJ = np.expand_dims(np.array(self.M_BdivTJ_matrices), (3, 4))  # (n_c, n_sp, n_s, 1, 1)
        BTJ = [np.zeros(0)] * self.M_n_coupling
        for n in range(self.M_n_coupling):
            BTJ[n] = np.expand_dims(np.array(self.M_BTJ_matrices[n]), (3, 4))  # (n_c, n_sl, n_s, 1, 1)
        if 'clot' in self.M_parametrizations:
            MclotJ = [np.zeros(0)] * n_clots
            for k in range(n_clots):
                MclotJ[k] = np.expand_dims(np.array(self.M_MclotJ_matrices[k]), (3, 4))  # (n_c, n_s, n_s, 1)

        tmp_uuu = np.expand_dims(np.sum(u_hat * time_contrib_uuu, axis=1), (1, 2))  # (n_c, 1, 1, n_t, n_t)
        tmp_u1uu1 = np.expand_dims(np.sum(u_hat * time_contrib_u1uu1, axis=1), (1, 2))  # (n_c, 1, 1, n_t, n_t)
        tmp_u2uu2 = np.expand_dims(np.sum(u_hat * time_contrib_u2uu2, axis=1), (1, 2))  # (n_c, 1, 1, n_t, n_t)
        tmp_uup = np.expand_dims(np.sum(u_hat * time_contrib_uup, axis=1), (1, 2))  # (n_c, 1, 1, n_t, n_tp)
        tmp_uul = [np.zeros(0)] * self.M_n_coupling
        for n in range(self.M_n_coupling):
            tmp_uul[n] = np.expand_dims(np.sum(u_hat * time_contrib_uul[n], axis=1), (1, 2))  # (n_c, 1, 1, n_t, n_tl)

        AJ_uuu = np.swapaxes(np.sum(AJ * tmp_uuu, axis=0), 1, 2).reshape((self.M_N['velocity'], self.M_N['velocity']))
        MJ_uuu = np.swapaxes(np.sum(MJ * tmp_uuu, axis=0), 1, 2).reshape((self.M_N['velocity'], self.M_N['velocity']))
        MJ_u1uu1 = np.swapaxes(np.sum(MJ * tmp_u1uu1, axis=0), 1, 2).reshape((self.M_N['velocity'], self.M_N['velocity']))
        MJ_u2uu2 = np.swapaxes(np.sum(MJ * tmp_u2uu2, axis=0), 1, 2).reshape((self.M_N['velocity'], self.M_N['velocity']))
        BdivTJ_uup = np.transpose(np.sum(BdivTJ * tmp_uup, axis=0),
                                  (0, 3, 1, 2)).reshape((self.M_N['pressure'], self.M_N['velocity']))
        BTJ_uul = [np.zeros(0)] * self.M_n_coupling
        for n in range(self.M_n_coupling):
            BTJ_uul[n] = np.transpose(np.sum(BTJ[n] * tmp_uul[n], axis=0),
                                      (0, 3, 1, 2)).reshape((self.M_N['lambda'][n], self.M_N['velocity']))
        if 'clot' in self.M_parametrizations:
            MclotJ_uuu = [np.zeros(0)] * n_clots
            for k in range(n_clots):
                MclotJ_uuu[k] = np.swapaxes(np.sum(MclotJ[k] * tmp_uuu, axis=0), 1, 2).reshape((self.M_N['velocity'], self.M_N['velocity']))

        self.M_J1_Blocks = [np.zeros(0)] * 9
        self.M_J1_Blocks[0] = (MJ_uuu + self.M_bdf_rhs * dt * AJ_uuu + self.M_bdf[0] * MJ_u1uu1 + self.M_bdf[1] * MJ_u2uu2)
        self.M_J1_Blocks[3] = self.M_bdf_rhs * dt * BdivTJ_uup
        self.M_J1_Blocks[6] = np.zeros((self.M_N_lambda_cumulative[-1], self.M_N['velocity']))
        for n in range(self.M_n_coupling):
            self.M_J1_Blocks[6][self.M_N_lambda_cumulative[n]:self.M_N_lambda_cumulative[n+1], :] = self.M_bdf_rhs * dt * BTJ_uul[n]

        if 'clot' in self.M_parametrizations:
            self.M_J1_Blocks[0] += np.sum(_fun(param_map['clot']) * _matrix * self.M_bdf_rhs * dt
                                          for _matrix, _fun in zip(MclotJ_uuu,
                                                                   self.M_Blocks_param_affine_fun['clot'][0]))

        self.M_J1_Blocks = [elem * (self.M_bdf_rhs * dt) for elem in self.M_J1_Blocks]

        return

    def __build_J2(self, u_hat):
        """
        Build the non-linear jacobian term Pi^T J^T (P_x)^(-1) J Pi
        """

        assert self.M_newton_specifics['use convective jacobian']

        logger.debug("Building J2")

        dt = self.dt

        u_hat2 = np.expand_dims(np.swapaxes(np.multiply.outer(u_hat, u_hat), 1, 2), (4, 5))  # (n_c, n_c, n_t, n_t, 1, 1)

        time_contrib_uuuu = np.expand_dims(np.array(self.M_NLterm_offline_time_tensor_uuuu), (0, 1))  # (1, 1, n_t, n_t, n_t, n_t)

        JJ = np.expand_dims(np.array(self.M_JJ_matrices), (4, 5))  # (n_c, n_c, n_s, n_s, 1, 1)

        tmp_uuuu = np.expand_dims(np.sum(u_hat2 * time_contrib_uuuu, axis=(2, 3)), (2, 3))  # (n_c, n_c, 1, 1, n_t, n_t)

        JJ_uuuu = np.swapaxes(np.sum(JJ * tmp_uuuu, axis=(0, 1)), 1, 2).reshape(self.M_N['velocity'], self.M_N['velocity'])

        self.M_J2_Blocks = [np.zeros(0)] * 9
        self.M_J2_Blocks[0] = - self.M_bdf[0] * self.M_bdf[1] * dt ** 2 * JJ_uuuu

        return

    def assemble_reduced_structures(self, _space_projection='standard', matrices=None,
                                    _tolerances=None, N_components=None):
        """
        MODIFY
        """

        assert _tolerances is not None

        rbmstpgrbS.RbManagerSTPGRBStokes.assemble_reduced_structures(self, _space_projection=_space_projection,
                                                                     matrices=matrices)

        self.import_reduced_convective_structures(_tolerances,
                                                  N_components=N_components, _space_projection=_space_projection)

        return

    def import_reduced_structures(self, _space_projection='standard', _tolerances=None, N_components=None):
        """
        MODIFY
        """

        import_success = rbmstpgrbS.RbManagerSTPGRBStokes.import_reduced_structures(self)

        self.reset_reduced_structures_nlterm()

        self.import_reduced_convective_structures(_tolerances,
                                                  N_components=N_components, _space_projection=_space_projection)

        return import_success

    def save_reduced_structures(self):
        """
        MODIFY
        """

        rbmstpgrbS.RbManagerSTPGRBStokes.save_reduced_structures(self)
        return

    def assemble_reduced_structures_nlterm(self, x_hat, param=None):
        """
        Assemble the reduced structures needed to assemble the reduced non-linear term and its jacobian
        """

        u_hat = np.reshape(x_hat[:self.M_N['velocity']], (self.M_N_space['velocity'], self.M_N_time['velocity']))

        if self.M_newton_specifics['use convective jacobian']:
            N_comp_nljac = len(self.M_NLJacobian_affine_components)
            if N_comp_nljac == 0:
                logger.warning("No affine components for the non-linear convective jacobian have been loaded!")
            else:
                self.__build_J1(u_hat[:N_comp_nljac], param)
                self.__build_J2(u_hat[:N_comp_nljac])

        N_comp_nlterm = len(self.M_NLTerm_affine_components)
        if N_comp_nlterm == 0:
            logger.warning("No affine components for the non-linear convective term have been loaded!")
        else:
            self.__build_K2(u_hat[:N_comp_nlterm], param)
            if self.M_newton_specifics['use convective jacobian']:
                self.__build_K1(x_hat)
                self.__build_K3(u_hat[:N_comp_nlterm])
                self.__build_K4(u_hat[:N_comp_nlterm])

        return

    def reset_reduced_structures_nlterm(self):
        """
        Reset the reduced structures needed to assemble the reduced non-linear term and its jacobian
        """

        self.M_J1_Blocks = [np.zeros(0)] * 9
        self.M_J2_Blocks = [np.zeros(0)] * 9
        self.M_K1_Blocks = [np.zeros(0)] * 3
        self.M_K2_Blocks = [np.zeros(0)] * 3
        self.M_K3_Blocks = [np.zeros(0)] * 3
        self.M_K4_Blocks = [np.zeros(0)] * 3

        return

    def reset_reduced_structures(self):
        """
        MODIFY
        """

        rbmstNS.RbManagerSpaceTimeNavierStokes.reset_reduced_structures(self)

        self.reset_reduced_structures_nlterm()

        return

    def build_rb_affine_decompositions(self, operators=None):
        """
        MODIFY
        """

        rbmstpgrbS.RbManagerSTPGRBStokes.build_rb_affine_decompositions(self, operators=operators)

        return

    def build_rb_parametric_RHS(self, param):
        """
        MODIFY
        """

        return rbmstpgrbS.RbManagerSTPGRBStokes.build_rb_parametric_RHS(self, param)

    def build_rb_approximation(self, _ns, _n_weak_io, _mesh_name, _tolerances,
                               _space_projection='standard', prob=None, ss_ratio=1, _used_norm='l2', _N_components=None):
        """
        MODIFY
        """

        self.M_used_norm = _used_norm

        rbmstNS.RbManagerSpaceTimeNavierStokes.build_rb_approximation(self, _ns, _n_weak_io, _mesh_name, _tolerances,
                                                                      _space_projection=_space_projection,
                                                                      prob=prob, ss_ratio=ss_ratio,
                                                                      _N_components=_N_components)

        return

    def check_dataset(self, _nsnap):
        """
        MODIFY
        """

        rbmstNS.RbManagerSpaceTimeNavierStokes.check_dataset(self, _nsnap)

        return
