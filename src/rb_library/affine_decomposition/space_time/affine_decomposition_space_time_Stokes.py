#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 8 13:31:57 2019
@author: Riccardo Tenderini
@mail: riccardo.tenderini@epfl.ch
"""

import numpy as np
import os

import src.rb_library.affine_decomposition.affine_decomposition_unsteady as adu
import src.rb_library.affine_decomposition.space_time.affine_decomposition_space_time as adst

import src.utils.array_utils as arr_utils

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

# TODO: this needs to be updated; at the moment it is almost useless, as all FOM/RB components are directly assembled
#       within the RB manager


class AffineDecompositionStokes(adu.AffineDecompositionUnsteady):
    """MODIFY
    """

    def __init__(self, _qa=None, _qf=None, _qm=None, _qb=None):
        """MODIFY
        """

        super().__init__(_qa=_qa, _qf=_qf, _qm=_qm)
        self.M_qb = _qb if _qb is not None else 0
        return

    def get_Qb(self):
        """MODIFY
        """

        return self.M_qb

    def set_Q(self, _qa, _qf, _qm=1, _qb=1):
        """MODIFY
       """

        super().set_Q(_qa, _qf, _qm)
        self.M_qb = _qb

        return

    def print_ad_summary(self):
        """Method to print the main features of the AffineDecompositionUnsteady class instance
        """

        logger.info(f"\n------------- AD SUMMARY -------------\n"
                    f"Number of affine decomposition matrices A {self.M_qa}\n"
                    f"Number of affine decomposition vectors  f {self.M_qf}\n"
                    f"Number of affine decomposition matrices M {self.M_qm}\n"
                    f"Number of affine decomposition matrices B {self.M_qb}\n")

        return


class AffineDecompositionHandlerSpaceTimeStokes(adst.AffineDecompositionHandlerSpaceTime):
    """MODIFY
        """

    def __init__(self, _qa=None, _qf=None, _qm=None, _qb=None):
        """MODIFY
        """

        super().__init__(_qa=_qa, _qf=_qf, _qm=_qm)
        self.M_feAffineBq = []
        self.M_rbAffineBq = []
        self.M_feAffineBTq = []
        self.M_rbAffineBTq = []
        self.M_affineDecomposition = AffineDecompositionStokes(_qa=_qa, _qf=_qf, _qm=_qm, _qb=_qb)

        return

    def get_Qb(self):
        """Getter method, which returns the number of affine components for the divergence matrix

        :return: number of affine components for the divergence matrix
        :rtype: int
        """
        return self.M_affineDecomposition.get_Qb()

    def print_rb_affine_components(self):
        """MODIFY
        """

        super().print_rb_affine_components()

        Qf = self.qf
        Qb = self.get_Qb()

        for iQf in range(Qf):
            logger.info(f"RB rhs affine components {iQf}:\n {self.M_rbAffineFq[iQf]}")

        for iQ in range(Qb):
            logger.info(f"RB mat affine components {iQ}:\n {self.M_rbAffineBq}")

        return

    def check_set_fom_arrays(self):
        """MODIFY
        """
        return super().check_set_fom_arrays() and len(self.M_feAffineBq) == self.get_Qb()

    def check_set_fom_arrays_reduced(self):
        """MODIFY
        """
        return super().check_set_fom_arrays_reduced() and len(self.M_feAffineBq) == self.get_Qb()

    def import_fom_affine_arrays(self, _fom_problem, operators=None):
        """MODIFY
        """

        try:
            assert 'final_time' in _fom_problem.M_fom_specifics.keys()
        except AssertionError:
            logger.critical(
                f"The FomProblem must be unsteady to get the fom affine components for operators {operators}")
            raise TypeError

        if operators is None:
            operators = {'A', 'f', 'M', 'B', 'BT'}

        if 'A' in operators:
            super().import_fom_affine_arrays(_fom_problem, operators={'A'})

        if 'f' in operators:
            super().import_fom_affine_arrays(_fom_problem, operators={'f'})

        if 'M' in operators:
            super().import_fom_affine_arrays(_fom_problem, operators={'M'})

        if 'f_reduced' in operators:
            super().import_fom_affine_arrays(_fom_problem, operators={'f_reduced'})

        if 'f_space' in operators:
            super().import_fom_affine_arrays(_fom_problem, operators={'f_space'})

        if 'B' in operators:
            if len(self.M_feAffineBq) < self.get_Qb():
                logger.info("Importing affine arrays for operator B ")

                BB = _fom_problem.retrieve_fom_affine_components('B', self.get_Qb() - len(self.M_feAffineBq))
                starting_Qb = len(self.M_feAffineBq)

                for iQb in range(self.get_Qb() - starting_Qb):
                    self.M_feAffineBq.append(np.array(BB['B' + str(iQb)]))

        if 'BT' in operators:
            if len(self.M_feAffineBq) < self.get_Qb():
                logger.info("Importing affine arrays for operator BT ")

                BB = _fom_problem.retrieve_fom_affine_components('B', self.get_Qb() - len(self.M_feAffineBq))
                starting_Qb = len(self.M_feAffineBq)

                for iQb in range(self.get_Qb() - starting_Qb):
                    self.M_feAffineBq.append(np.array(BB['B' + str(iQb)]))

            self.M_feAffineBTq = [B.T for B in self.M_feAffineBq]

        return

    def get_affine_matrix_B(self, _q=0, _fom_problem=None):
        """MODIFY
        """

        if _fom_problem is not None and not self.check_set_fom_arrays():
            self.import_fom_affine_arrays(_fom_problem, operators={'B'})

        if _q < self.M_affineDecomposition.get_Qb():
            return self.M_feAffineBq[_q]
        else:
            raise IndexError

    def get_affine_matrix_BT(self, _q=0, _fom_problem=None):
        """MODIFY
        """

        if _fom_problem is not None and not self.check_set_fom_arrays():
            self.import_fom_affine_arrays(_fom_problem, operators={'BT'})

        if _q < self.M_affineDecomposition.get_Qb():
            return self.M_feAffineBTq[_q]
        else:
            raise IndexError

    def reset_rb_approximation(self):
        """MODIFY
        """

        super().reset_rb_approximation()
        self.M_rbAffineBq = []
        self.M_rbAffineBTq = []

        return

    def build_rb_affine_decompositions(self, _M_matrix, _MA_matrix, _B_matrix, _BT_matrix,
                                       _basis_time, _basis_time_lambda, _fom_problem,
                                       _sampling_vector, _all_locations_lambda, operators=None):
        """MODIFY
        """

        super().build_rb_affine_decompositions(_M_matrix, _MA_matrix, _basis_time, _fom_problem,
                                               _sampling_vector, operators)

        if operators is None:
            operators = {'Mat', 'BT', 'B', 'f'}

        nz = len(_sampling_vector)
        ns = _M_matrix.shape[1]
        nt = _basis_time.shape[1]
        nlambda = _B_matrix.shape[0]
        nt_lambda = _basis_time_lambda.shape[1]

        if 'BT' in operators:

            Qb = 1

            logger.info('Constructing RB B^T matrix \n')

            for iQb in range(Qb):
                self.M_rbAffineBTq.append(np.zeros((nz, nlambda * nt_lambda)))
                for count, indexes in enumerate(_sampling_vector):
                    for i in range(nlambda):
                        for j in range(nt_lambda):
                            self.M_rbAffineBTq[iQb][count, int(i * nt_lambda + j)] = _BT_matrix[indexes[0], i, iQb] * \
                                                                                     _basis_time_lambda[indexes[1], j]
        if 'B' in operators:

            Qb = 1

            logger.info('Constructing RB B matrix \n')
            for iQb in range(Qb):
                self.M_rbAffineBq.append(np.zeros((nlambda, ns * nt)))
                for count, indexes in enumerate(_all_locations_lambda):
                    for i in range(ns):
                        for j in range(nt):
                            self.M_rbAffineBq[iQb][count, int(i * nt + j)] = _B_matrix[indexes[0], i, iQb] * \
                                                                             _basis_time[indexes[1], j]

        return

    def get_rb_affine_BT(self, _q):
        """MODIFY
        """
        Qb = self.get_Qb()
        if _q < Qb:
            return self.M_rbAffineBTq[_q]
        else:
            raise IndexError

    def get_rb_affine_B(self, _q):
        """MODIFY
        """
        Qb = self.get_Qb()
        if _q < Qb:
            return self.M_rbAffineBq[_q]
        else:
            raise IndexError

    def import_rb_affine_components(self, _input_file, operators=None):
        """MODIFY
        """

        import_failures_set = super().import_rb_affine_components(_input_file, operators)

        if operators is None:
            operators = {'Mat', 'BT', 'B', 'f'}

        if 'BT' in operators:

            Qb = self.get_Qb()

            for iQ in range(Qb):
                try:
                    self.M_rbAffineBTq.append(np.loadtxt(_input_file + '_BT' + str(iQ) + '.txt'))
                except (IOError, OSError, FileNotFoundError) as e:
                    logger.error(f"Error {e}. Impossible to open the desired file for matrix BT{str(iQ)}.")
                    import_failures_set.add('BT')
                    break

        if 'B' in operators:

            Qb = self.get_Qb()

            for iQ in range(Qb):
                try:
                    self.M_rbAffineBq.append(np.loadtxt(_input_file + '_B' + str(iQ) + '.txt'))
                except (IOError, OSError, FileNotFoundError) as e:
                    logger.error(f"Error {e}. Impossible to open the desired file for matrix B{str(iQ)}.")
                    import_failures_set.add('B')
                    break

        return import_failures_set

    def set_affine_rb_matrix(self, _rbAffineMatq, _rbAffineBq, _rbAffineBTq):
        """MODIFY
        """

        super().set_affine_rb_matrix(_rbAffineMatq)
        self.M_rbAffineBq = _rbAffineBq
        self.M_rbAffineBTq = _rbAffineBTq

        return

    def save_rb_affine_decomposition(self, _file_name, operators=None):
        """MODIFY
        """

        super.save_rb_affine_decomposition(_file_name, operators)

        if _file_name is not None:
            if operators is None:
                operators = {'Mat', 'B', 'BT', 'f'}

            if 'B' in operators:

                Qb = self.get_Qb()
                for iQ in range(Qb):
                    path = _file_name + '_B' + str(iQ) + '.txt'
                    arr_utils.save_array(self.M_rbAffineBq[iQ], path)

            if 'BT' in operators:

                Qb = self.get_Qb()
                for iQ in range(Qb):
                    path = _file_name + '_BT' + str(iQ) + '.txt'
                    arr_utils.save_array(self.M_rbAffineBTq[iQ], path)

        return


__all__ = [
    "AffineDecompositionStokes",
    "AffineDecompositionHandlerSpaceTimeStokes"
]
