#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 14:14:56 2021
@author: Riccardo Tenderini
@mail: riccardo.tenderini@epfl.ch
"""

import numpy as np
import os
import time

import src.rb_library.affine_decomposition.space_time.affine_decomposition_space_time as adst
import src.utils.array_utils as arr_utils


import logging.config
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class AffineDecompositionHandlerSTRB(adst.AffineDecompositionHandlerSpaceTime):
    """Class which handles the affine decomposition of the unsteady FOM problem at hand via ST-RB approach,
    interfacing with the FomProblemUnsteady class and the AffineDecompositionUnsteady class.
    It inherits from :class:`~affine_decomposition_unsteady.AffineDecompositionHandlerSpaceTime`
    """

    def __init__(self, _qa=None, _qf=None, _qm=None):
        """Initialization of the AffineDecompositionHandlerSTRB class

        :param _qa: number of affine components for the stiffness matrix, if not None. If None, the corresponding
           class attribute is set to 0
        :type _qa: int or NoneType
        :param _qf: number of affine components for the right-hand side vector, if not None. If None, the corresponding
           class attribute is set to 0
        :type _qf: int or NoneType
        :param _qm: number of affine components for the mass matrix if not None. If None, the corresponding class
           attribute is set to 0
        :type _qm: int or NoneType
        """

        super().__init__(_qa=_qa, _qf=_qf, _qm=_qm)

        return

    def build_rb_affine_decompositions(self, _basis_space, _basis_time, _fom_problem,
                                       operators=None):
        """Method which constructs the RB affine components for the operators passed in input. If no operator is passed,
        the RB affine arrays are constructed for the left-hand side matrix and the right-hand side vector,
        which arise after the application of the ST-RB method for the resolution of the fom problem at hand. If the
        FOM affine arrays are not yet stored in the class, they are constructed via a call to
        :func:`~affine_decomposition_unsteady.AffineDecompositionUnsteady.import_fom_affine_arrays`.
        If '_fom_problem' is not an instance of :class:`~fom_problem_unsteady.FomProblemUnsteady`, the method raises a
        TypeError.

        :param _basis_space: matrix encoding the Reduced Basis in space
        :type _basis_space: numpy.ndarray
        :param _basis_time: matrix encoding the Reduced Basis in time
        :type _basis_time: numpy.ndarray
        :param _fom_problem: unsteady parametrized FOM problem at hand, used to check that the time-related FOM
           specifics have been correctly set to import the FOM affine components, if not already stored in the class
        :type _fom_problem: FomProblemUnsteady
        :param operators: operators for which the affine components have to be imported. Admissible values are
          'Mat' for the left-hand side matrix and 'f' for the right-hand side vector arising after the application of
          the ST-RB method to solve the unsteady FOM problem at hand. Defaults to None.
        :type operators: set or NoneType
        """

        assert 'final_time' in _fom_problem.M_fom_specifics.keys(), \
            f"The FomProblem must be unsteady to get the fom affine components for the operators {operators}"

        if operators is None:
            operators = {'Mat', 'f', 'u0'}

        super().build_rb_affine_decompositions(_basis_space, _fom_problem,
                                               operators={'M', 'A', 'f'})

        Ns = _basis_space.shape[0]
        Nt = _basis_time.shape[0]
        ns = _basis_space.shape[1]
        nt = _basis_time.shape[1]

        if not self.check_set_fom_arrays():
            self.import_fom_affine_arrays(_fom_problem)
        else:
            logger.debug("Already set the RB affine arrays")

        if 'Mat' in operators:
            Qa = self.qa
            Qm = self.qm
            logger.info('Constructing RB matrices \n')
            for iQa in range(Qa):
                self.M_rbstAffineMatq.append(np.zeros((ns * nt, ns * nt)))
                for it in range(nt):
                    self.M_rbstAffineMatq[iQa][it::nt, it::nt] = self.M_rbAffineAq[iQa]

            for iQm in range(Qm):
                self.M_rbstAffineMatq.append(np.zeros((ns * nt, ns * nt)))
                for it in range(nt):
                    self.M_rbstAffineMatq[Qa + iQm][it::nt, it::nt] = self.M_rbAffineMq[iQm]

            for iQm in range(Qm):
                self.M_rbstAffineMatq.append(np.zeros((ns * nt, ns * nt)))
                for it in range(nt):
                    for jt in range(nt):
                        Kt = np.dot(_basis_time[1:, it], _basis_time[:-1, jt])
                        self.M_rbstAffineMatq[Qa + Qm + iQm][it::nt, jt::nt] = -self.M_rbAffineMq[iQm] * Kt

        if 'f' in operators:
            logger.info('Constructing RB rhs vector \n')
            Qf = self.qf
            for iQf in range(Qf):
                self.M_rbstAffineFq.append(np.zeros(ns * nt))
                for iS in range(ns):
                    for iT in range(nt):
                        current_index = int(iS * nt + iT)
                        self.M_rbstAffineFq[iQf][current_index] = np.sum(
                            np.outer(_basis_space[:, iS], _basis_time[:, iT]) *
                            np.reshape(self.M_feAffineFq[iQf][Ns:], (Ns, Nt), order='F'))

        if 'u0' in operators:
            self.build_rb_initial_condition(_basis_space, _basis_time, _fom_problem)

        return

    def build_rb_initial_condition(self, _basis_space, _basis_time, _fom_problem):
        """Method that allows to project the initial condition on the reduced subspace

        :param _basis_space: matrix encoding the reduced basis in space
        :type _basis_space: numpy.ndarray
        :param _basis_time: matrix encoding the reduced basis in time
        :type _basis_time: numpy.ndarray
        :param _fom_problem: unsteady FOM problem at hand
        :type _fom_problem: FomProblemUnsteady
        """

        assert 'final_time' in _fom_problem.M_fom_specifics.keys(), \
            "The FomProblem must be unsteady to get the reduced initial condition"

        Ns = _basis_space.shape[0]
        Nt = _basis_time.shape[0]

        IC = np.zeros((Ns, Nt))
        M = _fom_problem.retrieve_fom_affine_components('M', 1)['M0']
        u0 = _fom_problem.get_initial_condition(np.zeros(3))
        IC[:, 0] = np.squeeze(arr_utils.sparse_matrix_vector_mul(M, u0))
        self.M_rbInitialCondition = self.compute_generalized_coordinates(IC,
                                                                         _basis_space=_basis_space,
                                                                         _basis_time=_basis_time)

        return

    def build_rb_nonlinear_jacobian(self, u, _basis_space, _basis_time, _fom_problem, recompute_every=1):
        """Method that builds the jacobian matrix associated to the reduced non-linear term based on the current value
        of the solution (passed as first argument to the function). If the problem is linear, it simply returns an empty
        array of the right dimension.

        :param u: current reduced solution
        :type u: numpy.ndarray
        :param _basis_space: matrix encoding the reduced basis in space
        :type _basis_space: numpy.ndarray
        :param _basis_time: matrix encoding the reduced basis in time
        :type _basis_time: numpy.ndarray
        :param _fom_problem: unsteady FOM problem at hand
        :type _fom_problem: FomProblemUnsteady
        :param recompute_every: for unsteady problems, number of time instants after which the Jacobian is recomputed.
           It defaults to 1.
        :type recompute_every: int
        :return reduced non-linear jacobian (matrix of zeros if the problem is linear)
        :rtype: numpy.ndarray
        """

        start = time.time()

        if _fom_problem.is_linear():
            return np.zeros((_basis_space.shape[1], _basis_space.shape[1]))

        else:

            rb_nl_jac = np.array(super().build_rb_nonlinear_jacobian(u, _basis_space, _basis_time, _fom_problem,
                                                                     recompute_every=recompute_every))
            logger.debug(f"Elapsed time RB non-linear jacobian assembling: {(time.time() - start):.6f}")

            ns = _basis_space.shape[1]
            nt = _basis_time.shape[1]
            nst = ns * nt

            rbst_nl_jac = np.zeros((nst, nst))
            for iR in range(nst):
                iRs, iRt = int(iR / nt), iR % nt
                for iC in range(nst):
                    iCs, iCt = int(iC / nt), iC % nt
                    rbst_nl_jac[iR, iC] += np.sum(rb_nl_jac[:, iRs, iCs] * _basis_time[:, iRt] * _basis_time[:, iCt])
            logger.debug(f"Elapsed time ST-RB non-linear jacobian assembling: {(time.time() - start):.6f}")

            return rbst_nl_jac


__all__ = [
    "AffineDecompositionHandlerSTRB"
]