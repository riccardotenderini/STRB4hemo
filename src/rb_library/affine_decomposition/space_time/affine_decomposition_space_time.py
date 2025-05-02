#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 8 13:31:57 2019
@author: Riccardo Tenderini
@mail: riccardo.tenderini@epfl.ch
"""

import numpy as np
import os
import time

import src.rb_library.affine_decomposition.affine_decomposition_unsteady as adu
import src.utils.array_utils as arr_utils

import logging.config
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class AffineDecompositionHandlerSpaceTime(adu.AffineDecompositionHandlerUnsteady):
    """Class which handles the affine decomposition of the unsteady FOM problem at hand, interfacing with the
    FomProblemUnsteady class and the AffineDecompositionUnsteady class.
    It inherits from It inherits from :class:`~affine_decomposition_unsteady.AffineDecompositionHandlerUnsteady`
    """

    def __init__(self, _qa=None, _qf=None, _qm=None):
        """Initialization of the AffineDecompositionHandlerSpaceTime class

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

        self.M_rbstAffineMatq = []
        self.M_rbstAffineFq = []

        return

    def print_rb_affine_components(self):
        """Method to print the RB affine components for both the left-hand side matrix and the right-hand side vector,
        arising after the application of the ST-LSPG method for the resolution of the FOM problem a hand
        """

        Qf = self.qf
        Qa = self.qa
        Qm = self.qm

        for iQf in range(Qf):
            logger.info(f"RB rhs affine components {iQf}:\n {self.M_rbstAffineFq[iQf]}")

        for iQ in range(Qa + 2*Qm):
            logger.info(f"RB mat affine components {iQ}:\n {self.M_rbstAffineMatq}")

        return

    def reset_rb_approximation(self):
        """Method to reset the RB affine components for both the stiffness matrix and the right-hand side vector by
        redefining the lists containing such elements as empty lists
        """

        super().reset_rb_approximation()

        self.M_rbstAffineMatq = []
        self.M_rbstAffineFq = []

        return

    def get_rb_affine_matrix(self, _q):
        """Getter method, which returns the affine component of index _q among the RB affine components of the left-hand
        side matrix, which arises after the application of the ST-LSPG method for the resolution of an unsteady problem.
        If _qa exceeds the number of affine components for the left-hand side matrix an IndexError is raised.

        :param _q: index of the desired affine component for the left-hand side matrix
        :type _q: int
        :return: desired RB affine component of the left-hand side matrix
        :rtype: int
        """
        Qa = self.qa
        Qm = self.qm
        if _q < Qa + 2*Qm:
            return self.M_rbstAffineMatq[_q]
        else:
            raise IndexError

    def get_rb_affine_vector(self, _q, timestep=None):
        """Getter method, which returns the affine component of index _q among the FOM affine components of the
        right-hand side. If _q exceeds the number of affine components for the right-hand side an IndexError is raised.

        :param _q: index of the desired affine component for the reduced right-hand side vector
        :type _q: int
        :param timestep: added for signature compatibility with parent function
        :type timestep: NoneType
        :return: desired space-time RB affine component of the right-hand side vector
        :rtype: int
        """
        if _q < self.M_affineDecomposition.qf:
            return self.M_rbstAffineFq[_q]
        else:
            raise IndexError

    def import_rb_affine_components(self, _input_file, operators=None):
        """Method which allows to import the RB affine components for the operators passed in input from file. If no
        operator is passed, the RB affine arrays are saved for the left-hand side matrix and the right-hand side vector,
        which arise after the application of the ST-LSPG method for the resolution of the fom problem at hand. If the
        import of the affine components for a certain operator has failed, the operator name is stored in a set which is
        finally returned by the method itself

        :param _input_file: partial path to the files which contain the RB affine vectors for the right-hand side vector
          Partial since the final file name is actually retrieved from '_input_file' by adding the operator name and
          the index of the affine component which is currently imported
        :type _input_file: str
        :param operators: operators for which the affine components have to be imported. Admissible values are
          'Mat' for the left-hand side matrix and 'f' for the right-hand side vector arising after the application of
          the ST-LSPG method to solve the unsteady FOM problem at hand. Defaults to None.
        :type operators: set or NoneType
        :return: set containing the name of the operators whose import of the RB affine components has failed
        :rtype: set or NoneType
        """

        if operators is None:
            operators = {'Mat', 'f', 'u0'}

        import_failures_set = set()

        if 'Mat' in operators:
            self.M_rbstAffineMatq = []
            Qa = self.qa
            Qm = self.qm
            for iQ in range(Qa + 2*Qm):
                try:
                    self.M_rbstAffineMatq.append(np.loadtxt(_input_file + '_Mat' + str(iQ) + '.txt'))
                except (IOError, OSError, FileNotFoundError) as e:
                    logger.error(f"Error {e}. Impossible to open the desired file for matrix Mat{str(iQ)}.")
                    import_failures_set.add('Mat')
                    break

        if 'f' in operators:
            self.M_rbAffineFq = []
            Qf = self.qf
            for iQf in range(Qf):
                try:
                    self.M_rbstAffineFq.append(np.loadtxt(_input_file + '_f' + str(iQf) + '.txt'))
                except (IOError, OSError, FileNotFoundError) as e:
                    logger.error(f"Error {e}. Impossible to open the desired file for vector f{str(iQf)}.")
                    import_failures_set.add('f')
                    break

        if 'u0' in operators:
            try:
                self.M_rbInitialCondition = np.loadtxt(_input_file + '_u0' + '.txt')
            except (IOError, OSError, FileNotFoundError) as e:
                logger.error(f"Error {e}. Impossible to open the desired file for u0.")
                import_failures_set.add('u0')

        return import_failures_set

    def set_affine_rb_matrix(self, _rbAffineMatq):
        """Method to set the RB affine components for the left-hand side matrix arising from the application of the
        ST-LSPG approach to an unsteady parametrized PDE problem

        :param _rbAffineMatq: list containing all the RB affine components for the left-hand side matrix arising from the
           application of the ST-LSPG approach to an unsteady parametrized PDE problem
        :type _rbAffineMatq: list[numpy.ndarray]
        """

        self.M_rbstAffineMatq = _rbAffineMatq

        return

    def save_rb_affine_decomposition(self, _file_name, operators=None):
        """Method which saves the RB affine components of the operators passed in input to text files, whose path has
        been specified by the input argument '_file_name'. If no operator is passed, the RB affine arrays are saved for
        the left-hand side matrix and the right-hand side vector, which arise after the application of the ST-LSPG
        method for the resolution of the fom problem at hand. The final file name is constructed by adding to the input
        file name the operator name and the index of the affine components which is currently saved.

        :param _file_name: partial path to the files where the RB affine components for the desired operators are saved
          Partial since the final file name is completed with the operator name and the index of the affine component
          which is currently saved
        :type _file_name: str
        :param operators: operators for which the affine components have to be saved. Admissible values are
          'Mat' for the left-hand side matrix and 'f' for the right-hand side vector arising after the application of
          the ST-LSPG method to solve the unsteady FOM problem at hand. Defaults to None.
        :type operators: set or NoneType
        """

        if _file_name is not None:

            if operators is None:
                operators = {'Mat', 'f', 'u0'}

            if 'Mat' in operators:
                Qa = self.qa
                Qm = self.qm
                for iQ in range(Qa + 2*Qm):
                    path = _file_name + '_Mat' + str(iQ) + '.txt'
                    arr_utils.save_array(self.M_rbstAffineMatq[iQ], path)

            if 'f' in operators:
                Qf = self.qf
                for iQf in range(Qf):
                    path = _file_name + '_f' + str(iQf) + '.txt'
                    arr_utils.save_array(self.M_rbstAffineFq[iQf], path)

            if 'u0' in operators:
                path = _file_name + '_u0' + '.txt'
                arr_utils.save_array(self.M_rbInitialCondition, path)

        return

    def build_nonlinear_jacobian(self, u, _fom_problem, recompute_every=1):
        """Method that builds the jacobian matrix associated to the FOM non-linear term based on the current value
        of the solution (passed as first argument to the function). If the problem is linear, it simply return an empty
        array of the right dimension.

        :param u: current FOM solution
        :type u: numpy.ndarray
        :param _fom_problem: FOM problem at hand
        :type _fom_problem: FomProblem
        :param recompute_every: for unsteady problems, number of time instants after which the Jacobian is recomputed.
           It defaults to 1.
        :type recompute_every: int
        :return FOM non-linear jacobian (matrix of zeros if the problem is linear)
        :rtype: numpy.ndarray
        """

        if _fom_problem.is_linear():
            return np.zeros((u.shape[0], u.shape[0]))

        else:
            return _fom_problem.retrieve_fom_nonlinear_jacobian(u, recompute_every=recompute_every)

    def build_rb_nonlinear_term(self, u, _basis_space, _basis_time, _fom_problem):
        """Method that builds the reduced non-linear term based on the current value of the solution (passed as first
        argument to the function). If the problem is linear, it simply return an empty array of the right dimension.

        :param u: current reduced solution
        :type u: numpy.ndarray
        :param _basis_space: matrix encoding the reduced basis in space
        :type _basis_space: numpy.ndarray
        :param _basis_time: matrix encoding the reduced basis in time
        :type _basis_time: numpy.ndarray
        :param _fom_problem: unsteady FOM problem at hand
        :type _fom_problem: FomProblemUnsteady
        :return reduced non-linear term (array of zeros if the problem is linear)
        :rtype: numpy.ndarray
        """

        start = time.time()

        if _fom_problem.is_linear():
            return np.zeros(_basis_space.shape[1])

        else:
            u_fom = self.reconstruct_fem_solution(u, _basis_space, _basis_time)
            logger.debug(f"Elapsed time FOM solution reconstruction for non-linear term: {time.time() - start}")

            nl_term = self.build_nonlinear_term(u_fom, _fom_problem)
            logger.debug(f"Elapsed time FOM non-linear term assembling: {time.time() - start}")

            rb_nl_term = self.compute_generalized_coordinates(nl_term, _basis_space, _basis_time)
            logger.debug(f"Elapsed time RBST non-linear term assembling: {time.time() - start}")

            return rb_nl_term

    def build_rb_nonlinear_jacobian(self, u, _basis_space, _basis_time, _fom_problem, recompute_every=1):
        """Method that builds the jacobian matrix associated to the reduced non-linear term based on the current value
        of the solution (passed as first argument to the function). If the problem is linear, it simply return an empty
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

        if _fom_problem.is_linear():
            return [np.zeros((u.shape[0], u.shape[0]))] * u.shape[1]

        else:
            u_fom = self.reconstruct_fem_solution(u, _basis_space, _basis_time)
            nl_jac = self.build_nonlinear_jacobian(u_fom, _fom_problem,
                                                   recompute_every=recompute_every)
            rb_nl_jac = []
            for ind_t in range(u_fom.shape[1]):
                tmp = arr_utils.sparse_matrix_matrix_mul(nl_jac[ind_t], _basis_space)
                rb_nl_jac.append(_basis_space.T.dot(tmp))
            return rb_nl_jac

    @staticmethod
    def compute_generalized_coordinates(u_fom, _basis_space=None, _basis_time=None):
        """Method that allows to compute the generalized coordinates  (i.e. the coefficients of the linear combination of
        the SpaceTime basis functions) associated to a given FOM solution.

        :param u_fom: FOM solution, given as a 2D array of shape (Ns, Nt)
        :type u_fom: numpy.ndarray
        :param _basis_space: matrix encoding the reduced basis in space. If None, the input is assumed already reduced
           in space. Defaults to None
        :type _basis_space: numpy.ndarray or NoneType
        :param _basis_time: matrix encoding the reduced basis in time. If None, the input is assumed already reduced
           in time. Defaults to None
        :type _basis_time: numpy.ndarray or NoneType
        :return: array of the generalized coordinates associated to the given FOM solution
        :rtype: numpy.ndarray
        """

        u_fom = np.squeeze(u_fom)

        # space-time projection
        if _basis_space is not None and _basis_time is not None:
            u_hat = _basis_space.T.dot(u_fom.dot(_basis_time)).flatten()

        # time projection only
        elif _basis_space is None and _basis_time is not None:
            u_hat = u_fom.dot(_basis_time).flatten()

        # space projection only
        elif _basis_space is not None and _basis_time is None:
            u_hat = _basis_space.T.dot(u_fom).flatten()

        # no projection at all
        else:
            u_hat = u_fom

        return u_hat

    @staticmethod
    def reconstruct_fem_solution(u_rb, _basis_space, _basis_time, _indices_space=None, _indices_time=None):
        """Method which reconstructs the FOM solution (in both space and time) from the dimensionality reduced one,
        by linearly combining the elements of the SpaceTime Reduced Basis with weights given by the entries of the
        reduced solution itself. It is the inverse method of
        :func:`~rb_manager_space_time.RbManagerSpaceTime.get_generalized_coordinates`

        :param u_rb: dimensionality reduced vector, typically arising from the Least Squares solution of the reduced
          linear system
        :type u_rb: numpy.ndarray
        :param _basis_space: matrix encoding the Reduced Basis in space
        :type _basis_space: numpy.ndarray
        :param _basis_time: matrix encoding the Reduced Basis in time
        :type _basis_time: numpy.ndarray
        :param _indices_space: indices (in space) at which reconstructing the solution. If None, the solution is
           reconstructed everywhere in space. Defaults to None
        :type _indices_space: numpy.ndarray(int) or NoneType
        :param _indices_time: indices (in time) at which reconstructing the solution. If None, the solution is
           reconstructed everywhere in time. Defaults to None
        :type _indices_time: numpy.ndarray(int) or NoneType
        :return: FOM vector in both space and time, constructed by expanding the dimensionality reduced one with
          respect to the elements of the SpaceTime basis
        :rtype: numpy.ndarray
        """

        Ns = _basis_space.shape[0]
        ns = _basis_space.shape[1]
        Nt = _basis_time.shape[0]
        nt = _basis_time.shape[1]

        times = np.arange(Nt) if _indices_time is None else _indices_time
        spaces = np.arange(Ns) if _indices_space is None else _indices_space

        u_rb = np.reshape(u_rb, (ns, nt))
        u_fom = _basis_space[spaces].dot(u_rb.dot(_basis_time[times].T))

        return u_fom


__all__ = [
    "AffineDecompositionHandlerSpaceTime"
]
