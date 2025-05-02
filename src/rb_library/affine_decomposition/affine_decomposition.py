#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:41:45 2018
@author: Niccolo' Dal Santo
@email : niccolo.dalsanto@epfl.ch
"""

import numpy as np
import os

import logging.config

import src.utils.array_utils as arr_utils

# Create logger
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class AffineDecomposition:
    """Class which defines the affine decomposition features of a steady problem

    :var self.M_qa: number of affine components of the stiffness matrix
    :var self.M_qf: number of affine components of the right-hand side vector
    """

    def __init__(self, _qa=None, _qf=None):
        """AffineDecomposition class initialization

        :param _qa: number of affine components for the stiffness matrix, if not None. If None, the corresponding
           class attribute is set to 0
        :type _qa: int or NoneType
        :param _qf: number of affine components for the right-hand side vector, if not None. If None, the corresponding
           class attribute is set to 0
        :type _qf: int or NoneType
        """

        self.M_qa = _qa if _qa is not None else 0
        self.M_qf = _qf if _qf is not None else 0
        return

    @property
    def qa(self):
        """Getter method, which returns the number of affine components for the stiffness matrix

        :return: number of affine components for the stiffness matrix
        :rtype: int
        """
        return self.M_qa

    @property
    def qf(self):
        """Getter method, which returns the number of affine components for the right-hand side vector

        :return: number of affine components for the right-hand side vector
        :rtype: int
        """
        return self.M_qf

    def set_Q(self, _qa, _qf):
        """Setter method, which allows to set the number of affine components for the stiffness matrix and for the
        right-hand side vector, in case it has not already been done in
        :func:`~affine_decomposition.AffineDecomposition.__init__`

        :param _qa: number of affine components for the stiffness matrix
        :type _qa: int
        :param _qf: number of affine components for the right-hand side vector
        :type _qf: int
        """
        self.M_qa = _qa
        self.M_qf = _qf
        return

    def print_ad_summary(self):
        """Method to print the main features of the AffineDecomposition class instance
        """
        logger.info(f"\n------------- AD SUMMARY -------------\n"
                    f"Number of affine decomposition matrices A {self.M_qa}\n"
                    f"Number of affine decomposition vectors  f {self.M_qf}")
        return


class AffineDecompositionHandler:
    """Class which handles the affine decomposition of the FOM problem at hand, interfacing with the FomProblem class
    and the AffineDecomposition class
    """

    def __init__(self,  _qa=None, _qf=None):
        """Initialization of the AffineDecompositionHandler class

        :param _qa: number of affine components for the stiffness matrix, if not None. If None, the corresponding
           class attribute is set to 0
        :type _qa: int or NoneType
        :param _qf: number of affine components for the right-hand side vector, if not None. If None, the corresponding
           class attribute is set to 0
        """

        self.M_feAffineAq = []
        self.M_feAffineFq = []

        self.M_rbAffineAq = []
        self.M_rbAffineFq = []

        self.M_rbAffineNLTerm = []
        self.M_rbAffineNLTermJac = []

        self.M_N = 0
        self.M_Nh = 0

        self.M_affineDecomposition = AffineDecomposition(_qa=_qa, _qf=_qf)

        self.M_stiffness = np.zeros([])
        self.M_mass = np.zeros([])

        return

    @property
    def qa(self):
        """Getter method, which returns the number of affine components for the stiffness matrix

        :return: number of affine components for the stiffness matrix
        :rtype: int
        """
        return self.M_affineDecomposition.qa

    @property
    def qf(self):
        """Getter method, which returns the number of affine components for the right-hand side vector

        :return: number of affine components for the right-hand side vector
        :rtype: int
        """
        return self.M_affineDecomposition.qf

    def set_Q(self, _qa, _qf):
        """Setter method, which allows to set the number of affine components for the stiffness matrix and for the
        right-hand side vector.

        :param _qa: number of affine components for the stiffness matrix
        :type _qa: int
        :param _qf: number of affine components for the right-hand side vector
        :type _qf: int
        """
        self.M_affineDecomposition.set_Q(_qa, _qf)
        return

    def get_affine_matrix(self, _q, _fom_problem=None):
        """Getter method, which returns the affine component of index _q among the FOM affine components of the
        stiffness matrix. If _q exceeds the number of affine components for the stiffness matrix an IndexError is raised
        If the affine matrices are not stored in the class, the method which computes them is called, provided that an
        instance of FomProblem is given as input

        :param _q: index of the desired affine component for the stiffness matrix
        :type _q: int
        :param _fom_problem: fom problem at hand
        :type: FomProblem
        :return: desired FOM affine component of the stiffness matrix
        :rtype: int
        """

        if _fom_problem is not None and not self.check_set_fom_arrays():
            self.import_fom_affine_arrays(_fom_problem, operators={'A'})

        if _q < self.M_affineDecomposition.qa:
            return self.M_feAffineAq[_q]
        else:
            raise IndexError

    def get_affine_vector(self, _q, _fom_problem=None):
        """Getter method, which returns the affine component of index _q among the FOM affine components of the
        right-hand side. If _q exceeds the number of affine components for the right-hand side an IndexError is raised.
        If the affine vectors are not stored in the class, the method which computes them is called, provided that an
        instance of FomProblem is given as input

        :param _q: index of the desired affine component for the right-hand side vector
        :type _q: int
        :param _fom_problem: fom problem at hand
        :type: FomProblem
        :return: desired FOM affine component of the right-hand side vector
        :rtype: int
        """

        if _fom_problem is not None and not self.check_set_fom_arrays():
            self.import_fom_affine_arrays(_fom_problem, operators={'f'})

        if _q < self.M_affineDecomposition.qf:
            return self.M_feAffineFq[_q]
        else:
            raise IndexError

    def import_affine_matrices(self, _input_file):
        """Method which allows to import the FOM affine matrices for the stiffness matrix from file

        :param _input_file: partial path to the files which contain the affine matrices for the stiffness matrix.
            Partial since the final file name is actually retrieved from '_input_file' by adding the operator name and
            the index of the affine component which is currently imported
        :type _input_file: str
        """

        Qa = self.M_affineDecomposition.qa
        assert Qa > 0
        self.M_feAffineAq = []

        for iQa in range(Qa):
            try:
                self.M_feAffineAq.append(np.loadtxt(_input_file + str(iQa) + '.txt'))  # importing sparse matrix

                #  if the matrix indices start from 1, we rescale them
                if np.min(self.M_feAffineAq[iQa][:, :2]) > 0:
                    self.M_feAffineAq[iQa][:, :2] = self.M_feAffineAq[iQa][:, :2] - 1

            except (IOError, OSError, FileNotFoundError) as e:
                logger.error(f"Error {e}: impossible to load the affine matrices A")
                break

        return

    def import_affine_vectors(self, _input_file):
        """Method which allows to import the FOM affine matrices for the right-hand side vector from file

        :param _input_file: partial path to the files which contain the affine vectors for the right-hand side vector
            Partial since the final file name is actually retrieved from '_input_file' by adding the operator name and
            the index of the affine component which is currently imported
        :type _input_file: str
        """

        Qf = self.M_affineDecomposition.qf

        assert Qf > 0

        self.M_feAffineFq = []

        for iQf in range(Qf):
            try:
                self.M_feAffineFq.append(np.loadtxt(_input_file + str(iQf) + '.txt'))  # importing vectors

            except (IOError, OSError, FileNotFoundError) as e:
                logger.error(f"Error {e}: impossible to load the affine vectors f")
                break

        return

    def import_fom_affine_arrays(self, _fom_problem, operators=None):
        """Method which allows to 'import' the FOM affine arrays for the operators passed in input. If no operator is
        passed, the FOM affine arrays are constructed for both the stiffness matrix and the right-hand side vector.
        The FOM affine arrays are constructed via a call to the external engine that has been initialized, if any,
        passing through a wrapping function of the FomProblem class.

        :param _fom_problem: fom problem at hand
        :type: FomProblem
        :param operators: operators for which the FOM affine components have to be constructed. Admissible values are
            'A' for the stiffness matrix and 'f' for the right-hand side vector. Defaults to None
        :type operators: set or NoneType
        """

        if operators is None:
            operators = {'A', 'f'}

        if 'f' in operators:
            if len(self.M_feAffineFq) < self.qf:
                logger.info("I am importing affine arrays for operator f ")

                ff = _fom_problem.retrieve_fom_affine_components('f', self.qf - len(self.M_feAffineFq))
                starting_Qf = len(self.M_feAffineFq)

                for iQf in range(self.qf - starting_Qf):
                    self.M_feAffineFq.append(np.array(ff['f' + str(iQf)]))

        if 'A' in operators:
            if len(self.M_feAffineAq) < self.qa:
                logger.info("I am importing affine arrays for operator A starting from %d to %d "
                      % (len(self.M_feAffineAq), self.qa-1))

                AA = _fom_problem.retrieve_fom_affine_components('A', self.qa - len(self.M_feAffineAq))
                starting_Qa = len(self.M_feAffineAq)

                for iQa in range(self.qa - starting_Qa):
                    logger.info("I am importing A affine array %d " % (iQa + starting_Qa))
                    self.M_feAffineAq.append(AA['A' + str(iQa)])

        return

    def build_nonlinear_term(self, u, _fom_problem):
        """Method that builds the FOM non-linear term based on the current value of the solution (passed as first
        argument to the function). If the problem is linear, it simply return an empty array of the right dimension.

        :param u: current FOM solution
        :type u: numpy.ndarray
        :param _fom_problem: FOM problem at hand
        :type _fom_problem: FomProblem
        :return FOM non-linear term (array of zeros if the problem is linear)
        :rtype: numpy.ndarray
        """

        if _fom_problem.is_linear():
            return np.zeros_like(u)
        else:
            return _fom_problem.retrieve_fom_nonlinear_term(u)

    def build_nonlinear_jacobian(self, u, _fom_problem):
        """Method that builds the jacobian matrix associated to the FOM non-linear term based on the current value
        of the solution (passed as first argument to the function). If the problem is linear, it simply return an empty
        array of the right dimension.

        :param u: current FOM solution
        :type u: numpy.ndarray
        :param _fom_problem: FOM problem at hand
        :type _fom_problem: FomProblem
        :return FOM non-linear jacobian (matrix of zeros if the problem is linear)
        :rtype: numpy.ndarray
        """

        if _fom_problem.is_linear():
            return np.zeros((u.shape[0], u.shape[0]))

        else:
            return _fom_problem.retrieve_fom_nonlinear_jacobian(u)

    def print_ad_summary(self):
        """Method to print the main features of the AffineDecomposition class instance
        """
        self.M_affineDecomposition.print_ad_summary()
        return

    def print_rb_affine_components(self):
        """Method to print the RB affine components for both the stiffness matrix and the right-hand side vector
        """

        Qf = self.qf
        Qa = self.qa

        for iQf in range(Qf):
            logger.info('\nRB rhs affine components %d \n' % iQf)
            logger.info(self.M_rbAffineFq[iQf])

        for iQa in range(Qa):
            logger.info('\nRB mat affine components %d \n' % iQa)
            logger.info(self.M_rbAffineAq[iQa])

        return

    def reset_rb_approximation(self):
        """Method to reset the RB affine components for both the stiffness matrix and the right-hand side vector by
        redefining the lists containing such elements as empty lists
        """
        self.M_rbAffineFq = []
        self.M_rbAffineAq = []

        return

    def set_affine_a(self, _feAffineAq):
        """Method to set the FOM affine matrices for the stiffness matrix

        :param _feAffineAq: list containing all the FOM affine matrices for the stiffness matrix
        :type _feAffineAq: list[numpy.ndarray]

        """
        self.M_feAffineAq = _feAffineAq

        return

    def set_affine_f(self, _feAffineFq):
        """Method to set the FOM affine vectors for the right-hand side vector

        :param _feAffineFq: list containing all the FOM affine vectors for the right-hand side vector
        :type _feAffineFq: list[numpy.ndarray]
        """
        self.M_feAffineFq = _feAffineFq

        return

    def check_set_fom_arrays(self):
        """Method which checks whether the FOM affine arrays are stored in the class instance, either by constructing
        them or by importing them from file

        :return: True if the FOM affine arrays are stored in the class instance, False otherwise
        :rtype: bool
        """
        return len(self.M_feAffineAq) == self.qa and len(self.M_feAffineFq) == self.qf

    def build_rb_affine_decompositions(self, _basis, _fom_problem, operators=None):
        """Method which constructs the RB affine components for the operators passed in input, suitably multiplying the
        considered affine component by the matrix encoding the reduced basis; in particular vectors are pre-multiplied
        by the transpose of such matrix, while matrices are pre-multiplied by the transpose basis matrix and
        post-multiplied by the basis matrix itself. If no operator is passed, the RB affine arrays are constructed for
        the stiffness matrix and the right-hand side vector. If the FOM affine arrays are not yet stored in the class,
        they are constructed via a call to
        :func:`~affine_decomposition.AffineDecompositionHandler.import_fom_affine_arrays`.

        :param _basis: matrix encoding the reduced basis
        :type _basis: numpy.ndarray
        :param _fom_problem: FOM problem at hand
        :type _fom_problem: FomProblem
        :param operators: operators for which the RB affine components have to be constructed. Admissible values are 'A'
            for the stiffness matrix, 'M' for the mass matrix and 'f' for the right-hand side vector. Defaults to None.
        :type operators: set or NoneType
        """

        if operators is None:
            operators = {'A', 'f', 'M'}
        self.M_Nh= _basis.shape[0]
        self.M_N = _basis.shape[1]

        Qf = self.qf
        Qa = self.qa

        if not self.check_set_fom_arrays():
            logger.info("Importing FOM affine arrays ")
            self.import_fom_affine_arrays(_fom_problem)
        else:
            logger.debug("Already set the FOM affine arrays ")

        if 'f' in operators:
            for iQf in range(Qf):
                self.M_rbAffineFq.append(np.zeros(self.M_N))
                self.M_rbAffineFq[iQf] = _basis.T.dot(self.M_feAffineFq[iQf])

        if 'A' in operators:
            for iQa in range(Qa):
                Av = arr_utils.sparse_matrix_vector_mul(self.M_feAffineAq[iQa], _basis)
                self.M_rbAffineAq.append(np.zeros((self.M_N, self.M_N)))
                self.M_rbAffineAq[iQa] = _basis.T.dot(Av)

        logger.info('Finished to build the RB affine arrays')

        return

    def build_rb_nonlinear_term(self, u, _basis, _fom_problem):
        """Method that builds the reduced non-linear term based on the current value of the solution (passed as first
        argument to the function). If the problem is linear, it simply return an empty array of the right dimension.

        :param u: current solution
        :type u: numpy.ndarray
        :param _basis: matrix encoding the reduced basis
        :type _basis: numpy.ndarray
        :param _fom_problem: FOM problem at hand
        :type _fom_problem: FomProblem
        :return reduced non-linear term (array of zeros if the problem is linear)
        :rtype: numpy.ndarray
        """

        if _fom_problem.is_linear():
            return np.zeros_like(u)

        else:
            u_fom = _basis.dot(u)
            nl_term = self.build_nonlinear_term(u_fom, _fom_problem)
            rb_nl_term = _basis.T.dot(nl_term)
            return rb_nl_term

    def build_rb_nonlinear_jacobian(self, u, _basis, _fom_problem):
        """Method that builds the jacobian matrix associated to the reduced non-linear term based on the current value
        of the solution (passed as first argument to the function). If the problem is linear, it simply return an empty
        array of the right dimension.

        :param u: current solution
        :type u: numpy.ndarray
        :param _basis: matrix encoding the reduced basis
        :type _basis: numpy.ndarray
        :param _fom_problem: FOM problem at hand
        :type _fom_problem: FomProblem
        :return reduced non-linear jacobian (matrix of zeros if the problem is linear)
        :rtype: numpy.ndarray
        """

        if _fom_problem.is_linear():
            return np.zeros((u.shape[0], u.shape[0]))

        else:
            u_fom = _basis.dot(u)
            nl_jac = self.build_nonlinear_jacobian(u_fom, _fom_problem)
            tmp = arr_utils.sparse_matrix_matrix_mul(nl_jac, _basis)
            rb_nl_jac = _basis.T.dot(tmp)
            return rb_nl_jac

    def get_rb_affine_matrix(self, _q):
        """Getter method, which returns the affine component of index _q among the FOM affine components of the
        stiffness matrix, projected over the RB-space. If _q exceeds the number of affine components for the stiffness
        matrix an IndexError is raised.

        :param _q: index of the desired affine component for the stiffness matrix
        :type _q: int
        :return: desired RB affine component of the stiffness matrix
        :rtype: int
        """

        if _q < self.M_affineDecomposition.qa:
            return self.M_rbAffineAq[_q]
        else:
            raise IndexError

    def get_rb_affine_vector(self, _q):
        """Getter method, which returns the affine component of index _q among the FOM affine components of the
        right-hand side. If _q exceeds the number of affine components for the right-hand side an IndexError is raised.

        :param _q: index of the desired affine component for the right-hand side vector
        :type _q: int
        :return: desired FOM affine component of the right-hand side vector
        :rtype: int
        """
        if _q < self.M_affineDecomposition.qf:
            return self.M_rbAffineFq[_q]
        else:
            raise IndexError

    def import_rb_affine_matrices(self, _input_file):
        """Method which allows to import the RB affine matrices for the stiffness matrix from file

        :param _input_file: partial path to the files which contain the RB affine matrices for the stiffness matrix.
            Partial since the final file name is actually retrieved from '_input_file' by adding the operator name and
            the index of the affine component which is currently imported
        :type _input_file: str
        """

        Qa = self.qa
        assert Qa > 0

        self.M_rbAffineAq = []

        try:
            for iQa in range(Qa):
                self.M_rbAffineAq.append(np.loadtxt(_input_file + str(iQa) + '.txt'))  # importing RB matrix

        except (IOError, OSError, FileNotFoundError) as e:
            logger.error(f"Error {e}: impossible to load the RB affine matrices")

        return

    def import_rb_affine_vectors(self, _input_file):
        """Method which allows to import the RB affine matrices for the right-hand side vector from file

        :param _input_file: partial path to the files which contain the RB affine vectors for the right-hand side vector
            Partial since the final file name is actually retrieved from '_input_file' by adding the operator name and
            the index of the affine component which is currently imported
        :type _input_file: str
        """

        Qf = self.qf
        assert Qf > 0

        self.M_rbAffineFq = []

        try:
            for iQf in range(Qf):
                self.M_rbAffineFq.append(np.loadtxt(_input_file + str(iQf) + '.txt'))  # importing RB vectors

        except (IOError, OSError, FileNotFoundError) as e:
            logger.error(f"Error {e}: impossible to load the RB affine vectors")

        return

    def import_rb_affine_components(self, _input_file, operators=None):
        """Method which allows to import the RB affine components for the operators passed in input from file. If no
        operator is passed, the RB affine arrays are constructed for both the stiffness matrix and the right-hand side
        vector. If the import of the affine components for a certain operator has failed, the operator name is stored
        in a set which is finally returned by the method itself

        :param _input_file: partial path to the files which contain the RB affine vectors for the right-hand side vector
            Partial since the final file name is actually retrieved from '_input_file' by adding the operator name and
            the index of the affine component which is currently imported
        :type _input_file: str
        :param operators: operators for which the RB affine components have to be constructed. Admissible values are 'A'
            for the stiffness matrix and 'f' for the right-hand side vector. Defaults to None.
        :type operators: set or NoneType
        :return: set containing the name of the operators whose import of the RB affine components has failed
        :rtype: set or NoneType
        """

        if operators is None:
            operators = {'A', 'f'}

        self.M_rbAffineAq = []
        self.M_rbAffineFq = []

        Qf = self.qf
        Qa = self.qa

        import_failures_set = set()

        if 'A' in operators:
            for iQa in range(Qa):
                try:
                    self.M_rbAffineAq.append(np.loadtxt(_input_file + '_A' + str(iQa) + '.txt'))
                except (IOError, OSError, FileNotFoundError) as e:
                    logger.error(f"Error {e}. Impossible to open the desired file for matrix A{str(iQa)}.")
                    import_failures_set.add('A')
                    break

        if 'f' in operators:
            for iQf in range(Qf):
                try:
                    self.M_rbAffineFq.append(np.loadtxt(_input_file + '_f' + str(iQf) + '.txt'))
                except (IOError, OSError, FileNotFoundError) as e:
                    logger.error(f"Error {e}. Impossible to open the desired file for vector f{str(iQf)}.")
                    import_failures_set.add('f')
                    break

        return import_failures_set

    def set_affine_rb_a(self, _rbAffineAq):
        """Method to set the RB affine matrices for the stiffness matrix

        :param _rbAffineAq: list containing all the RB affine matrices for the stiffness matrix
        :type _rbAffineAq: list[numpy.ndarray]

        """
        self.M_rbAffineAq = _rbAffineAq

        return

    def set_affine_rb_f(self, _rbAffineFq):
        """Method to set the RB affine vectors for the right-hand side vector

        :param _rbAffineFq: list containing all the RB affine vectors for the right-hand side vector
        :type _rbAffineFq: list[numpy.ndarray]
        """
        self.M_rbAffineFq = _rbAffineFq

        return

    def set_rb_basis_dim(self, N):
        """Method to set the dimension of the RB basis. If a negative number is passed, the dimension is defaulted to 0.

        :param N: dimension of the RB basis
        :type N: int
        """

        self.M_N = N if N > 0 else 0

        return

    def check_set_rb_arrays(self):
        """Method which checks whether the RB affine arrays are stored in the class instance, either by constructing
        them or by importing them from file

        :return: True if the RB affine arrays are stored in the class instance, False otherwise
        :rtype: bool
        """
        return len(self.M_rbAffineAq) == self.qa and len(self.M_rbAffineFq) == self.qf

    def save_rb_affine_decomposition(self, _file_name, operators=None):
        """Method which saves the RB affine components of the operators passed in input to text files, whose path has
        been specified by the input argument '_file_name'. If no operator is passed, the RB affine arrays are saved for
        both the stiffness matrix and the right-hand side vector. The final file name is constructed by adding to the
        input file name the operator name and the index of the affine components which is currently saved.

        :param _file_name: partial path to the files where the RB affine components for the desired operators are saved
            Partial since the final file name is completed with the operator name and the index of the affine component
            which is currently saved
        :type _file_name: str
        :param operators: operators for which the RB affine components have to be saved. Admissible values are 'A'
            for the stiffness matrix, 'M' for the mass matrix and 'f' for the right-hand side vector. Defaults to None
        :type operators: set or NoneType
        """

        if operators is None:
            operators = {'A', 'f'}

        Qf = self.qf
        Qa = self.qa

        if 'A' in operators:
            for iQa in range(Qa):
                path = _file_name + '_A' + str(iQa) + '.txt'
                arr_utils.save_array(self.M_rbAffineAq[iQa], path)

        if 'f' in operators:
            for iQf in range(Qf):
                path = _file_name + '_f' + str(iQf) + '.txt'
                arr_utils.save_array(self.M_rbAffineFq[iQf], path)

        return

    def get_mass_stiffness_matrices(self, _fom_problem, withMass=True):
        """Method which allows to construct and store in the class the stiffness and the mass matrices; such matrices
        are intended to be used for the computation of the L2 and H10 norms, thus the stiffness matrix is computed by
        assuming all the parameters to be equal to 1. The construction is performed via the external engine, passing
        through an intermediate wrapping function in the FomProblem class. Additionally, it is possible to just compute
        the stiffness matrix, by setting the input flag 'withMass' to False. If the stiffness matrix is already stored
        in the class it will not be computed again

        :param _fom_problem: fom problem at hand
        :type _fom_problem: FomProblem
        :param withMass: if True also the mass matrix is computed. Defaults to True
        :type withMass: bool
        :return: stiffness matrix and, if `withMass==True`, mass matrix
        :rtype: tuple(numpy.ndarray, numpy.ndarray) or numpy.ndarray
        """

        M_stiffness = self.M_stiffness
        M_mass = self.M_mass

        if withMass:
            if len(self.M_feAffineAq):
                M_stiffness = np.sum(self.M_feAffineAq, axis=0)

            if M_stiffness is None or len(M_stiffness.shape) == 0:
                M_stiffness, M_mass = _fom_problem.assemble_fom_matrix(1, withMass=True)
            else:
                _, M_mass = _fom_problem.assemble_fom_matrix(1, withMass=True)
            logger.debug("Computed Mass and Stiffness matrices")

        else:
            if len(self.M_feAffineAq):
                M_stiffness = np.sum(self.M_feAffineAq, axis=0)
            else:
                M_stiffness = _fom_problem.assemble_fom_matrix(1, withMass=False)
            logger.debug("Computed Stiffness matrix")

        return M_stiffness, M_mass

    def compute_norm(self, vec, _fom_problem=None, norm_types=None):
        """Method which allows to compute some different norms of a given vector vec. In particular, it allows to
        compute the l2, L2, H1 and H10 norms; the last three can readily be computed only if the input vector is a FOM
        vector, whose dimensions match with the ones of the mass matrix and of the stiffness matrix; if not so, a
        ValueError is raised. Also, the last three norms can be computed only if the mass and stiffness matrix are
        stored in the class instance; if not so, the method can construct them, provided that a FomProblem instance is
        passed in input. Finally, if no norm type is specified, the L2 norm is computed.

        :param vec: vector whose norm(s) has(ve) to be computed
        :type vec: numpy.ndarray
        :param _fom_problem: fom problem at hand. Defaults to None
        :type _fom_problem: FomProblem or NoneType
        :param norm_types: type of the norms that have to be computed for the input vector 'vec'. Defaults to None
        :type norm_types: set{str} or NoneType
        :return: norms that have been computed for the vector 'vec' in the form of a dictionary; the key represents the
            norm type, while the value represents the corresponding value of the norm.
        :rtype: dict{str: float}
        """

        if norm_types is None:
            norm_types = {"L2"}

        if "L2" in norm_types or "H1" in norm_types or "H10" in norm_types:
            assert _fom_problem is not None, \
                "'_fom_problem' must be passed to the function to compute L2, H1 or H10 norms"

            if not len(self.M_stiffness.shape) or not len(self.M_mass.shape):
                withMass = "L2" in norm_types or "H1" in norm_types
                if withMass:
                    self.M_stiffness, self.M_mass = _fom_problem.assemble_fom_matrix(1, withMass=True)
                else:
                    self.M_stiffness = _fom_problem.assemble_fom_matrix(1, withMass=False)

        if len(vec.shape) == 1:
            _fun1 = arr_utils.sparse_matrix_vector_mul
            _fun2 = np.dot
        elif len(vec.shape) == 2:
            _fun1 = arr_utils.sparse_matrix_matrix_mul
            _fun2 = lambda a,b: np.einsum('ij,ij->j', a, b)
        else:
            raise ValueError(f"The input has an invalid shape: {vec.shape}")

        norms = dict()

        if "l2" in norm_types:
            norms["l2"] = np.sqrt(np.sum(np.square(vec), axis=0))

        if "L2" in norm_types:
            norms["L2"] = np.sqrt(_fun2(vec, _fun1(self.M_mass, vec)))

        if "H10" in norm_types:
            norms["H10"] = np.sqrt(_fun2(vec, _fun1(self.M_stiffness, vec)))

        if "H1" in norm_types:
            if "L2" in norm_types and "H10" in norm_types:
                norms["H1"] = norms["L2"] ** 2 + norms["H10"] ** 2
            elif "L2" in norm_types:
                norms["H1"] = norms["L2"] ** 2
                norms["H1"] += _fun2(vec, _fun1(self.M_stiffness, vec))
            elif "H10" in norm_types:
                norms["H1"] = norms["H10"] ** 2
                norms["H1"] += _fun2(vec, _fun1(self.M_mass, vec))
            else:
                norms["H1"] = _fun2(vec, _fun1(self.M_mass, vec))
                norms["H1"] += _fun2(vec, _fun1(self.M_stiffness, vec))

            norms["H1"] = np.sqrt(norms["H1"])

        return norms


__all__ = [
    "AffineDecomposition",
    "AffineDecompositionHandler"
]

