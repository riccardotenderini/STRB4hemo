#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 18:34:23 2019
@author: Riccardo Tenderini
@mail: riccardo.tenderini@epfl.ch
"""

import numpy as np
import os

import src.rb_library.affine_decomposition.affine_decomposition as ad
import src.utils.array_utils as arr_utils

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class AffineDecompositionUnsteady(ad.AffineDecomposition):
    """Class which defines the affine decomposition features of an unsteady problem.
    It inherits from :class:`~affine_decomposition.AffineDecomposition`

    :var self.M_qa: number of affine components of the stiffness matrix
    :var self.M_qf: number of affine components of the right-hand side vector
    :var self.M_qm: number of affine components of the mass matrix
    """

    def __init__(self, _qa=None, _qf=None, _qm=None):
        """AffineDecompositionUnsteady class initialization

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

        super().__init__(_qa=_qa, _qf=_qf)
        self.M_qm = _qm if _qm is not None else 0
        return

    @property
    def qm(self):
        """Getter method, which returns the number of affine components for the mass matrix

        :return: number of affine components for the mass matrix
        :rtype: int
        """

        return self.M_qm

    def set_Q(self, _qa, _qf, _qm=1):
        """Setter method, which allows to set the number of affine components for the stiffness matrix, for the mass
       matrix and for the right-hand side vector, in case it has not already been done in
       :func:`~affine_decomposition.AffineDecompositionUnsteady.__init__`

       :param _qa: number of affine components for the stiffness matrix
       :type _qa: int
       :param _qf: number of affine components for the right-hand side vector
       :type _qf: int
       :param _qm: number of affine components for the mass matrix. Defaults to 1.
       :type _qm: int
       """

        super().set_Q(_qa, _qf)
        self.M_qm = _qm

        return

    def print_ad_summary(self):
        """Method to print the main features of the AffineDecompositionUnsteady class instance
        """

        logger.info(f"\n------------- AD SUMMARY -------------\n"
                    f"Number of affine decomposition matrices A {self.M_qa}\n"
                    f"Number of affine decomposition vectors  f {self.M_qf}\n"
                    f"Number of affine decomposition matrices M {self.M_qm}\n")

        return


class AffineDecompositionHandlerUnsteady(ad.AffineDecompositionHandler):
    """Class which handles the affine decomposition of the unsteady FOM problem at hand, interfacing with the
    FomProblemUnsteady class and the AffineDecompositionUnsteady class. It inherits from AffineDecomposition
    """

    def __init__(self, _qa=None, _qf=None, _qm=None):
        """Initialization of the AffineDecompositionHandlerUnsteady class

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

        super().__init__()

        self.M_feAffineMq = []
        self.M_rbAffineMq = []
        self.M_feAffineFReducedq = []
        self.M_rbAffineFReducedq = []
        self.M_feAffineFSpaceq = []
        self.M_rbAffineFSpaceq = []
        self.M_feInitialCondition = np.zeros([])
        self.M_rbInitialCondition = np.zeros([])

        self.M_affineDecomposition = AffineDecompositionUnsteady(_qa=_qa, _qf=_qf, _qm=_qm)
        self.M_Nt = 0
        self.M_Nt_Reduced = 0

        return

    @property
    def qm(self):
        """Getter method, which returns the number of affine components for the mass matrix

        :return: number of affine components for the mass matrix
        :rtype: int
        """

        return self.M_affineDecomposition.qm

    def set_Q(self, _qa, _qf, _qm=1):
        """Setter method, which allows to set the number of affine components for the stiffness matrix, for the mass
        matrix and for the right-hand side vector.

        :param _qa: number of affine components for the stiffness matrix
        :type _qa: int
        :param _qf: number of affine components for the right-hand side vector
        :type _qf: int
        :param _qm: number of affine components for the mass matrix. Defaults to 1
        :type _qm: int
        """

        self.M_affineDecomposition.set_Q(_qa, _qf, _qm)
        return

    def set_timesteps(self, _timesteps, _timesteps_reduced=None):
        """Method which allows to set the number of timesteps used for the time integration of the problem at hand. In
        particular, it sets the timesteps which have been used for the time integration in snapshots computation (via
        the input argument _timesteps) and, optionally, the timesteps which are used to perform the time integration
        of the RB solution (via the input argument _timesteps_reduced), which can be selected as an equispaced subset
        of the 'original' FOM timesteps. In case the latter value is not passed, the corresponding class attribute is
        initialized to the 'original' number of timesteps '_timesteps'.

        :param _timesteps: number of timesteps used for the time integration of the snapshots (FOM problem)
        :type _timesteps: int
        :param _timesteps_reduced: number of timesteps used for the time integration of the RB solution (RB problem).
           If None, the corresponding class attribute is not initialized. Defaults to None.
        :type _timesteps_reduced: int or NoneType
        """

        self.M_Nt = _timesteps
        if _timesteps_reduced is not None:
            self.M_Nt_Reduced = int(_timesteps_reduced)
        else:
            self.M_Nt_Reduced = _timesteps
        return

    @property
    def Nt(self):
        return self.M_Nt

    @property
    def Nt_reduced(self):
        return self.M_Nt_Reduced

    def get_timesteps_length(self, _fom_problem):
        """Method which allows to get the length of the timesteps used for the time integration of both the FOM problem
        (i.e. snapshots computation) and the RB problem

        :param _fom_problem: unsteady fom problem at hand
        :type _fom_problem: FomProblem
        :return: lengths of the timesteps used for the time integration of the FOM problem and of the RB problem
        :rtype: tuple(int, int)
        """

        try:
            assert 'final_time' in _fom_problem.M_fom_specifics.keys()
        except AssertionError:
            logger.critical("The FomProblem must be unsteady to get the timesteps lengths")
            raise TypeError

        return _fom_problem.M_fom_specifics['final_time'] / self.M_Nt, \
               _fom_problem.M_fom_specifics['final_time'] / self.M_Nt_Reduced

    def get_affine_matrix_A(self, _q, _fom_problem=None):
        """Getter method, which returns the affine component of index _q among the FOM affine components of the
        stiffness matrix. If _q exceeds the number of affine components for the stiffness matrix an IndexError is raised
        If the affine matrices are not stored in the class, the method which computes them is called, provided that an
        instance of FomProblem is given as input

        :param _q: index of the desired affine component for the stiffness matrix
        :type _q: int
        :param _fom_problem: fom problem at hand
        :type: FomProblem
        :return: desired FOM affine component of the stiffness matrix
        :rtype: numpy.ndarray
        """

        return super().get_affine_matrix(_q, _fom_problem=_fom_problem)

    def get_affine_matrix_M(self, _q=0, _fom_problem=None):
        """Getter method, which returns the affine component of index _q among the FOM affine components of the
       mass matrix. If _q exceeds the number of affine components for the mass matrix an IndexError is raised
       If the affine matrices are not stored in the class, the method which computes them is called, provided that an
       instance of FomProblemUnsteady is given as input

       :param _q: index of the desired affine component for the mass matrix. Defaults to 0
       :type _q: int
       :param _fom_problem: fom problem at hand
       :type: FomProblemUnsteady
       :return: desired FOM affine component of the mass matrix
       :rtype: numpy.ndarray
       """

        if _fom_problem is not None:
            try:
                assert 'final_time' in _fom_problem.M_fom_specifics.keys()
            except AssertionError:
                logger.critical("The FomProblem must be unsteady to get the affine matrices for the mass matrix M")
                raise TypeError

        if _fom_problem is not None and not self.check_set_fom_arrays():
            self.import_fom_affine_arrays(_fom_problem, operators={'M'})

        if _q < self.M_affineDecomposition.qm:
            return self.M_feAffineMq[_q]
        else:
            raise IndexError

    def get_affine_vector(self, _q, _fom_problem=None):
        """Getter method, which returns the affine component of index _q among the FOM affine components of the
        right-hand side. If _q exceeds the number of affine components for the right-hand side an IndexError is raised.
        If the affine vectors are not stored in the class, the method which computes them is called, provided that an
        instance of FomProblemUnsteady is given as input

        :param _q: index of the desired affine component for the right-hand side vector
        :type _q: int
        :param _fom_problem: fom problem at hand
        :type: FomProblemUnsteady
        :return: desired FOM affine component of the right-hand side vector
        :rtype: numpy.ndarray
        """

        if _fom_problem is not None:
            try:
                assert 'final_time' in _fom_problem.M_fom_specifics.keys()
            except AssertionError:
                logger.critical("The FomProblem must be unsteady to get the affine components for the right-hand side "
                                "vector f")
                raise TypeError

        return super().get_affine_vector(_q, _fom_problem=_fom_problem)

    def get_affine_vector_reduced(self, _q, _fom_problem=None):
        """Getter method, which returns the affine component of index _q among the FOM affine components of the
        right-hand side, evaluated over a the reduced number of timesteps at which the time integration of the RB
        problem has been performed. If _q exceeds the number of affine components for the right-hand side an IndexError
        is raised. If the affine vectors are not stored in the class, the method which computes them is called,
        provided that an instance of FomProblemUnsteady is given as input

        :param _q: index of the desired affine component for the right-hand side vector, evaluated over the 'reduced'
            timesteps
        :type _q: int
        :param _fom_problem: fom problem at hand
        :type: FomProblemUnsteady or NoneType
        :return: desired FOM affine component of the right-hand side vector, evaluated over the reduced timesteps
        :rtype: numpy.ndarray
        """

        if _fom_problem is not None:
            try:
                assert 'final_time' in _fom_problem.M_fom_specifics.keys()
            except AssertionError:
                logger.critical("The FomProblem must be unsteady to get the affine components for the right-hand side "
                                "vector f")
                raise TypeError

        if _fom_problem is not None and not self.check_set_fom_arrays_reduced():
            self.import_fom_affine_arrays(_fom_problem, operators={'f_reduced'})

        return self.M_feAffineFReducedq[_q]

    def get_affine_initial_condition(self, _fom_problem=None):
        """Getter method for the FOM initial condition

        :param _fom_problem: fom problem at hand
        :type: FomProblemUnsteady or NoneType
        :return: FOM initial condition
        :rtype: numpy.ndarray
        """

        if _fom_problem is not None:
            try:
                assert 'final_time' in _fom_problem.M_fom_specifics.keys()
            except AssertionError:
                logger.critical("The FomProblem must be unsteady to get the FOM initial condition")
                raise TypeError

        if _fom_problem is not None and not self.check_set_fom_arrays_reduced():
            self.import_fom_affine_arrays(_fom_problem, operators={'u0'})

        return self.M_feInitialCondition

    def import_affine_matrices_A(self, _input_file):
        """Method which allows to import the FOM affine matrices for the stiffness matrix from file

        :param _input_file: partial path to the files which contain the affine matrices for the stiffness matrix.
            Partial since the final file name is actually retrieved from '_input_file' by adding the operator name and
            the index of the affine component which is currently imported
        :type _input_file: str
        """

        super().import_affine_matrices(_input_file)

        return

    def import_affine_matrices_M(self, _input_file):
        """Method which allows to import the FOM affine matrices for the mass matrix from file

        :param _input_file: partial path to the files which contain the affine matrices for the mass matrix.
            Partial since the final file name is actually retrieved from '_input_file' by adding the operator name and
            the index of the affine component which is currently imported
        :type _input_file: str
        """

        self.M_feAffineMq = []
        Qm = self.qm

        for iQm in range(Qm):
            try:
                self.M_feAffineMq.append(np.loadtxt(_input_file + str(0) + '.txt'))  # importing matrix in sparse format
            except (IOError, OSError, FileNotFoundError) as e:
                logger.error(f"Error {e}: impossible to load the affine matrices M")
                break

        #  if the matrix indices start from 1, we rescale them
        if np.min(self.M_feAffineMq[0][:, :2]) > 0:
            self.M_feAffineMq[0][:, :2] = self.M_feAffineMq[0][:, :2] - 1

        return

    def import_fom_affine_arrays(self, _fom_problem, operators=None):
        """Method which allows to 'import' the FOM affine arrays for the operators passed in input. If no operator is
        passed, the FOM affine arrays are constructed for the stiffness matrix, the mass matrix and the right-hand side
        vector. The FOM affine arrays are constructed via a call to the external engine that has been initialized,
        if any, passing through a wrapping function of the FomProblemUnsteady class. If '_fom_problem' is not an
        instance of FomProblemUnsteady class, the method raises a TypeError

        :param _fom_problem: fom problem at hand
        :type: FomProblemUnsteady
        :param operators: operators for which the FOM affine components have to be constructed. Admissible values are
            'A' for the stiffness matrix, 'M' for the mass matrix, 'f' for the right-hand side vector evaluated over
            the FOM timesteps, 'f_reduced' for the right-hand side vector evaluated over the sub-sampled RB timesteps
            and 'f_space' for the spacial part of the forcing term, provided that it can be written as
            f(x,t) = f1(x) * f2(t). Defaults to None.
        :type operators: set or NoneType
        """

        try:
            assert 'final_time' in _fom_problem.M_fom_specifics.keys()
        except AssertionError:
            logger.critical(
                f"The FomProblem must be unsteady to get the fom affine components fo operators {operators}")
            raise TypeError

        if operators is None:
            operators = {'A', 'f', 'M', 'u0'}

        if 'A' in operators:
            super().import_fom_affine_arrays(_fom_problem, operators={'A'})

        if 'f' in operators:
            super().import_fom_affine_arrays(_fom_problem, operators={'f'})

        if 'M' in operators:
            if len(self.M_feAffineMq) < self.qm:
                logger.info("I am importing affine arrays for operator M ")

                MM = _fom_problem.retrieve_fom_affine_components('M', self.qm - len(self.M_feAffineMq))
                starting_Qm = len(self.M_feAffineMq)

                for iQm in range(self.qm - starting_Qm):
                    self.M_feAffineMq.append(np.array(MM['M' + str(iQm)]))

        if 'f_reduced' in operators:
            assert self.M_Nt_Reduced is not None and self.M_Nt_Reduced
            if len(self.M_feAffineFReducedq) < self.qf:
                logger.info("I am importing affine arrays for operator f_reduced ")

                ff = _fom_problem.retrieve_fom_affine_components('f', self.qf - len(self.M_feAffineFReducedq),
                                                                 self.M_Nt_Reduced)
                starting_Qf = len(self.M_feAffineFReducedq)

                for iQf in range(self.qf - starting_Qf):
                    self.M_feAffineFReducedq.append(np.array(ff['f' + str(iQf)]))

        if 'f_space' in operators:
            if len(self.M_feAffineFSpaceq) < self.qf:
                logger.info("I am importing affine arrays for operator f_space")

                ff = _fom_problem.retrieve_fom_affine_components('f_space', self.qf - len(self.M_feAffineFSpaceq))
                starting_Qf = len(self.M_feAffineFSpaceq)

                for iQf in range(self.qf - starting_Qf):
                    self.M_feAffineFSpaceq.append(np.array(ff['f_space' + str(iQf)]))

        if 'u0' in operators:
            self.M_feInitialCondition = _fom_problem.get_initial_condition(np.zeros(3))

        return

    def set_affine_M(self, _feAffineMq):
        """Method to set the FOM affine matrices for the mass matrix

        :param _feAffineMq: list containing all the FOM affine matrices for the mass matrix
        :type _feAffineMq: list[numpy.ndarray]
        """

        self.M_feAffineMq = _feAffineMq

        return

    def set_affine_f_space(self, _feAffineFSpaceq):
        """Method to set the FEM affine components for the spatial part of the right-hand side forcing term

        :param _feAffineFSpaceq: list containing all the FEM affine components for the spatial part of the right-hand
           side forcing term
        :type _feAffineFSpaceq: list[numpy.ndarray]
        """

        self.M_feAffineFSpaceq = _feAffineFSpaceq

        return

    def set_initial_condition(self, _initial_condition):
        """ Setter method for the FOM initial condition

        :param _initial_condition: FOM initial condition to be set
        :type _initial_condition: numpy.ndarray
        """

        self.M_feInitialCondition = _initial_condition

        return

    def expand_f_space_affine_components(self, _fom_problem):
        """Method to expand the spatial affine components of the right-hand side vector in time

        :param _fom_problem: fom problem at hand
        :type: FomProblemUnsteady
        """
        ff = _fom_problem.expand_f_space_affine_components(self.M_feAffineFSpaceq)
        for iQf in range(self.qf):
            self.M_feAffineFq.append(np.array(ff['f' + str(iQf)])[:, 0])
        return

    def check_set_fom_arrays(self):
        """Method which checks whether the FOM affine arrays are stored in the class instance, either by constructing
        them or by importing them from file. Regarding the right-hand side, the 'non-reduced' affine components are
        checked

        :return: True if the FOM affine arrays are stored in the class instance, False otherwise
        :rtype: bool
        """
        return super().check_set_fom_arrays() and len(self.M_feAffineMq) == self.qm

    def check_set_fom_arrays_reduced(self):
        """Method which checks whether the FOM affine arrays are stored in the class instance, either by constructing
       them or by importing them from file. Regarding the right-hand side, the 'reduced' affine components are
       checked

       :return: True if the FOM affine arrays are stored in the class instance, False otherwise
       :rtype: bool
       """
        return len(self.M_feAffineAq) == self.qa and len(self.M_feAffineFReducedq) == self.qf and \
               len(self.M_feAffineMq) == self.qm

    def check_length_fom_f_reduced(self):
        """Function which checks if the components of f_reduced have all the proper
        length 'L=((self.M_Nh+1)**2)*(self.M_Nt_Reduced+1)'

        :return: True if all elements of f_reduced have the proper length, False otherwise
        :rtype: bool
        """

        if len(self.M_feAffineFReducedq) == 0:
            logger.warning("No f_reduced FOM affine affine component is stored in AffineDecompositionUnsteady class")
            return False
        else:
            L = ((self.M_Nh+ 1) ** 2) * (self.M_Nt_Reduced + 1)
            return all([len(elem) == L for elem in self.M_feAffineFReducedq])

    def reset_rb_approximation(self):
        """Method to reset the RB affine components for the stiffness matrix, the mass matrix and the right-hand side
        vector (both in reduced and non-reduced form) by redefining the lists containing such elements as empty lists
        """

        super().reset_rb_approximation()
        self.M_rbAffineMq = []
        self.M_rbAffineFReducedq = []

        return

    def build_rb_affine_decompositions(self, _basis, _fom_problem, operators=None):
        """Method which constructs the RB affine components for the operators passed in input, suitably multiplying the
        considered affine component by the matrix encoding the reduced basis; in particular vectors are pre-multiplied
        by the transpose of such matrix, while matrices are pre-multiplied by the transpose basis matrix and
        post-multiplied by the basis matrix itself. If no operator is passed, the RB affine arrays are constructed for
        the stiffness matrix, the mass matrix and the right-hand side vector. If the FOM affine arrays are not yet
        stored in the class, they are constructed via a call to
        :func:`~affine_decomposition_unsteady.AffineDecompositionUnsteady.import_fom_affine_arrays`.
        If '_fom_problem' is not an instance of :class:`~fom_problem_unsteady.FomProblemUnsteady`, the method raises a
        TypeError

        :param _basis: matrix encoding the reduced basis
        :type _basis: numpy.ndarray
        :param _fom_problem: FOM problem at hand
        :type _fom_problem: FomProblemUnsteady
        :param operators: operators for which the RB affine components have to be constructed. Admissible values are
            'A' for the stiffness matrix, 'M' for the mass matrix, 'f' for the right-hand side vector evaluated over
            the FOM timesteps, 'f_reduced' for the right-hand side vector evaluated over the sub-sampled RB timesteps
            and 'f_space' for the spacial part of the forcing term, provided that it can be written as
            f(x,t) = f1(x) * f2(t). Defaults to None.
        :type operators: set or NoneType
        """

        try:
            assert 'final_time' in _fom_problem.M_fom_specifics.keys()
        except AssertionError:
            logger.critical(
                f"The FomProblem must be unsteady to get the fom affine components for operators {operators}")
            raise TypeError

        self.M_Nh= _basis.shape[0]
        self.M_N = _basis.shape[1]

        if operators is None:
            operators = {'A', 'f', 'M', 'u0'}

        if 'f' in operators and not self.check_set_fom_arrays():
            self.import_fom_affine_arrays(_fom_problem, operators=operators)
        elif 'f_reduced' in operators and not self.check_set_fom_arrays_reduced():
            self.import_fom_affine_arrays(_fom_problem, operators=operators)
        elif 'f_reduced' in operators and not self.check_length_fom_f_reduced():
            self.M_feAffineFReducedq = []
            self.import_fom_affine_arrays(_fom_problem, operators={'f_reduced'})
        else:
            logger.debug("Already set the FOM affine arrays")

        if arr_utils.is_empty(self.M_feInitialCondition):
            self.import_fom_affine_arrays(_fom_problem, operators={'u0'})

        Qa = self.qa
        Qf = self.qf
        Qm = self.qm

        if 'f' in operators:
            for iQf in range(Qf):
                self.M_rbAffineFq.append(np.zeros(self.M_N * (self.M_Nt + 1)))
                for iT in range(self.M_Nt + 1):
                    self.M_rbAffineFq[iQf][iT * self.M_N:(iT + 1) * self.M_N] = _basis.T.dot(self.M_feAffineFq[iQf]
                                                                                             [iT * self.M_Nh:(iT + 1) * self.M_Nh])

        if 'f_reduced' in operators:
            for iQf in range(Qf):
                self.M_rbAffineFReducedq.append(np.zeros(self.M_N * (self.M_Nt_Reduced + 1)))
                for iT in range(self.M_Nt_Reduced + 1):
                    self.M_rbAffineFReducedq[iQf][iT * self.M_N:(iT + 1) * self.M_N] = _basis.T.dot(
                        self.M_feAffineFReducedq[iQf]
                        [iT * self.M_Nh:(iT + 1) * self.M_Nh])

        if 'f_space' in operators:
            for iQf in range(Qf):
                self.M_rbAffineFSpaceq.append(_basis.T.dot(self.M_feAffineFSpaceq[iQf]))

        if 'A' in operators:
            for iQa in range(Qa):
                Av = arr_utils.sparse_matrix_matrix_mul(self.M_feAffineAq[iQa], _basis)
                self.M_rbAffineAq.append(np.zeros((self.M_N, self.M_N)))
                self.M_rbAffineAq[iQa] = _basis.T.dot(Av)

        if 'M' in operators:
            for iQm in range(Qm):
                Mv = arr_utils.sparse_matrix_matrix_mul(self.M_feAffineMq[iQm], _basis)
                self.M_rbAffineMq.append(np.zeros((self.M_N, self.M_N)))
                self.M_rbAffineMq[iQm] = _basis.T.dot(Mv)

        if 'u0' in operators:
            self.build_rb_initial_condition(_basis, _fom_problem)

        logger.info('Finished to build the unsteady RB affine arrays')

        return

    def build_rb_nonlinear_term(self, u, _basis, _fom_problem):
        """Method that builds the reduced non-linear term based on the current value of the solution (passed as first
        argument to the function). If the problem is linear, it simply return an empty array of the right dimension.

        :param u: current reduced solution
        :type u: numpy.ndarray
        :param _basis: matrix encoding the reduced basis
        :type _basis: numpy.ndarray
        :param _fom_problem: unsteady FOM problem at hand
        :type _fom_problem: FomProblemUnsteady
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

        :param u: current reduced solution
        :type u: numpy.ndarray
        :param _basis: matrix encoding the reduced basis
        :type _basis: numpy.ndarray
        :param _fom_problem: unsteady FOM problem at hand
        :type _fom_problem: FomProblemUnsteady
        :return reduced non-linear jacobian (matrix of zeros if the problem is linear)
        :rtype: numpy.ndarray
        """

        if _fom_problem.is_linear():
            return [np.zeros((u.shape[0], u.shape[0]))] * u.shape[1]

        else:
            u_fom = _basis.dot(u)
            nl_jac = self.build_nonlinear_jacobian(u_fom, _fom_problem)
            rb_nl_jac = []
            for ind_t in range(u.shape[1]):
                tmp = arr_utils.sparse_matrix_matrix_mul(nl_jac[ind_t], _basis)
                rb_nl_jac[ind_t] = _basis.T.dot(tmp)
            return rb_nl_jac

    def build_rb_initial_condition(self, _basis, _fom_problem):
        """Method that allows to project the initial condition on the reduced subspace

        :param _basis: matrix encoding the reduced basis
        :type _basis: numpy.ndarray
        :param _fom_problem: unsteady FOM problem at hand
        :type _fom_problem: FomProblemUnsteady
        """

        try:
            assert 'final_time' in _fom_problem.M_fom_specifics.keys()
        except AssertionError:
            logger.critical(f"The FomProblem must be unsteady to get the reduced initial condition")
            raise TypeError

        if arr_utils.is_empty(self.M_feInitialCondition):
            self.import_fom_affine_arrays(_fom_problem, operators={'u0'})

        self.M_rbInitialCondition = _basis.T.dot(self.M_feInitialCondition)

        return

    def check_set_rb_arrays(self):
        """Method which checks whether the RB affine arrays are stored in the class instance, either by constructing
        them or by importing them from file. Regarding the right-hand side, the 'non-reduced' RB affine components are
        checked

        :return: True if the RB affine arrays are stored in the class instance, False otherwise
        :rtype: bool
        """
        return super().check_set_rb_arrays() and len(self.M_rbAffineMq) == self.qm

    def check_set_rb_arrays_reduced(self):
        """Method which checks whether the RB affine arrays are stored in the class instance, either by constructing
        them or by importing them from file. Regarding the right-hand side, the 'reduced' RB affine components are
        checked

        :return: True if the RB affine arrays are stored in the class instance, False otherwise
        :rtype: bool
        """
        return len(self.M_rbAffineAq) == self.qa and len(self.M_rbAffineFReducedq) == self.qf and \
               len(self.M_rbAffineMq) == self.qm

    def get_rb_affine_matrix_A(self, _qa):
        """Getter method, which returns the affine component of index _qa among the FOM affine components of the
        stiffness matrix, projected over the RB-space. If _qa exceeds the number of affine components for the stiffness
        matrix an IndexError is raised.

        :param _qa: index of the desired affine component for the stiffness matrix
        :type _qa: int
        :return: desired RB affine component of the stiffness matrix
        :rtype: int
        """
        return super().get_rb_affine_matrix(_qa)

    def get_rb_affine_matrix_M(self, _qm=0):
        """Getter method, which returns the affine component of index _qm among the FOM affine components of the
        mass matrix, projected over the RB-space. If _qm exceeds the number of affine components for the mass matrix an
        IndexError is raised.

        :param _qm: index of the desired affine component for the mass matrix. Defaults to 0.
        :type _qm: int
        :return: desired RB affine component of the mass matrix
        :rtype: int
        """
        if _qm < self.M_affineDecomposition.qm:
            return self.M_rbAffineMq[_qm]
        else:
            raise IndexError

    def get_rb_affine_vector(self, _qf, timestep=None):
        """Getter method, which returns the affine component of index _qf among the RB affine components of the
        right-hand side. If _qf exceeds the number of affine components for the right-hand side an IndexError is raised.
        The evaluation timesteps are assumed to be the ones of the FOM problem, i.e. the 'non-reduced' ones.
        Additionally, the desired RB affine component can be evaluated:

        * at all the timesteps, if 'timestep' is passed as None (default value)
        * at a specific timestep, if 'timestep' is of type int
        * at some specific timesteps if 'timestep' is of type list[int]

        :param _qf: index of the desired affine component for the right-hand side vector
        :type _qf: int
        :param timestep: timestep or timesteps at which the evaluation of the RB affine vector is desired. If None, the
            evaluation is performed at all the timesteps. Defaults to None
        :type timestep: int, list[int] or NoneType
        :return: desired RB affine component of the right-hand side vector
        :rtype: int
        """

        if timestep is None:
            return self.M_rbAffineFq[_qf]
        elif type(timestep) is int:
            return self.M_rbAffineFq[_qf][timestep * self.M_N:(timestep + 1) * self.M_N]
        elif type(timestep) is list and type(timestep[0]) is int:
            result = np.zeros((len(timestep), self.M_N))
            for cnt in range(len(timestep)):
                result[cnt, :] = self.M_rbAffineFq[_qf][timestep[cnt] * self.M_N:(timestep[cnt] + 1) * self.M_N]
            return result
        else:
            logger.critical("The argument 'timestep' must be either of type 'int' or 'list[int]' or NoneType!")
            raise ValueError

    def get_rb_affine_vector_reduced(self, _qf=0, timestep=None):
        """Getter method, which returns the affine component of index _qf among the RB affine components of the
        right-hand side. If _qf exceeds the number of affine components for the right-hand side an IndexError is raised.
        The evaluation timesteps are assumed to be the ones of the RB problem, i.e. the 'reduced' ones.
        Additionally, the desired RB affine component can be evaluated:

        * at all the timesteps, if 'timestep' is passed as None (default value)
        * at a specific timestep, if 'timestep' is of type int
        * at some specific timesteps if 'timestep' is of type list[int]

        :param _qf: index of the desired affine component for the right-hand side vector
        :type _qf: int
        :param timestep: timestep or timesteps at which the evaluation of the RB affine vector is desired. If None, the
            evaluation is performed at all the timesteps. Defaults to None
        :type timestep: int, list[int] or NoneType
        :return: desired RB affine component of the right-hand side vector
        :rtype: int
        """

        if timestep is None:
            return self.M_rbAffineFReducedq[_qf]
        elif type(timestep) is int:
            return self.M_rbAffineFReducedq[_qf][timestep * self.M_N:(timestep + 1) * self.M_N]
        elif type(timestep) is list and type(timestep[0]) is int:
            result = np.zeros((len(timestep), self.M_N))
            for cnt in range(len(timestep)):
                result[cnt, :] = self.M_rbAffineFReducedq[_qf][timestep[cnt] * self.M_N:(timestep[cnt] + 1) * self.M_N]
            return result
        else:
            logger.critical("The argument 'timestep' must be either of type 'int' or 'list[int]' or NoneType!")
            raise ValueError

    def get_rb_affine_vector_space(self, _qf=0):
        """Getter method, which returns the affine component of index _qf among the RB affine components of the
        spatial part of the right-hand side vector, under the assumption that it can be written as f(x,t) = f1(x) * f2(t),
        projected over the RB-space. If _qf exceeds the number of affine components for the right-hand side vector an
        IndexError is raised.

        :param _qf: index of the desired affine component for the spatial part of the right-hand side vector.
            Defaults to 0.
        :type _qf: int
        :return: desired RB affine component of the spatial part of the right-hand side vector
        :rtype: int
        """
        if _qf < self.M_affineDecomposition.qf:
            return self.M_rbAffineFSpaceq[_qf]
        else:
            raise IndexError

    def get_rb_initial_condition(self):
        """Getter method, which returns the space-reduced initial condition

        :return: space-reduced initial condition
        :rtype: numpy.ndarray
        """

        if arr_utils.is_empty(self.M_rbInitialCondition):
            raise ValueError("Impossible to return the reduced initial condition, "
                             "as no reduced initial condition has been set!")

        return self.M_rbInitialCondition

    def import_rb_affine_components(self, _input_file, operators=None):
        """Method which allows to import the RB affine components for the operators passed in input from file. If no
        operator is passed, the RB affine arrays are constructed for the stiffness matrix, the mass matrix and the
        right-hand side vector. If the import of the affine components for a certain operator has failed, the operator
        name is stored in a set which is finally returned by the method itself

        :param _input_file: partial path to the files which contain the RB affine vectors for the right-hand side vector
            Partial since the final file name is actually retrieved from '_input_file' by adding the operator name and
            the index of the affine component which is currently imported
        :type _input_file: str
        :param operators: operators for which the RB affine components have to be imported. Admissible values are
            'A' for the stiffness matrix, 'M' for the mass matrix, 'f' for the right-hand side vector evaluated over
            the FOM timesteps, 'f_reduced' for the right-hand side vector evaluated over the sub-sampled RB timesteps
            and 'f_space' for the spacial part of the forcing term, provided that it can be written as
            f(x,t) = f1(x) * f2(t). Defaults to None.
        :type operators: set or NoneType
        :return: set containing the name of the operators whose import of the RB affine components has failed
        :rtype: set or NoneType
        """

        if operators is None:
            operators = {'A', 'f', 'M', 'u0'}

        import_failures_set = set()
        if 'A' in operators:
            import_failures_set = super().import_rb_affine_components(_input_file, operators={'A'})

        if 'f' in operators:
            import_failures_set_f = super().import_rb_affine_components(_input_file, operators={'f'})
            import_failures_set.union(import_failures_set_f)

        if 'M' in operators:
            Qm = self.qm
            for iQm in range(Qm):
                try:
                    self.M_rbAffineMq.append(np.loadtxt(_input_file + '_M' + str(iQm) + '.txt'))
                except (IOError, OSError, FileNotFoundError) as e:
                    logger.error(f"Error {e}. Impossible to open the desired file for matrix M{str(iQm)}.")
                    import_failures_set.add('M')
                    break

        if 'f_reduced' in operators:
            Qf = self.qf
            for iQf in range(Qf):
                try:
                    self.M_rbAffineFReducedq.append(np.loadtxt(_input_file + '_f_reduced' + str(iQf) +
                                                               '_timesteps_dilation_' + str(int(self.M_Nt /
                                                                                                self.M_Nt_Reduced))
                                                               + '.txt'))
                except (IOError, OSError, FileNotFoundError) as e:
                    logger.error(f"Error {e}. Impossible to open the desired file for vector f{str(iQf)}.")
                    import_failures_set.add('f_reduced')
                    break

        if 'f_space' in operators:
            Qf = self.qf
            for iQf in range(Qf):
                try:
                    self.M_rbAffineFSpaceq.append(np.loadtxt(_input_file + '_f_space' + str(iQf) + '.txt'))
                except (IOError, OSError, FileNotFoundError) as e:
                    logger.error(f"Error {e}. Impossible to open the desired file for vector f{str(iQf)}.")
                    import_failures_set.add('f_space')
                    break

        if 'u0' in operators:
            try:
                self.M_rbInitialCondition = np.loadtxt(_input_file + '_u0' + '.txt')
            except (IOError, OSError, FileNotFoundError) as e:
                logger.error(f"Error {e}. Impossible to open the desired file for u0.")
                import_failures_set.add('u0')

        return import_failures_set

    def set_affine_rb_M(self, _rbAffineMq):
        """Method to set the RB affine matrices for the mass matrix

        :param _rbAffineMq: list containing all the RB affine matrices for the mass matrix
        :type _rbAffineMq: list[numpy.ndarray]
        """

        self.M_rbAffineMq = _rbAffineMq

        return

    def set_affine_rb_f_reduced(self, _rbAffineFReducedq):
        """Method to set the RB affine matrices for the reduced right-hand side vector

        :param _rbAffineFReducedq: list containing all the RB affine components for the reduced right-hand side vector
        :type _rbAffineFReducedq: list[numpy.ndarray]
        """

        self.M_rbAffineFReducedq = _rbAffineFReducedq

        return

    def set_affine_rb_f_space(self, _rbAffineFSpaceq):
        """Method to set the RB affine components for the spatial part of the right-hand side forcing term

        :param _rbAffineFSpaceq: list containing all the RB affine components for the spatial part of the right-hand side
           orcing term
        :type _rbAffineFSpaceq: list[numpy.ndarray]
        """

        self.M_rbAffineFSpaceq = _rbAffineFSpaceq

        return

    def set_rb_initial_condition(self, _rbInitialCondition):
        """Setter method for the reduced initial condition

        :param _rbInitialCondition: reduced initial condition
        :type _rbInitialCondition: numpy.ndarray
        """

        self.M_rbInitialCondition = _rbInitialCondition

        return

    def save_rb_affine_decomposition(self, _file_name, operators=None):
        """Method which saves the RB affine components of the operators passed in input to text files, whose path has
        been specified by the input argument '_file_name'. If no operator is passed, the RB affine arrays are saved for
        the stiffness matrix, the mass matrix and the right-hand side vector. The final file name is constructed by
        adding to the input file name the operator name and the index of the affine components which is currently saved.

        :param _file_name: partial path to the files where the RB affine components for the desired operators are saved
            Partial since the final file name is completed with the operator name and the index of the affine component
            which is currently saved
        :type _file_name: str
        :param operators: operators for which the RB affine components have to be saved. Admissible values are
            'A' for the stiffness matrix, 'M' for the mass matrix, 'f' for the right-hand side vector evaluated over
            the FOM timesteps, 'f_reduced' for the right-hand side vector evaluated over the sub-sampled RB timesteps
            and 'f_space' for the spacial part of the forcing term, provided that it can be written as
            f(x,t) = f1(x) * f2(t). Defaults to None.
        :type operators: set or NoneType
        """

        if operators is None:
            operators = {'A', 'f', 'M', 'u0'}

        if 'A' in operators:
            super().save_rb_affine_decomposition(_file_name, operators={'A'})
        if 'f' in operators:
            super().save_rb_affine_decomposition(_file_name, operators={'f'})

        if 'M' in operators:
            Qm = self.qm
            for iQm in range(Qm):
                path = _file_name + '_M' + str(iQm) + '.txt'
                arr_utils.save_array(self.M_rbAffineMq[iQm], path)

        if 'f_reduced' in operators:
            Qf = self.qf
            for iQf in range(Qf):
                path = _file_name + '_f_reduced' + str(iQf) + '_timesteps_dilation_' + str(
                    int(self.M_Nt / self.M_Nt_Reduced)) + '.txt'
                arr_utils.save_array(self.M_rbAffineFReducedq[iQf], path)

        if 'f_space' in operators:
            Qf = self.qf
            for iQf in range(Qf):
                path = _file_name + '_f_space' + str(iQf) + '.txt'
                arr_utils.save_array(self.M_rbAffineFSpaceq[iQf], path)

        if 'u0' in operators:
            path = _file_name + '_u0' + '.txt'
            arr_utils.save_array(self.M_rbInitialCondition, path)

        return

    def get_mass_stiffness_matrices(self, _fom_problem, withMass=True):
        """Method which allows to construct and store in the class the stiffness and the mass matrices; such matrices
        are intended to be used for the computation of the L2 and H10 norms, thus the stiffness matrix is computed by
        assuming all the parameters to be equal to 1. The construction is performed via the external engine, passing
        through an intermediate wrapping function in the FomProblem class. Additionally, it is possible to just compute
        the stiffness matrix, by setting the input flag 'withMass' to False. If the mass matrix is already stored in the
        class, it will not be computed again. If the stiffness matrix is already stored in the class it will not be
        computed again

        :param _fom_problem: fom problem at hand
        :type _fom_problem: FomProblem
        :param withMass: if True also the mass matrix is computed. Defaults to True
        :type withMass: bool
        """

        if withMass:
            if len(self.M_feAffineMq):
                self.M_mass = np.sum(self.M_feAffineMq, axis=0)
            if len(self.M_feAffineAq):
                self.M_stiffness = np.sum(self.M_feAffineAq, axis=0)

            if self.M_mass is not None and self.M_stiffness is None:
                self.M_stiffness = _fom_problem.assemble_fom_matrix(1, withMass=False)
            elif self.M_mass is None and self.M_stiffness is not None:
                _, self.M_mass = _fom_problem.assemble_fom_matrix(1, withMass=True)
            elif self.M_stiffness is None and self.M_mass is None:
                self.M_stiffness, self.M_mass = _fom_problem.assemble_fom_matrix(1, withMass=True)
            else:
                pass

        else:
            if len(self.M_feAffineAq):
                self.M_stiffness = np.sum(self.M_feAffineAq, axis=0)
            else:
                self.M_stiffness = _fom_problem.assemble_fom_matrix(1, withMass=False)

        return


__all__ = [
    "AffineDecompositionHandlerUnsteady",
    "AffineDecompositionUnsteady"
]
