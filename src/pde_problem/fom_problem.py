#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 12:02:21 2018
@author: Niccolo' Dal Santo
@email : niccolo.dalsanto@epfl.ch
"""

import sys
import os
sys.path.insert(0, os.path.normpath('../..'))
sys.path.insert(0, os.path.normpath('../'))

import numpy as np
import random
import os

import logging.config
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def default_theta_function(_param, _q):
    """ Function which is intended to return the default parameter-dependent function corresponding to index q.
    Since all parameter-dependent affine-decomposable problems are characterized by different parameter-dependent
    functions, the function will actually just raise a SystemError, inviting the user to define proper functions
    for the problem at hand.

    :param _param: value of the parameters
    :type _param: list or numpy.ndarray
    :param _q: index of the desired parameter-dependent function
    :type _q: int
    """
    assert _q >= 0

    raise Exception("You are using the default theta function, please provide specific ones for your problem!")

    pass


def default_full_theta_function(*args):
    """ Function which is intended to return the default parameter-dependent functions.
    Since all parameter-dependent affine-decomposable problems are characterized by different parameter-dependent
    functions, the function will actually just raise a SystemError, inviting the user to define proper functions
    for the problem at hand.

    :param _param: value of the parameters
    :type _param: list or numpy.ndarray
    """

    raise Exception("You are using the default full theta function, please provide specific ones for your problem!")

    pass


class FomProblem:
    """Abstract class defining a generic parameter-dependent steady FOM problem; since it does not refer to any specific
    FOM problem, such class cannot be instantiated. Indeed the main method to define the parameter-dependent functions
    flags an error whenever invoked. All the specific FOM problems are coded as children of this virtual parent class.
    """

    def __init__(self, _parameter_handler, _external_engine=None, _fom_specifics=None):
        """Initializing the Full Order Model (FOM) parameter-dependent problem, equipping it with the parameter handler
        and, if provided, the external engine to solve the FOM problem (either Matlab or C++) and the specifics of the
        problem itself.

        :param _parameter_handler: ParameterHandler object, which performs the task of handling the parameters involved
            in the problem
        :type _parameter_handler: ParameterHandler
        :param _external_engine: engine which perform the computation of all the FOM-related quantities. In the context
            of this project it can be either a Matlab external engine or a C++ external engine.
            If None, no initialization of the external engine is made. Defaults to None.
        :type: _external_engine: ExternalEngine or NoneType
        :param _fom_specifics: dictionary containing all the specifics of the FOM problem at hand.
            If None, no initialization of the FOM specifics is made. Defaults to None.
        :type _fom_specifics: dict or NoneType
        """

        self.M_theta_a = default_theta_function
        self.M_theta_f = default_theta_function
        self.M_full_theta_a = default_full_theta_function
        self.M_full_theta_f = default_full_theta_function
        self.M_current_parameter = np.zeros(0)

        if _external_engine is not None and _fom_specifics is not None:
            self.configure_fom(_external_engine, _fom_specifics)
        else:
            if _external_engine is not None:
                self.M_external_engine = _external_engine
                self.M_fom_specifics = None
            elif _fom_specifics is not None:
                self.M_fom_specifics = _fom_specifics
                self.M_external_engine = None
            else:
                self.M_fom_specifics = None
                self.M_external_engine = None
            self.M_configured_fom = False

        self.M_theta_nl_term = default_theta_function
        self.M_theta_nl_jac = default_theta_function
        self.M_full_theta_nl_term = default_full_theta_function
        self.M_full_theta_nl_jac = default_full_theta_function

        self.define_theta_functions()
        self.M_parameter_handler = _parameter_handler

        return

    def get_theta_a(self, _param, _q):
        """Method which returns the value of the parameter-dependent function with index _q associated to the
        stiffness matrix A if the value of the parameters is stored in `_param`

        :param _param: value of the parameters
        :type _param: list or numpy.ndarray
        :param _q: index of the target parameter-dependent function. Notice: **_q >= 0**
        :type _q: int
        :return: value of the desired parameter-dependent function
        :rtype: float
        """
        assert _q >= 0
        return self.M_theta_a(_param, _q)

    def get_theta_f(self, _param, _q):
        """Method which returns the value of the parameter-dependent function with index _q associated to the
        right-hand side vector f if the value of the parameters is stored in `_param`

        :param _param: value of the parameters
        :type _param: list or numpy.ndarray
        :param _q: index of the target parameter-dependent function. Notice: **_q >= 0**
        :type _q: int
        :return: value of the desired parameter-dependent function
        :rtype: float
        """
        assert _q >= 0
        return self.M_theta_f(_param, _q)

    def get_full_theta_a(self, _param):
        """Method which returns the value of all the parameter-dependent functions associated to the
        stiffness matrix A if the value of the parameters is stored in _param

        :param _param: value of the parameters
        :type _param: list or numpy.ndarray
        :return: value of the desired parameter-dependent functions
        :rtype: float
        """
        return self.M_full_theta_a(_param)

    def get_full_theta_f(self, _param):
        """Method which returns the value of the parameter-dependent function associated to the
        right-hand side vector f if the value of the parameters is stored in _param

        :param _param: value of the parameters
        :type _param: list or numpy.ndarray
        :return: value of the desired parameter-dependent functions
        :rtype: float
        """
        return self.M_full_theta_f(_param)

    def get_full_theta_nl_term(self, _u, _param):
        """
        MODIFY
        """
        return self.M_full_theta_nl_term(_u, _param)

    def get_full_theta_nl_jac(self, _u, _param):
        """
        MODIFY
        """
        return self.M_full_theta_nl_jac(_u, _param)

    @staticmethod
    def define_theta_functions():
        """Method to define the parameter-dependent functions characterizing the FOM problem. Being this class
        representative of a virtual generic parameter-dependent FOM problem, it just raises an error if called;
        indeed no parameter-dependent functions are defined here, since they all must be defined in the specific
        FOM problems, which all configure as children of this parent class"""

        raise Exception("You should define the theta function specific for your problem in the inherited class.")

    def compute_theta_functions(self, _params, _Qa, _Qf):
        """Method which allows to evaluate the scalar parameter-dependent functions arising from the affine
        decomposition of the stiffness matrix and of the right-hand side vector of the FOM problem at hand, given a
        list of parameter values. The shape of such functions depends uniquely on the expression of the FOM problem
        at hand.

        :param _params: values of the parameters
        :type _params: numpy.ndarray
        :param _Qa: number of affine components for the stiffness matrix A
        :type _Qa: int
        :param _Qf: number of affine components for the right-hand side vector f
        :type _Qf: int
        :return: evaluation of the parameter-dependent functions of both A and f and the input parameter values
        :rtype: numpy.ndarray
        """

        theta_as = np.zeros((_params.shape[0], _Qa))
        theta_fs = np.zeros((_params.shape[0], _Qf))

        for iP in range(_params.shape[0]):
            mu = _params[iP, :]

            for iQa in range(_Qa):
                theta_q = self.get_theta_a(mu, iQa)
                theta_as[iP, iQa] = theta_q

            for iQf in range(_Qf):
                theta_q = self.get_theta_f(mu, iQf)
                theta_fs[iP, iQf] = theta_q

        return theta_as, theta_fs

    def configure_fom(self, _external_engine=None, _fom_specifics=None):
        """Method which allows to configure the FOM problem by setting the external engine and\or the problem specifics.
        It must be called if the FOM problem has been initialized missing either the external engine
        or the FOM specifics. It is called in the __init__ of the class.

        :param _external_engine: engine which perform the computation of all the FOM-related quantities. In the context
            of this project it can be either a Matlab external engine or a C++ external engine.
            If None, no initialization of the external engine is made. Defaults to None.
        :type: _external_engine: ExternalEngine or NoneType
        :param _fom_specifics: dictionary containing all the specifics of the FOM problem at hand.
            If None, no initialization of the FOM specifics is made. Defaults to None.
        :type _fom_specifics: dict or NoneType
        """
        if _external_engine is not None:
            self.M_external_engine = _external_engine
        if _fom_specifics is not None:
            self.M_fom_specifics = _fom_specifics

        if self.M_external_engine is not None and self.M_fom_specifics is not None:
            self.M_configured_fom = True
        return

    def check_configured_fom(self):
        """Method which checks whether the FOM problem has been correctly initialized and raises an error if that is not
        the case

        :return: True if the FOM problem has been correctly initialized; raises an error otherwise
        :rtype: bool or NoneType
        """

        if not self.M_configured_fom:
            raise Exception("The FOM problem has not been configured.")
        return True

    def update_fom_specifics(self, _fom_specifics_update):
        """Method which allows to update the dictionary containing the specifics of the FOM problem at hand. If the
        input updating dictionary has fields with keys that are not present in the original dictionary, then such keys
        are added to the original dictionary; conversely, if the updating dictionary has keys in common with the
        original dictionary, the corresponding values of the original dictionary will be updated with the ones of the
        input dictionary.

        :param _fom_specifics_update: updating dictionary
        :type _fom_specifics_update: dict
        """

        self.M_fom_specifics.update(_fom_specifics_update)

        return

    def is_linear(self):
        """Method that just returns True if the problem is linear and False otherwise, based on the content of the
        field named 'is_linear_problem' of the 'fom_specifics' dictionary class member

        :return True if the problem is linear and False otherwise
        :rtype: bool
        """

        return self.M_fom_specifics['is_linear_problem']

    def get_fom_dimension(self):
        """Method which returns the number of degrees of freedom of the FOM problem.

        ..note :: The FOM problem is assumed to be solved using a polynomial degree of 1. If a different degree is
        present in the _fom_specifics, then the method will raise a SystemError.

        :return: Number of DOFS of the FOM problem, provided that the polynomial degree is 1; else it raises an error
        :rtype: int or NoneType
        """

        if 'polynomial_degree' not in self.M_fom_specifics:
            logger.warning("FOM polynomial degree undefined.")
            return 0
        
        if self.M_fom_specifics['polynomial_degree'] == 'P1':
            return (self.M_fom_specifics['number_of_elements'] + 1) ** 2
        elif self.M_fom_specifics['polynomial_degree'] == 'P2':
            return (2*self.M_fom_specifics['number_of_elements'] + 1) ** 2
        else:
            raise ValueError("The FOM problem is assumed to be solved using either P1 or P2 finite elements.")

    def solve_fom_problem(self, _param, _get_nonlinear_terms=False):
        """ Method which provides the solution to the FOM problem at hand, given the value of the parameters stored
        in `_param`. If the FOM problem is not correctly configured, it flags an error.

        :param _param: value of the parameters
        :type _param: list or numpy.ndarray
        :param _get_nonlinear_terms: if True, also the non linear term and its jacobian are returned; Defaults to False
        :type _get_nonlinear_terms: bool
        :return: solution to the FOM problem corresponding to the parameter value _param. None if the FOM problem is
          not correctly initialized.
        :rtype: numpy.ndarray or NoneType
        """

        self.check_configured_fom()
        sol = self.M_external_engine.solve_parameter(_param, self.M_fom_specifics,
                                                     _get_nonlinear_terms=_get_nonlinear_terms)
        return sol

    def retrieve_fom_affine_components(self, _operator, _num_affine_components):
        """Method to compute the affine components associated to the operator _operator. If the problem is not
        affine-decomposable, the computation of such components is achieved by approximation using the DEIM-MDEIM
        algorithms; in such a case it may be useful to be given just the subset of the most relevant affine_components,
        which can be made via the argument _num_affine_components. If the FOM problem is not correctly initialized, it
        raises an error

        .. note:: The input argument **_operator** refers to the operator involved in the system to be solved to get
          the FOM solution f the problem at hand. Due to that, admissible values for such argument are 'A', referring to
          the stiffness matrix, 'f', referring to the right-hand side vector and 'M', referring to the mass matrix.

        .. note:: If the operator refers to a matrix, the affine components are returned in sparse COO format.

        :param _operator: operator of which the computation of the affine components is desired. Admissible values are
          either 'A', 'M' or 'f'
        :type _operator: str
        :param _num_affine_components: number of the desired affine components
        :type _num_affine_components: int
        :return: dictionary containing al the requested affine components associated to the desired operator
        :rtype: dict or NoneType
        """
        self.check_configured_fom()
        return self.M_external_engine.build_fom_affine_components(_operator, _num_affine_components,
                                                                  self.M_fom_specifics)

    def retrieve_fom_nonlinear_term(self, u, _elements=None, _indices=None,):
        """Method that builds the FOM non-linear term based on the current value of the solution (passed as first
        argument to the function). If the problem is linear, it simply return an empty array of the right dimension.

        :param u: current FOM solution
        :type u: numpy.ndarray
        :param _elements: set of elements over which evaluating the vector or None. Defaults to None.
        :type _elements: numpy.ndarray or NoneType
        :param _indices: set of dofs indices at which evaluating the vector or None. Defaults to None.
        :type _indices: numpy.ndarray or NoneType
        :return FOM non-linear term
        :rtype: numpy.ndarray
        """

        self.check_configured_fom()
        return self.M_external_engine.retrieve_fom_nonlinear_term(u, self.M_fom_specifics,
                                                                  _elements=_elements,
                                                                  _indices=_indices)

    def retrieve_fom_nonlinear_jacobian(self, u, _elements=None, _indices=None):
        """Method that builds the FOM non-linear jacobian based on the current value of the solution (passed as first
        argument to the function). If the problem is linear, it simply return an empty array of the right dimension.

        :param u: current FOM solution
        :type u: numpy.ndarray
        :param _elements: set of elements over which evaluating the matrix or None. Defaults to None.
        :type _elements: numpy.ndarray or NoneType
        :param _indices: set of dofs indices at which evaluating the matrix or None. Defaults to None.
        :type _indices: numpy.ndarray or NoneType
        :return FOM non-linear jacobian
        :rtype: numpy.ndarray
        """

        self.check_configured_fom()
        return self.M_external_engine.retrieve_fom_nonlinear_jacobian(u, self.M_fom_specifics,
                                                                      _elements=_elements,
                                                                      _indices=_indices)

    def assemble_fom_matrix(self, _param, _elements=None, _indices=None, withMass=False):
        """Method which performs the assembling of the FOM system matrices. In particular, it always returns the FOM
        stiffness matrix A; if withMass is set to True, it also returns the FOM mass matrix. The keyword arguments
        _elements and _indices respectively allow to evaluate the matrices over a subset of mesh elements and in a
        subset of dofs indices. If the FOM problem is not correctly initialized, it raises an error.

        .. note:: The matrices are returned in sparse COO format, unless `_indices` is not None. In such a case, instead,
          the value of the matrix at the selected indices is returned.

        .. note:: If `_param` has the default value of 1, the stiffness matrix is computed assuming all the parameters
          equal to 1; such workaround is useful when the computation of the H10-norm is desired.

        :param _param: value of the parameter or 1, if the parameters are all defaulted to 1
        :type _param: numpy.ndarray or int
        :param _elements: set of elements over which evaluating the matrices or None. Defaults to None.
        :type _elements: numpy.ndarray or NoneType
        :param _indices: set of dofs indices at which evaluating the matrices or None. Defaults to None.
        :type _indices: numpy.ndarray or NoneType
        :param withMass: if True, also the mass matrix is computed. Defaults to False.
        :type withMass: bool
        :return: matrix (or matrices) in the desired format
        :rtype: numpy.ndarray or tuple(numpy.ndarray, numpy.ndarray) or NoneType
        """
        self.check_configured_fom()
        return self.M_external_engine.assemble_fom_matrix(_param, self.M_fom_specifics,
                                                          _elements=_elements, _indices=_indices, withMass=withMass)

    def assemble_fom_rhs(self, _param, _elements=None, _indices=None):
        """Method which performs the assembling of the FOM system right-hand side vector. The keyword arguments
        _elements and _indices respectively allow to evaluate the vector over a subset of mesh elements and in a subset
        of dofs indices. If the FOM problem is not correctly initialized, it raises an error.

        :param _param: value of the parameter
        :type _param: numpy.ndarray
        :param _elements: set of elements over which evaluating the matrices or None. Defaults to None.
        :type _elements: numpy.ndarray or NoneType
        :param _indices: set of dofs indices at which evaluating the matrices or None. Defaults to None.
        :type _indices: numpy.ndarray or NoneType
        :return: right-hand side vector in the desired format
        :rtype: numpy.ndarray or NoneType
        """
        self.check_configured_fom()
        return self.M_external_engine.assemble_fom_rhs(_param, self.M_fom_specifics,
                                                       _elements=_elements, _indices=_indices)

    def plot_fe_solution(self, solution1, solution2=None, folder="", name=None):
        """Method which allows, via the Matlab external engine, to plot a FOM vector over the corresponding FE space.
         If two FOM vectors are passed, a subplot of size (1,2) is built; otherwise a single plot is
         realized. The plots are Matlab `surf` plots and are saved in `.eps` format if a valid saving path is passed to
         the function, via the input argument 'folder'

        .. note :: If the adopted external engine is not the Matlab one, then the plotting method is not available and
          the function raises an error.

        :param solution1: first (and eventually only) FOM vector to be plotted over the corresponding FE space
        :type solution1: numpy.ndarray
        :param solution2: second FOM vector to be plotted over the corresponding FE space. If None, just a single plot
          is realized. Defaults to None.
        :type solution2: numpy.ndarray or NoneType
        :param folder: path to the directory where the plots has to be saved. Defaults to ""
        :type folder: str
        :param name: name of the plot to be saved in 'folder'. Defaults to None
        :type name: str or NoneType
        """
        try:
            assert self.M_external_engine.M_engine_type == 'matlab'
        except AssertionError:
            logger.critical("The plotting method is currently available just via the Matlab external engine")
            raise TypeError

        total_fom_DOFs = self.get_fom_dimension()

        try:
            assert solution1.shape[0] == total_fom_DOFs and \
                   (solution2.shape[0] == total_fom_DOFs if solution2 is not None else True)
        except AssertionError:
            logger.critical(f"Invalid input shapes! "
                            f"RB-solution shape: {solution1.shape}; "
                            f"FE-solution shape: {solution2.shape}")
            raise ValueError

        self.M_external_engine.plot_fe_solution(solution1, self.M_fom_specifics, fe_solution=solution2,
                                                folder=folder, name=name)
        return

    @property
    def num_parameters(self):
        """Method to get the number of characteristic parameters of the problem at hand

        :return: number of characteristic parameters
        :rtype: int
        """
        return self.M_parameter_handler.num_parameters

    def generate_parameter(self, prob=None, seed=0):
        """Method to generate one set of characteristic parameters, according to a specific discrete probability
        distribution, which defaults to the uniform one if not specified

        :param prob: vector encoding the discrete probability distribution used to generate the parameter.
          If None, a uniform probability distribution with 10**4 values is used. Defaults to None.
        :type prob: list or tuple or numpy.ndarray or NoneType
        :param seed: seed for the random number generation. Defaults to 0
        :type seed: int
        """
        self.M_parameter_handler.generate_parameter(prob=prob, seed=seed)
        return

    @property
    def param(self):
        """Getter method, to get the current parameter value from the parameter handler

        :return: value of the latest generated parameter
        :rtype: numpy.ndarray
        """
        self.M_current_parameter = self.M_parameter_handler.param
        return self.M_current_parameter

    @property
    def parameter_handler(self):
        """Getter method, to get the parameter handler attribute

        :return: parameter handler attribute of the class
        :rtype: ParameterHandler
        """
        return self.M_parameter_handler

    @staticmethod
    def generate_fom_coordinates(number_of_fom_coordinates, total_fom_coordinates_number,
                                 min_coord=0, max_coord=None,
                                 sampling='random', dof_per_direction=0):
        """Static method to execute the subsampling of a FEM solution is a proper subset of FEM coordinates in the
        computational domain. The number of coordinates to be chosen, the limiting values of the FEM coordinates indices
        and the type of sampling must be specified in input.

        :param number_of_fom_coordinates: number of FEM coordinates to be sampled (if the sampling is random) or
           list of the number of fem coordinates to be generated in each direction (if the sampling is tensorial).
        :type number_of_fom_coordinates: int or list[int, int]
        :param total_fom_coordinates_number: total number of FOM coordinates in the considered computational domain
        :type total_fom_coordinates_number: int
        :param min_coord: minimum value for the FEM coordinates indices. Defaults to 0
        :type min_coord: int
        :param max_coord: maximum value for the FEM coordinates indices. If smaller than 'min_coord' a ValueError is
           raised. Defaults to None, i.e. to the total number of FOM coordinates in the considered computational domain
        :type max_coord: int or NoneType
        :param sampling: type of sampling to be executed. It can be:

            * "random": both the input and the output coordinates are selected uniformly at random in
              [`min_coord`, `max_coord`]
            * "random_equal": the input and the output coordinates coincide and they are selected uniformly at random in
              [`min_coord`, `max_coord`]
            * "tensorial": the selected input and output coordinates are selected to form a grid in the unit square
               with the desired number of elements along the two directions

           Otherwise, a ValueError is raised. Defaults to "random".
        :type sampling: str
        :param dof_per_direction: number of DOFS in both directions, used if the sampling is tensorial. It must be
           strictly positive if tensorial sampling is chosen, otherwise a ValueError is raised. For instance, it
           equals the number of elements in each direction plus 1 if P1 FE are used. Defaults to 0.
        :type dof_per_direction: int
        :return: generated fem locations where to sample the FEM solutions to construct the input and output datasets
        :rtype: tuple(numpy.ndarray, numpy.ndarray)
        """

        if max_coord is None:
            max_coord = total_fom_coordinates_number

        if type(number_of_fom_coordinates) is int and \
                number_of_fom_coordinates >= total_fom_coordinates_number\
                or type(number_of_fom_coordinates) is list and \
                number_of_fom_coordinates[0]*number_of_fom_coordinates[1] >= total_fom_coordinates_number:
            logger.warning("The desired number of fom coordinates exceeds their total number."
                           " All the coordinates are selected.")
            fom_locations = np.arange(total_fom_coordinates_number)

        else:
            if sampling == 'random':

                if type(number_of_fom_coordinates) is not int:
                    raise TypeError("Coordinates generation via random sampling requires the first argument to be "
                                    "of type int")

                fom_locations = np.zeros(number_of_fom_coordinates)

                for iCoord in range(number_of_fom_coordinates):
                    fom_locations[iCoord] = random.randint(min_coord, max_coord)

            elif sampling == 'tensorial':

                if type(number_of_fom_coordinates) is not list:
                    raise TypeError("Coordinates generation via tensorial sampling requires "
                                    "the first argument to be of type list and have a length of 2.")
                if dof_per_direction <= 0:
                    raise ValueError("A positive value has to be passed as dof_per_direction so that fom coordinates "
                                     "are generated")

                fom_locations = np.zeros(number_of_fom_coordinates[0] * number_of_fom_coordinates[1])

                jump_x = np.ceil(float(dof_per_direction) / float(number_of_fom_coordinates[0] + 1))
                jump_from_border_x = np.floor(
                    (float(dof_per_direction) - jump_x * float(number_of_fom_coordinates[0] + 1)) / 2.)

                jump_y = np.ceil(float(dof_per_direction) / float(number_of_fom_coordinates[1] + 1))
                jump_from_border_y = np.floor(
                    (float(dof_per_direction) - jump_y * float(number_of_fom_coordinates[1] + 1)) / 2.)

                logger.debug(f'Choosing tensorial grid selection, with jumps ({jump_x}, {jump_y}) and '
                             f'jumps from border ({jump_from_border_x}, {jump_from_border_y})')

                fom_location_counter = 0
                for iX in range(number_of_fom_coordinates[0]):
                    for iY in range(number_of_fom_coordinates[1]):
                        fom_locations[fom_location_counter] = (jump_from_border_x +
                                                               dof_per_direction * jump_from_border_y +
                                                               (iY + 1) * jump_y * dof_per_direction +
                                                               (iX + 1) * jump_x)
                        fom_location_counter += 1
            else:
                raise ValueError(f"Unrecognized sampling type {sampling}. Admissible values: ['random', 'tensorial']")

        return fom_locations.astype(int)


__all__ = [
    "default_theta_function",
    "default_full_theta_function",
    "FomProblem"
    ]
