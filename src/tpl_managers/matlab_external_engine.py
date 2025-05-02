#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 13:51:49 2018
@author: Niccolo' Dal Santo
@email : niccolo.dalsanto@epfl.ch
"""

import matlab.engine
import numpy as np
import os
import io
from src.tpl_managers import external_engine as ee

import logging.config
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class MatlabExternalEngine(ee.ExternalEngine):
    """Class which defines a wrapper around the 'feamat' Matlab library, which is used to solve the 2D FOM problems.
    The library can be found at https:\\github.com\lucapegolotti\feamat\tree\pyorb_wrappers_parabolic
    The current class actually defines a wrapper around the  'pyorb_wrappers' submodule which can be found at
    https:\\github.com\lucapegolotti\feamat\tree\pyorb_wrappers_parabolic\TPL_wrappers\pyorb_wrappers
    """

    def __init__(self, _library_path):
        """ Initialization of the Matlab external engine

        :param _library_path: path to the library used to solve the FOM problems. It this case it is a path to 'feamat'
          library
        :type _library_path: str
        """

        ee.ExternalEngine.__init__(self, 'matlab', _library_path)
        return

    def start_specific_engine(self):
        """ Method to start the Matlab External Engine and initialize the selected Matlab library
        """

        self.M_engine = matlab.engine.start_matlab()
        self.M_engine.addpath(self.M_engine.genpath(self.M_library_path))
        logger.info('Successfully started Matlab engine and corresponding FOM library %s ' % self.M_library_path)
        return

    def quit_specific_engine(self):
        """ Method to quit the Matlab External Engine
        """

        self.M_engine.quit()
        logger.info('Successfully quit Matlab engine')
        return

    def convert_parameter(self, _param):
        """ Method which converts python numpy arrays into Matlab arrays of doubles. In the context of the project it
        is used to pass the characteristic parameters to the Matlab FOM solver

        :param _param: value of the parameter
        :type _param: numpy.ndarray
        :return: value of the parameter, converted into an array of Matlab doubles
        :rtype: matlab.double
        """

        return self.convert_double(_param)

    @staticmethod
    def convert_double(_np_array):
        """Method which converts python numpy arrays into Matlab arrays of type double.

        :param _np_array: input python numpy array
        :type _np_array: numpy.ndarray
        :return: numpy array, converted into an array of Matlab doubles
        :rtype: matlab.double
        """

        return matlab.double(_np_array.tolist())

    @staticmethod
    def convert_indices(_indices):
        """Method which converts python indices, of type int, into Matlab indices, of type matlab.int64

        :param _indices: indices to be converted
        :type _indices: numpy.ndarray
        :return: value of the indices, converted into a matlab array of type int64
        :rtype: matlab.int64
        """

        return matlab.int64(_indices.tolist())

    def convert_types(self, _fom_specifics):
        """Method that converts the FOM problem specifics, as defined in Python, in such a way that they can be handled
        and used by the Matlab FOM solver. In particular all numpy arrays are converted to Matlab doubles, while
        default conversions are applied to all the field variables of a different type

        :param _fom_specifics: specifics of the FOM problem at hand
        :type _fom_specifics: dict
        :return: fom specifics of the problem at hand, converted in a Matlab-friendly way
        :rtype: dict
        """

        converted_fom_specifics = {}

        for key in _fom_specifics:

            if type(_fom_specifics[key]) is np.ndarray:
                converted_fom_specifics.update({key: self.convert_double(_fom_specifics[key])})
            else:
                converted_fom_specifics.update({key: _fom_specifics[key]})

        return converted_fom_specifics

    def solve_parameter(self, _param, _fom_specifics, _get_nonlinear_terms=False):
        """ Method which provides the solution to the FOM problem, given the values of the characteristic parameters
        and the specifics of the problem. Additionally, the identificative number of the test (which affects the
        boundary conditions and the expression of the forcing term and of the initial condition) can be provided.
        If not provided, the default problem settings are used; such settings can be found in the 'solve_parameter'
        function at https:\\github.com\lucapegolotti\feamat\tree\pyorb_wrappers_parabolic\TPL_wrappers\pyorb_wrappers

        .. note:: If the FOM problem at hand is unsteady, the solution is returned without the initial condition, since
          that one is known, equal for all the parameter values and retrievable via the method
          :func:`~matlab_external_engine.MatlabExternalEngine.get_initial_condition`

        :param _param: value of the parameter
        :type _param: numpy.ndarray
        :param _fom_specifics: specifics of the FOM problem at hand
        :type _fom_specifics: dict
        :param _get_nonlinear_terms: if True, also the non-linear term and its jacobian are returned. Defaults to False
        :type _get_nonlinear_terms: bool
        :return: solution to the FOM problem at hand and execution time if the problem in unsteady
        :rtype: numpy.ndarray or tuple(numpy.ndarray, float)
        """

        matlab_sol = self.M_engine.solve_parameter(self.convert_parameter(_param), _fom_specifics, stdout=io.StringIO())

        sol = np.array(matlab_sol['u'])
        execution_time = matlab_sol['execution_time']
        if _get_nonlinear_terms:
            nl_term = np.array(matlab_sol['nl_term'])
            nl_term_jac = np.array(matlab_sol['nl_jac'])

        if 'final_time' in _fom_specifics.keys():
            if not _get_nonlinear_terms:
                return sol[:, 1:], execution_time
            else:
                nl_term_jac[..., :2] = nl_term_jac[..., :2] - 1
                return sol[:, 1:], execution_time, nl_term, nl_term_jac

        else:
            if not _get_nonlinear_terms:
                return sol[:, 0], execution_time
            else:
                nl_term_jac[:, :2] = nl_term_jac[:, :2] - 1
                return sol[:, 0], execution_time, nl_term[:, 0], nl_term_jac

    def get_initial_condition(self, _fom_specifics, _param):
        """ Method which returns the initial condition to the FOM problem, provided that the FOM problem is unsteady.
        If the FOM problem is steady, it just returns None.

        :param _fom_specifics: specifics of the FOM problem at hand
        :type _fom_specifics: dict
        :param _param: value of the parameter
        :type _param: numpy.ndarray
        :return: initial condition to the FOM problem if it is unsteady, else None
        :rtype: numpy.ndarray or None
        """

        if 'final_time' in _fom_specifics.keys():
            u0 = self.M_engine.get_initial_condition(self.convert_parameter(_param), _fom_specifics)
            u0 = np.array(u0)
        else:
            logger.error("The FOM problem appears to be steady; it is not possible to return an initial condition")
            u0 = None

        return u0

    def get_reference_sol(self, _param, _fom_specifics, timestep_nb, number_of_time_instances=None):
        """Method to get a reference solution to the FOM problem at hand. In the context of the current project, the
        computation of a reference solution is supported only for unsteady problems; therefore such method returns None
        if it is called passing the specifics of a steady problem. In particular, the computation of a reference
        solution for a steady problem makes use of the ode23t Matlab package, whose features can be found at
        https:\\it.mathworks.com\help\matlab\ref\ode23t.html. The method is implemented only for the thermal-block
        problem. Finally, the reference solution may be even computed just up to a certain time instance and for a
        specific time-step, via the keyword arguments number_of_time_instances and timestep_nb.

        :param _param: value of the parameter
        :type _param: numpy.ndarray
        :param _fom_specifics: specifics of the FOM problem at hand
        :type _fom_specifics: dict
        :param number_of_time_instances: number of time instances used in the time marching scheme to solve the FOM
          problem and get the reference solution. If None, the value declared in the config file is considered.
          Defaults to None.
        :type number_of_time_instances: int or NoneType
        :param timestep_nb: number of timesteps at which the computation of the reference solution is desired.
          If None, the reference solution is computed at all the timesteps. Defaults to None.
        :type timestep_nb: int or NoneType
        :return: reference solution to the FOM problem at hand, computed via the ode23t Matlab package
        :rtype: numpy.ndarray
        """

        if 'final_time' in _fom_specifics.keys() and _fom_specifics['model'] == 'thermal_block':
            if number_of_time_instances is None:
                number_of_time_instances = _fom_specifics['number_of_time_instances']
            used_fom_specifics = self.convert_types(_fom_specifics)
            exact_sol = self.M_engine.get_exact_sol(self.convert_parameter(_param), used_fom_specifics, timestep_nb,
                                                    number_of_time_instances)
            u_exact = np.array(exact_sol['u_exact'])
        else:
            u_exact = None

        return u_exact

    def get_reference_rb_sol(self, _param, _fom_specifics, rb_basis, timestep_nb, number_of_time_instances=None):
        """Method to get a reference solution to the FOM problem at hand, computed as the RB projection over a certain
        basis rb_basis In the context of the current project, the computation of a reference solution is supported only
        for unsteady problems; therefore such method returns None if it is called passing the specifics of a steady
        problem. In particular, the computation of a reference solution for a steady problem makes use of the ode23t
        Matlab package, whose features can be found at https:\\it.mathworks.com\help\matlab\ref\ode23t.html
        The method is implemented only for the thermal-block problem. Finally, the reference solution may be even
        computed just up to a certain time instance and for a specific time-step, via the keyword arguments
        number_of_time_instances and timestep_nb.

        :param _param: value of the parameter
        :type _param: numpy.ndarray
        :param _fom_specifics: specifics of the FOM problem at hand
        :type _fom_specifics: dict
        :param rb_basis: Reduced Basis basis
        :type rb_basis: numpy.ndarray
        :param number_of_time_instances: number of time instances used in the time marching scheme to solve the FOM
          problem and get the reference solution. If None, the value declared in the config file is considered.
          Defaults to None.
        :type number_of_time_instances: int or NoneType
        :param timestep_nb: number of timesteps at which the computation of the reference solution is desired.
          If None, the reference solution is computed at all the timesteps. Defaults to None.
        :type timestep_nb: int or NoneType
        :return: reference solution to the FOM problem at hand, computed via the ode23t Matlab package and projected
          over the Reduced Space
        :rtype: numpy.ndarray
        """
        if 'final_time' in _fom_specifics.keys() and _fom_specifics['model'] == 'thermal_block':
            if number_of_time_instances is None:
                number_of_time_instances = _fom_specifics['number_of_time_instances']
            used_fom_specifics = self.convert_types(_fom_specifics)
            exact_sol = self.M_engine.get_exact_rb_sol(self.convert_parameter(_param), used_fom_specifics,
                                                       self.convert_double(rb_basis),
                                                       timestep_nb, number_of_time_instances)
            u_exact = np.array(exact_sol['u_exact'])
        else:
            u_exact = None

        return u_exact

    def build_fom_affine_components(self, _operator, _num_affine_components, _fom_specifics,
                                    number_of_time_instances=None):
        """Method which allows to compute the FOM affine components of the operators involved in the FOM problem at hand
        The computation of such components is equivalent for steady and unsteady problems regarding the operators 'A'
        and 'M' (stiffness and mass matrices), being time-independent. The computation of the right-hand side affine
        components, instead, is different between steady and unsteady problems, since to ones of unsteady problems
        must be evaluated over time. Additionally, it is possible to evaluate such time-dependent rhs affine components
        in a subset of equispaced time instants, by suitably setting the input argument number_of_time_instances.
        Finally, it is even possible to pass 'f_space' as an operator; in such case the affine components are computed
        referring to the spacial part of the forcing term (i.e. to the whole forcing term, if the problem is steady,
        to the part of the forcing term which depends of space, if the problem is unsteady, under the assumption that
        the forcing term can be written as f(x,t) = f1(x)*f2(t))

        :param _operator: operator of which the computation of the affine components is desired. Admissible values are
          either 'A', 'M', 'f' or 'f_space'
        :type _operator: str
        :param _num_affine_components: number of the desired affine components
        :type _num_affine_components: int
        :param _fom_specifics: specifics of the FOM problem at hand
        :type _fom_specifics: dict
        :param number_of_time_instances: number of time instances at which the right-hand side vector has to be
          evaluated. If None or if higher than the total number of time instances, the evaluation occurs at all the
          time instances considered in the problem. Defaults to None.
        :type number_of_time_instances: int or NoneType
        :return: dictionary containing al the requested affine components associated to the desired operator
        :rtype: dict or NoneType
        """

        if _operator == 'f' and number_of_time_instances is not None:
            logger.info('Building affine components for operator %s_reduced' % _operator)
        else:
            logger.info('Building affine components for operator %s' % _operator)

        used_fom_specifics = self.convert_types(_fom_specifics)

        if 'final_time' in _fom_specifics.keys() and 'number_of_time_instances' in _fom_specifics.keys() and \
                _operator == 'f' and number_of_time_instances is not None:
            affine_components = self.M_engine.build_fom_affine_components_unsteady(_operator, used_fom_specifics,
                                                                                   float(number_of_time_instances))
        elif 'final_time' in _fom_specifics.keys() and 'number_of_time_instances' in _fom_specifics.keys():
            affine_components = self.M_engine.build_fom_affine_components_unsteady(_operator, used_fom_specifics)
        else:
            affine_components = self.M_engine.build_fom_affine_components(_operator, used_fom_specifics)

        # rescale the matrix indices so that the counting starts from 0 (and not from 1 as in MATLAB)
        if _operator == 'A':
            matrix_affine = {}
            for iQa in range(_num_affine_components):
                matrix_affine[_operator + str(iQa)] = np.array(affine_components[_operator + str(iQa)])
                matrix_affine[_operator + str(iQa)][:, :2] = matrix_affine[_operator + str(iQa)][:, :2] - 1

            affine_components = matrix_affine

        if 'final_time' in _fom_specifics.keys() and 'number_of_time_instances' in _fom_specifics.keys():
            if _operator == 'M':
                matrix_affine = {}
                for iQm in range(_num_affine_components):
                    matrix_affine[_operator + str(iQm)] = np.array(affine_components[_operator + str(iQm)])
                    matrix_affine[_operator + str(iQm)][:, :2] = matrix_affine[_operator + str(iQm)][:, :2] - 1

                affine_components = matrix_affine

        # resetting to just one-size array
        if _operator == 'f' or _operator == 'f_space':
            rhs_affine = {}
            for iQf in range(_num_affine_components):
                rhs_affine[_operator + str(iQf)] = np.array(affine_components[_operator + str(iQf)])
                rhs_affine[_operator + str(iQf)] = np.reshape(rhs_affine[_operator + str(iQf)],
                                                              rhs_affine[_operator + str(iQf)].shape[0])
            affine_components = rhs_affine

        return affine_components

    def expand_f_space_affine_components(self, _feAffineFSpace, _fom_specifics):
        """Method to expand the spatial affine components of the right-hand side vector in time

        :param _feAffineFSpace: FOM affine components for the rhs vector
        :type: list[numpy.ndarray]
        :param _fom_specifics: specifics of the FOM problem at hand
        :type _fom_specifics: dict
        """
        return self.M_engine.expand_f_space_affine_components(self.convert_parameter(np.array(_feAffineFSpace)),
                                                              self.convert_types(_fom_specifics))

    def retrieve_fom_nonlinear_term(self, u, _fom_specifics, _elements=None, _indices=None):
        """Method that builds the FOM non-linear term based on the current value of the solution (passed as first
        argument to the function). If the problem is linear, it simply return an empty array of the right dimension.

        :param u: current FOM solution
        :type u: numpy.ndarray
        :param _fom_specifics: specifics of the FOM problem at hand
        :type _fom_specifics: dict
        :param _elements: set of elements over which evaluating the matrices or None. Defaults to None.
        :type _elements: numpy.ndarray or NoneType
        :param _indices: set of dofs indices at which evaluating the matrices or None. Defaults to None.
        :type _indices: numpy.ndarray or NoneType
        :return FOM non-linear term
        :rtype: numpy.ndarray
        """

        if _elements is None or _indices is None:
            nl_term = self.M_engine.build_fom_nonlinear_term(self.convert_parameter(u),
                                                             self.convert_types(_fom_specifics))
        else:
            nl_term = self.M_engine.build_fom_nonlinear_term(self.convert_parameter(u),
                                                             self.convert_types(_fom_specifics),
                                                             self.convert_parameter(_elements),
                                                             self.convert_parameter(_indices + 1))
        nl_term = np.array(nl_term)

        return nl_term

    def retrieve_fom_nonlinear_jacobian(self, u, _fom_specifics, recompute_every=1, _elements=None, _indices=None):
        """Method that builds the FOM non-linear jacobian based on the current value of the solution (passed as first
        argument to the function). If the problem is linear, it simply return an empty array of the right dimension.

        :param u: current FOM solution
        :type u: numpy.ndarray
        :param _fom_specifics: specifics of the FOM problem at hand
        :type _fom_specifics: dict
        :param recompute_every: for unsteady problems, number of time instants after which the Jacobian is recomputed.
           It defaults to 1.
        :type recompute_every: int
        :param _elements: set of elements over which evaluating the matrix or None. Defaults to None.
        :type _elements: numpy.ndarray or NoneType
        :param _indices: set of dofs indices at which evaluating the matrix or None. Defaults to None.
        :type _indices: numpy.ndarray or NoneType
        :return FOM non-linear jacobian
        :rtype: numpy.ndarray
        """

        if _elements is None or _indices is None:
            nl_jac = self.M_engine.build_fom_nonlinear_jacobian(self.convert_parameter(u),
                                                                self.convert_types(_fom_specifics),
                                                                recompute_every)
        else:
            nl_jac = self.M_engine.build_fom_nonlinear_jacobian(self.convert_parameter(u),
                                                                self.convert_types(_fom_specifics),
                                                                recompute_every,
                                                                self.convert_parameter(_elements),
                                                                self.convert_parameter(_indices + 1))

        nl_jac = np.array(nl_jac)
        nl_jac[..., :2] = nl_jac[..., :2] - 1

        return nl_jac

    def assemble_fom_matrix(self, _param, _fom_specifics, _elements=None, _indices=None, withMass=False):
        """Method which performs the assembling of the FOM system matrices. In particular, it always returns the FOM
        stiffness matrix A; if withMass is set to True, it also returns the FOM mass matrix. The keyword arguments
        _elements and _indices respectively allow to evaluate the matrices over a subset of mesh elements and in a
        subset of dofs indices.

        .. note:: The matrices are returned in sparse COO format, unless _indices is not None. In such a case, instead,
          the value of the matrix at the selected indices is returned.

        .. note:: if _param has the default value of 1, the stiffness matrix is computed assuming all the parameters
          equal to 1; such workaround is useful when the computation of the H10-norm is desired.

        :param _param: value of the parameters or 1, if the parameters are all defaulted to 1
        :type _param: numpy.ndarray or int
        :param _fom_specifics: specifics of the FOM problem at hand
        :type _fom_specifics: dict
        :param _elements: set of elements over which evaluating the matrices or None. Defaults to None.
        :type _elements: numpy.ndarray or NoneType
        :param _indices: set of dofs indices at which evaluating the matrices or None. Defaults to None.
        :type _indices: numpy.ndarray or NoneType
        :param withMass: if True, also the mass matrix is computed. Defaults to False.
        :type withMass: bool
        :return: matrix (or matrices) in the desired format
        :rtype: numpy.ndarray or tuple(numpy.ndarray, numpy.ndarray) or NoneType
        """

        if type(_param) is not int:
            _param = self.convert_parameter(_param)

        if _elements is None or _indices is None:
            matrices = self.M_engine.assemble_fom_matrix(_param, _fom_specifics)
            A = np.array(matrices['A'])
            A[:, :2] = A[:, :2] - 1

            if withMass:
                M = np.array(matrices['M'])
                M[:, :2] = M[:, :2] - 1
                return A, M
            else:
                return A
        else:
            matrices = self.M_engine.assemble_fom_matrix(_param, _fom_specifics,
                                                         self.convert_parameter(_elements),
                                                         self.convert_parameter(_indices + 1))
            A = np.array(matrices['A'])
            A[:, :2] = A[:, :2] - 1

            if withMass:
                M = np.array(matrices['M'])
                M[:, :2] = M[:, :2] - 1
                return A, M
            else:
                return A

    def assemble_fom_rhs(self, _param, _fom_specifics, _elements=None, _indices=None):
        """Method which performs the assembling of the FOM system right-hand side vector. The keyword arguments
        _elements and _indices respectively allow to evaluate the vector over a subset of mesh elements and in a subset
        of dofs indices.

        :param _param: value of the parameter
        :type _param: numpy.ndarray
        :param _fom_specifics: specifics of the FOM problem at hand
        :type _fom_specifics: dict
        :param _elements: set of elements over which evaluating the matrices or None. Defaults to None.
        :type _elements: numpy.ndarray or NoneType
        :param _indices: set of dofs indices at which evaluating the matrices or None. Defaults to None.
        :type _indices: numpy.ndarray or NoneType
        :return: right-hand side vector in the desired format
        :rtype: numpy.ndarray or NoneType
        """

        if _elements is None or _indices is None:
            rhs = self.M_engine.assemble_fom_rhs(self.convert_parameter(_param), _fom_specifics)
            ff = np.array(rhs['f'])
            ff = np.reshape(ff, (ff.shape[0],))
            return ff

        else:
            rhs = self.M_engine.assemble_fom_rhs(self.convert_parameter(_param), _fom_specifics,
                                                 self.convert_parameter(_elements),
                                                 self.convert_parameter(_indices + 1))

            ff = np.array(rhs['f'])

            return ff

    def plot_fe_solution(self, rb_solution, _fom_specifics, fe_solution=None, folder="", name=None):
        """Method which allows to plot a FOM vector over the corresponding FE space.
        If two FOM vectors are passed, a subplot of size (1,2) is built; otherwise a single plot is
        realized. The plots are Matlab `surf` plots and are saved in `.eps` format if a valid saving path is passed to
        the function, via the input argument 'folder'

        :param rb_solution: first (and eventually only) FOM vector to be plotted over the corresponding FE space
        :type rb_solution: numpy.ndarray
        :param _fom_specifics: dictionary containing the specifics of the FOM problem at hand
        :type _fom_specifics: dict
        :param fe_solution: second FOM vector to be plotted over the corresponding FE space. If None, just a single plot
          is realized. Defaults to None.
        :type fe_solution: numpy.ndarray or NoneType
        :param folder: path to the directory where the plots has to be saved. Defaults to ""
        :type folder: str
        :param name: name of the plot to be saved in 'folder'. Defaults to None
        :type name: str or NoneType
        """
        try:
            assert folder is not None and os.path.isdir(folder)
        except AssertionError:
            logger.critical(f"{folder} is not a valid folder")
            raise ValueError
        try:
            assert name is not None
        except AssertionError:
            logger.warning("No name has been selected for the plot. Assigning the default name "
                           "Solution plot")
            name = "Solution plot"

        if folder[-1] != os.sep:
            folder += os.sep

        if fe_solution is None:
            self.M_engine.plot_fe_solution(self.convert_parameter(rb_solution), self.convert_parameter(np.array([])),
                                           _fom_specifics, folder, name, nargout=0)
        else:
            self.M_engine.plot_fe_solution(self.convert_parameter(rb_solution), self.convert_parameter(fe_solution),
                                           _fom_specifics, folder, name, nargout=0)

        return


__all__ = [
    "MatlabExternalEngine"
]
