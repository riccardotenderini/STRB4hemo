#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 21:35:38 2019
@author: Riccardo Tenderini
@email : riccardo.tenderini@epfl.ch
"""

import sys
import os
sys.path.insert(0, os.path.normpath('../../'))
sys.path.insert(0, os.path.normpath('../'))

import os
import src.pde_problem.fom_problem as fp


import logging.config
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class FomProblemUnsteady(fp.FomProblem):
    """Class defining a generic unsteady parameter-dependent FOM problem; since it does not refer to any specific
    FOM problem, such class cannot be instantiated. Indeed the main method to define the parameter-dependent functions
    flags an error whenever invoked; actually this method is the one of the parent class FomProblem and it is not
    overwritten by the present class. All the specific unsteady FOM problems are coded as children of this virtual
    parent class. It inherits from :class:`~fom_problem.FomProblem`
    """

    def __init__(self, _parameter_handler, _external_engine=None, _fom_specifics=None):
        """Initializing the Full Order Model (FOM) unsteady parameter-dependent problem, equipping it with the parameter
        handler and, if provided, the external engine to solve the FOM problem (either Matlab or C++) and the specifics
        of the problem itself.

        :param _parameter_handler: ParameterHandler object, which performs the task of handling the parameters involved
            in the problem
        :type _parameter_handler: ParameterHandler
        :param _external_engine: engine which perform the computation of all the FOM-related quantities. In the context
            of this project it can be either a Matlab external engine or a C++ external engine.
            If None, no initialization of the external engine is made. Defaults to None.
        :type _external_engine: ExternalEngine or NoneType
        :param _fom_specifics: dictionary containing all the specifics of the FOM problem at hand. Being the current
            problem an unsteady one, additional infos must be added to the 'original' steady-based  _fom_specific
            dictionary. Refer to the configuration file for further details.
            If None, no initialization of the FOM specifics is made. Defaults to None.
        :type _fom_specifics: dict or NoneType
        """
        if _fom_specifics is not None:
            if 'final_time' in _fom_specifics.keys() and 'number_of_time_instances' in _fom_specifics.keys():
                super().__init__(_parameter_handler, _external_engine=_external_engine, _fom_specifics=_fom_specifics)
            else:
                raise Exception("Some of the fom specifics fields characterizing an unsteady problem is missing")
        else:
            super().__init__(_parameter_handler, _external_engine=_external_engine, _fom_specifics=None)
        return

    def configure_fom(self, _external_engine=None, _fom_specifics=None):
        """Method which allows to configure the unsteady FOM problem by setting the external engine and\or the problem
        specifics. It must be called if the FOM problem has been initialized missing either the external engine
        or the FOM specifics. It is called in the __init__ of the class.

        :param _external_engine: engine which perform the computation of all the FOM-related quantities. In the context
          of this project it can be either a Matlab external engine or a C++ external engine.
          If None, no initialization of the external engine is made. Defaults to None.
        :type _external_engine: ExternalEngine or NoneType
        :param _fom_specifics: dictionary containing all the specifics of the FOM problem at hand.
          If None, no initialization of the FOM specifics is made. Defaults to None.
        :type _fom_specifics: dict or NoneType
        """

        if _fom_specifics is not None:
            if 'final_time' in _fom_specifics.keys() and 'number_of_time_instances' in _fom_specifics.keys():
                logger.debug("The unsteady fom problem can be correctly configured")
            else:
                raise Exception("Some of the fom specifics fields characterizing an unsteady problem is missing")

        super().configure_fom(_external_engine=_external_engine, _fom_specifics=_fom_specifics)
        return

    def check_time_specifics(self):
        """Method which checks if the M_fom_specifics attribute contains all the necessary fields to initialize and
        solve an unsteady FOM problem, i.e.:

            * 'final_time': final time of the simulation
            * 'number_of_time_instances': number of time instants at which the FOM solution is evaluated
            * 'method': adopted time marching scheme, which can be chosen among 'Theta' (Theta method), 'BDF'
              (Backward Differential Formulas) and 'AM' (Adams-Moulton)
            * 'theta': value of the parameter Theta, used in the Theta-method
            * 'step_number_fom': number of steps used by the multistep solvers BDF and AM.
              Defaults to 1 if the time marching scheme is the Theta-method

        :return: True if the fom_specifics fit an unsteady FOM problem, False otherwise
        :rtype: bool
        """

        if ('final_time' in self.M_fom_specifics.keys() and
                'number_of_time_instances' in self.M_fom_specifics.keys()):
            return True
        else:
            return False

    @property
    def dt(self):
        """Method which returns the value of the time-step used to perform the time marching in the FOM problem at hand.
        Raises an error if the FOM problem has not been correctly initialized as an unsteady one.

        :return: value of the time-step if the unsteady FOM problem has been correctly initialized, else None
        :rtype: float or NoneType
        """

        if self.check_time_specifics():
            dt = self.M_fom_specifics['final_time'] / self.M_fom_specifics['number_of_time_instances']
        else:
            raise Exception("Some of the fom specifics fields characterizing an unsteady problem is missing. "
                            "The current FOM problem is thus initialized as a steady one. No dt can be returned")

        return dt

    @property
    def time_dimension(self):
        """Method which returns the number of time instances used for the time marching in the FOM problem at hand.
        Raises an error if the FOM problem has not been correctly initialized as an unsteady one.

        :return: number of used time instances if the unsteady FOM problem has been correctly initialized, else None
        :rtype: int or NoneType
        """

        if self.check_time_specifics():
            return int(self.M_fom_specifics['number_of_time_instances'])
        else:
            raise Exception("The field number_of_time_instances that characterizes any unsteady FOM problem has not "
                            "been initialized in the fom_specifics. No time dimension can be thus returned")

    @property
    def time_specifics(self):
        """Method which returns the value of the time-step and the number of time instances used for the time marching
        in the FOM problem at hand. It raises an error if the FOM problem has not been correctly initialized as an
        unsteady one.

        :return: value of the time-step and number time instances used if the unsteady FOM problem has been correctly
            initialized, else None
        :rtype: tuple(float, int) or NoneType
        """

        return self.dt, self.time_dimension

    def retrieve_fom_affine_components(self, _operator, _num_affine_components, number_of_time_instances=None):
        """Method which overrides the corresponding one of the parent steady class. In particular, it calls the parent
        method if the operator to be considered is either 'A' or 'M' (i.e. a time-independent operator), while it
        tackles in a different way the computation of the affine components for the right-hand side, being it
        time-dependent in this context and thus to be evaluated over all the time instants. Additionally, it allows to
        evaluate the right-hand side vector just over an equispaced subset of time instances by properly setting the
        'number_of_time_instances' argument.

        :param _operator: operator of which the computation of the affine components is desired. Admissible values are
          either 'A' or 'f'
        :type _operator: str
        :param _num_affine_components: number of the desired affine components
        :type _num_affine_components: int
        :param number_of_time_instances: number of time instances at which the right-hand side vector has to be
          evaluated. If None or if higher than the total number of time instances, the evaluation occurs at all the
          time instances considered in the problem. Defaults to None.
        :type number_of_time_instances: int or NoneType
        :return: dictionary containing al the requested affine components associated to the desired operator
        :rtype: dict or NoneType
        """
        self.check_configured_fom()
        if _operator == 'f' and (number_of_time_instances is not None and
                                 number_of_time_instances < self.M_fom_specifics['number_of_time_instances']):
            return self.M_external_engine.build_fom_affine_components(_operator, _num_affine_components,
                                                                      self.M_fom_specifics,
                                                                      number_of_time_instances)
        elif _operator == 'f' and (number_of_time_instances is None or
                                   number_of_time_instances >= self.M_fom_specifics['number_of_time_instances']):
            return self.M_external_engine.build_fom_affine_components(_operator, _num_affine_components,
                                                                      self.M_fom_specifics,
                                                                      self.M_fom_specifics['number_of_time_instances'])
        else:
            return self.M_external_engine.build_fom_affine_components(_operator, _num_affine_components,
                                                                      self.M_fom_specifics)

    def expand_f_space_affine_components(self, _feAffineFSpace):
        """Method to expand the spatial affine components of the right-hand side vector in time

        :param _feAffineFSpace: FOM affine components for the rhs vector
        :type: list[numpy.ndarray]
        """
        return self.M_external_engine.expand_f_space_affine_components(_feAffineFSpace, self.M_fom_specifics)

    def get_initial_condition(self, _param):
        """Method to get the initial condition of the problem at hand, for a given parameter value. If the FOM
        problem has not been correctly initialized as an unsteady one it raises an error. If the external engine is not
        the Matlab one, it raises a TypeError.

        :param _param: value of the parameter
        :type _param: numpy.ndarray
        :return: initial condition of the problem at hand if correctly initialized, else None
        :rtype: numpy.ndarray or NoneType
        """

        try:
            assert self.M_external_engine.M_engine_type == 'matlab'
        except AssertionError:
            logger.critical("The initial condition is currently available just via the Matlab external engine")
            raise TypeError

        if self.check_time_specifics() and self.check_configured_fom():
            u0 = self.M_external_engine.get_initial_condition(self.M_fom_specifics, _param)
        else:
            raise Exception("Some of the fom specifics fields characterizing an unsteady problem is missing. "
                            "The current FOM problem is thus initialized as a steady one. No dt can be returned")

        return u0

    def get_reference_solution(self, _param, number_of_time_instances=None, timestep_nb=None):
        """Method to get the reference solution of the problem at hand. The reference solution is available only via the
        Matlab external engine and it is computed via the ode23t Matlab package
        (https:\\it.mathworks.com\help\matlab\ref\ode23t.html) over a grid which is 10 times finer than the grid used to
        solve the FOM problem with the implemented methods. Additionally, the method is implemented only for the
        thermal-block problem. Finally, the reference solution may be even computed just up to a certain time instance
        and for a specific time-step, via the keyword arguments number_of_time_instances and timestep_nb.
        If the external engine is not the Matlab one, the method raises a TypeError. If the FOM problem has not been
        correctly initialized as an unsteady one, it raises an error.

        :param _param: value of the parameter
        :type _param: numpy.ndarray
        :param number_of_time_instances: number of time instances used in the time marching scheme to solve the FOM
          problem and get the reference solution. If None, the value declared in the config file is considered.
          Defaults to None.
        :type number_of_time_instances: int or NoneType
        :param timestep_nb: number of timesteps at which the computation of the reference solution is desired.
          If None, the reference solution is computed at all the timesteps. Defaults to None.
        :type timestep_nb: int or NoneType
        :return: reference solution to the FOM problem at hand, computed at the desired timesteps and with the desired
          number of time instances
        :rtype: numpy.ndarray
        """

        try:
            assert self.M_external_engine.M_engine_type == 'matlab'
        except AssertionError:
            logger.critical("The reference solution is currently available just via the Matlab external engine")
            raise TypeError

        self.check_configured_fom()

        if self.check_time_specifics() and self.M_fom_specifics['model'] == "thermal_block":
            logger.debug("The current problem admits the computation of a reference solution via ode23t Matlab package")
            if timestep_nb is None:
                timestep_nb = self.M_fom_specifics['number_of_time_instances']
            if number_of_time_instances is not None:
                u_exact = self.M_external_engine.get_reference_sol(_param, self.M_fom_specifics, timestep_nb,
                                                                   number_of_time_instances)
            else:
                u_exact = self.M_external_engine.get_reference_sol(_param, self.M_fom_specifics, timestep_nb,
                                                                   self.M_fom_specifics['number_of_time_instances'])
        else:
            logger.error("The current problem does not admit the computation of a reference solution")
            u_exact = None

        return u_exact

    def get_reference_rb_solution(self, _param, rb_basis, number_of_time_instances=None, timestep_nb=None):
        """Method to get the reference solution of the problem at hand, computed as the RB projection over a certain
        basis rb_basis. The reference solution is available only via the Matlab external engine and it is computed via
        the ode23t Matlab package (https:\\it.mathworks.com\help\matlab\ref\ode23t.html), over a grid which is 10
        times finer than the grid used to solve th FOM problem with the implemented methods. Additionally, the method
        is implemented only for the thermal-block problem. Finally, the reference solution may be even computed just up
        to a certain time instance and for a specific time-step, via the keyword arguments number_of_time_instances and
        timestep_nb. If the external engine is not the Matlab one, the method raises a TypeError. If the FOM problem
        has not been correctly initialized as an unsteady one, it raises an error.

        :param _param: value of the parameter
        :type _param: numpy.ndarray
        :param rb_basis: Reduced Basis basis
        :type rb_basis: numpy.ndarray
        :param number_of_time_instances: number of time instances used in the time marching scheme to solve the FOM
          problem and get the reference solution. If None, the value declared in the config file is considered.
          Defaults to None.
        :type number_of_time_instances: int or NoneType
        :param timestep_nb: number of timesteps at which the computation of the reference solution is desired.
          If None, the reference solution is computed at all the timesteps. Defaults to None.
        :type timestep_nb: int or NoneType
        :return: RB-projected reference solution to the problem at hand, computed at the desired timesteps and with the
          desired number of time instances
        :rtype: numpy.ndarray
        """

        if self.check_time_specifics() and self.M_fom_specifics['model'] == "thermal_block":
            logger.debug("The current problem admits the computation of a reference RB solution "
                         "via ode23t Matlab package")
            if timestep_nb is None:
                timestep_nb = self.M_fom_specifics['number_of_time_instances']
            if number_of_time_instances is not None:
                u_exact = self.M_external_engine.get_reference_rb_sol(_param, self.M_fom_specifics, rb_basis,
                                                                      timestep_nb, number_of_time_instances)
            else:
                u_exact = self.M_external_engine.get_reference_rb_sol(_param, self.M_fom_specifics, rb_basis,
                                                                      timestep_nb,
                                                                      self.M_fom_specifics['number_of_time_instances'])
        else:
            logger.error("The current problem does not admit the computation of a reference RB solution")
            u_exact = None

        return u_exact

    def retrieve_fom_nonlinear_jacobian(self, u, _elements=None, _indices=None, recompute_every=1):
        """Method that builds the FOM non-linear jacobian based on the current value of the solution (passed as first
        argument to the function). If the problem is linear, it simply returns an empty array of the right dimension.

        :param u: current FOM solution
        :type u: numpy.ndarray
        :param _elements: set of elements over which evaluating the matrix or None. Defaults to None.
        :type _elements: numpy.ndarray or NoneType
        :param _indices: set of dofs indices at which evaluating the matrix or None. Defaults to None.
        :type _indices: numpy.ndarray or NoneType
        :param recompute_every: for unsteady problems, number of time instants after which the Jacobian is recomputed.
           It defaults to 1.
        :type recompute_every: int
        :return FOM non-linear jacobian
        :rtype: numpy.ndarray
        """

        self.check_configured_fom()
        return self.M_external_engine.retrieve_fom_nonlinear_jacobian(u, self.M_fom_specifics,
                                                                      _elements=_elements,
                                                                      _indices=_indices,
                                                                      recompute_every=recompute_every)


__all__ = [
    "FomProblemUnsteady"
]
