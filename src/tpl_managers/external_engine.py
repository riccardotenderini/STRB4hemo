#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 14:17:29 2018
@author: Niccolo' Dal Santo
@email : niccolo.dalsanto@epfl.ch
"""


class ExternalEngine:
    """ Abstract class which defines the external engine used to solve the FOM problems
    """

    def __init__(self, _engine_type, _library_path):
        """ Initialization of the external engine

        :param _engine_type: identificative string of the external engine (either 'matlab' or 'cpp' in this project)
        :type _engine_type: str
        :param _library_path: path to the library used to solve the FOM problems
        :type _library_path: str
        """

        self.M_engine_type = _engine_type
        self.M_library_path = _library_path
        self.M_engine = None
        return

    def start_engine(self):
        """Start the external engine
        """

        self.start_specific_engine()
        return

    def quit_engine(self):
        """ Quit the external engine
        """

        self.quit_specific_engine()
        return

    def start_specific_engine(self):
        """ Virtual method which starts the specific external engine that has been selected.
        It simply raises an Exception
        """

        raise Exception("You are using the default start_specific_engine, "
                        "please provide specific ones for your specific engine ")

    def quit_specific_engine(self):
        """ Virtual method which quits the specific external engine that has been selected.
        It simply raises an Exception
        """
        raise Exception("You are using the default quit_specific_engine, "
                        "please provide specific ones for your specific engine ")

    def convert_parameter(self, _param):
        """ Virtual method which converts the characteristic model parameters in a format which can be handled by the
        specific engine that has been chosen. It simply raises an Exception.

        :param _param: value of the parameter
        :type _param: numpy.ndarray
        """

        raise Exception("You are using the default convert_parameter, "
                        "please provide specific ones for your specific engine ")

    def convert_types(self, _param):
        """ Virtual Method that converts the FOM problem specifics, as defined in Python, in such a way that they can
        be handled and used by the Matlab\C++ FOM solver. It simply raises an Exception.

        :param _param: value of the parameter
        :type _param: numpy.ndarray
        """

        raise Exception("You are using the default convert_types, "
                        "please provide specific ones for your specific engine ")

    def get_initial_condition(self, _fom_specifics, _param):
        """ Virtual Method which returns the initial condition to the FOM problem, provided that the FOM problem is
        unsteady. If the FOM problem is steady, it just returns None. It simply raises an Exception

        :param _fom_specifics: specifics of the FOM problem at hand
        :type _fom_specifics: dict
        :param _param: value of the parameter
        :type _param: numpy.ndarray
        :return: initial condition to the FOM problem if it is unsteady, else None
        :rtype: numpy.ndarray or None
        """

        raise Exception("You are using the default get_initial_condition, "
                        "please provide a specific one for your specific engine ")

    def get_reference_sol(self, _param, _fom_specifics, timestep_nb):
        """Virtual method to get a reference solution to the FOM problem at hand. In the context of the current project,
        the computation of a reference solution is supported only for unsteady 2D problems; therefore such method
        returns None if it is called passing the specifics of a steady and\or 3D problem. It simply raises an Exception

        :param _param: value of the parameter
        :type _param: numpy.ndarray
        :param _fom_specifics: specifics of the FOM problem at hand
        :type _fom_specifics: dict
        :param timestep_nb: number of timesteps at which the computation of the reference solution is desired.
          If None, the reference solution is computed at all the timesteps. Defaults to None.
        :type timestep_nb: int or NoneType
        :return: reference solution to the FOM problem at hand, computed via the ode23t Matlab package
        :rtype: numpy.ndarray
        """

        raise Exception("You are using the default get_reference_sol, "
                        "please provide a specific one for your specific engine ")

    def get_reference_rb_sol(self, _param, _fom_specifics, rb_basis, timestep_nb):
        """Virtual method to get a reference RB solution to the FOM problem at hand. In the context of the current
        project, the computation of a reference solution is supported only for unsteady 2D problems; therefore such
        method returns None if it is called passing the specifics of a steady and\or 3D problem.
        It simply raises an Exception

        :param _param: value of the parameter
        :type _param: numpy.ndarray
        :param _fom_specifics: specifics of the FOM problem at hand
        :type _fom_specifics: dict
        :param rb_basis: Reduced Basis basis
        :type rb_basis: numpy.ndarray
        :param timestep_nb: number of timesteps at which the computation of the reference solution is desired.
          If None, the reference solution is computed at all the timesteps. Defaults to None.
        :type timestep_nb: int or NoneType
        :return: reference solution to the FOM problem at hand, computed via the ode23t Matlab package
        :rtype: numpy.ndarray
        """

        raise Exception("You are using the default get_reference_rb_sol, "
                        "please provide a specific one for your specific engine ")

    def solve_parameter(self, _param, _fom_specifics):
        """ Virtual method which provides the solution to the FOM problem, given the problem specifics and the values
        of the parameters. It simply raises an Exception.

        :param _param: value of the parameter
        :type _param: numpy.ndarray
        :param _fom_specifics: specifics of the FOM problem at hand
        :type _fom_specifics: dict
        """

        raise Exception("You are using the default solve_parameter, "
                        "please provide specific ones for your specific engine ")

    def build_fom_affine_components(self, _operator, _num_affine_components, _fom_specifics):
        """ Virtual method which provides the construction of the FOM affine components of the stiffness matrix, the
        mass matrix and the right-hand side vector. It simply raises an Exception.

        :param _operator: operator of which the computation of the affine components is desired. Admissible values are
          either 'A', 'M' or 'f'
        :type _operator: str
        :param _num_affine_components: number of the desired affine components; useful in DEIM-MDEIM approximate affine
          components computation
        :type _num_affine_components: int
        :param _fom_specifics: specifics of the FOM problem at hand
        :type _fom_specifics: dict
        :return:
        """

        raise Exception("You are using the default build_fom_affine_components, "
                        "please provide specific ones for your specific engine ")

    def assemble_fom_matrix(self, _param, _fom_specifics, _elements=None, _indices=None):
        """ Virtual method which performs the assembling of the FOM system matrices. The keyword arguments
        _elements and _indices respectively allow to evaluate the matrices over a subset of mesh elements and in a
        subset of dofs indices. It simply raises an Exception.

        :param _param: value of the parameter
        :type _param: list or numpy.ndarray
        :param _fom_specifics: specifics of the FOM problem at hand
        :type _fom_specifics: dict
        :param _elements: set of elements over which evaluating the matrices or None. Defaults to None.
        :type _elements: numpy.ndarray or None
        :param _indices: set of dofs indices at which evaluating the matrices or None. Defaults to None.
        :type _indices: numpy.ndarray
        """

        raise Exception("You are using the default assemble_fom_matrix, "
                        "please provide specific ones for your specific engine ")

    def assemble_fom_rhs(self, _param, _fom_specifics, _elements=None, _indices=None):
        """ Virtual method which performs the assembling of the FOM system right-hand side vector. The keyword arguments
        _elements and _indices respectively allow to evaluate the vector over a subset of mesh elements and in a subset
        of dofs indices. It simply raises an Exception.

        :param _param: value of the parameter
        :type _param: list or numpy.ndarray
        :param _fom_specifics: specifics of the FOM problem at hand
        :type _fom_specifics: dict
        :param _elements: set of elements over which evaluating the matrices or None. Defaults to None.
        :type _elements: numpy.ndarray or None
        :param _indices: set of dofs indices at which evaluating the matrices or None. Defaults to None.
        :type _indices: numpy.ndarray
        """

        raise Exception("You are using the default assemble_fom_rhs, "
                        "please provide specific ones for your specific engine ")

    def plot_fe_solution(self, solution, _fom_specifics, folder="", name=None):
        """Virtual method which allows to plot a FOM vector over the corresponding FE space.
        It simply raises an Exception

        :param solution: FOM vector to be plotted over the corresponding FE space
        :type solution: numpy.ndarray
        :param _fom_specifics: dictionary containing the specifics of the FOM problem at hand
        :type _fom_specifics: dict
        :param folder: path to the directory where the plots has to be saved. Defaults to ""
        :type folder: str
        :param name: name of the plot to be saved in 'folder'. Defaults to None
        :type name: str or NoneType
        """

        raise Exception("You are using the default plot_fe_solution, "
                        "please provide specific ones for your specific engine ")


__all__ = [
    "ExternalEngine"
]
