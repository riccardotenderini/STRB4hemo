#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:22:53 2018
@author: Niccolo' Dal Santo
@email : niccolo.dalsanto@epfl.ch

"""

import os
import time
import numpy as np
import scipy.linalg

import src.rb_library.proper_orthogonal_decomposition as podec
import src.utils.general_utils as gen_utils
import src.utils.array_utils as arr_utils
from src.utils.newton import Newton

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class RbManager:
    """Class which handles the assembling and the resolution of parametrized PDE problems using the Reduced Basis RB
    method. In particular, this class handles steady problems and the Reduced Basis is retrieved by applying a POD onto
    a set of FOM arrays, which configure as the solutions to the parametric FOM problem at hand for several distinct
    parameter values. Such solutions are furthermore computed interfacing to another library, which could be either the
    'feamat' Matlab library (https:\\github.com\lucapegolotti\feamat\tree\pyorb_wrappers_parabolic) or the 'LifeV' C++
    library (https:\\bitbucket.org\lifev-dev\lifev-release\wiki\Home) in the context of this project.
    """

    def __init__(self, _fom_problem, _affine_decomposition=None):
        """Initialization of the RbManager class

        :param _fom_problem: FOM problem at hand
        :type _fom_problem: FomProblem
        :param _affine_decomposition: AffineDecompositionHandler object, used to handle the affine decomposition of the
            parametric FOM problem at hand with respect to the characteristic parameters. Defaults to None.
        :type _affine_decomposition: AffineDecompositionHandler or NoneType
        """

        self.M_verbose = False
        self.M_get_test = False

        self.M_Nh = _fom_problem.get_fom_dimension()
        self.M_ns = 0
        self.M_snapshots_matrix = np.zeros([])
        self.M_offline_ns_parameters = np.zeros([])
        self.M_ns_test = 0
        self.M_test_snapshots_matrix = np.zeros([])
        self.M_test_offline_ns_parameters = np.zeros([])

        self.M_nl_term_snapshot_matrix = np.zeros([])
        self.M_nl_term_jac_snapshot_matrix = np.zeros([])

        self.M_N = 0
        self.M_basis = np.zeros(0)
        self.M_gen_coords_path = np.zeros(0)

        self.M_fom_problem = None
        self.M_affine_decomposition = None

        self.M_import_snapshots = False
        self.M_import_offline_structures = False
        self.M_save_offline_structures = False
        self.M_save_results = False

        self.M_basis_path = ""
        self.M_snapshots_path = ""
        self.M_affine_components_path = ""
        self.M_parameters_path = ""
        self.M_results_path = ""
        self.M_test_snapshots_path = ""
        self.M_test_parameters_path = ""

        self.M_nl_term_snapshot_path = ""
        self.M_nl_term_jac_snapshot_path = ""
        self.M_nl_term_affine_components_path = ""
        self.M_nl_term_jac_affine_components_path = ""

        self.M_mesh_name = ""

        self.M_used_Qa = 0
        self.M_used_Qf = 0
        self.M_An = np.zeros((0, 0))
        self.M_fn = np.zeros(0)
        self.M_un = np.zeros(0)
        self.M_utildeh = np.zeros(0)
        self.M_solver_converged = True

        if _affine_decomposition is not None:
            self.affine_decomposition = _affine_decomposition
        self.fom_problem = _fom_problem

        self.M_nonlinear_jacobian = not self.M_fom_problem.is_linear()

        return

    @property
    def affine_decomposition(self):
        return self.M_affine_decomposition

    @affine_decomposition.setter
    def affine_decomposition(self, _affineDecomposition):
        """Setter method, which allows to set the AffineDecompositionHandler class attribute, if it has not already been
        done in :func:`~rb_manager.RbManager.__init__`

        :param _affineDecomposition: AffineDecompositionHandler object
        :type _affineDecomposition: AffineDecompositionHandler
        """

        self.M_affine_decomposition = _affineDecomposition
        return

    @property
    def fom_problem(self):
        return self.M_fom_problem

    @fom_problem.setter
    def fom_problem(self, _fom_problem):
        """Setter method, which allows to set the FomProblem class attribute.

        :param _fom_problem: FOM problem at hand
        :type _fom_problem: FomProblem
        """

        self.M_fom_problem = _fom_problem
        return

    @property
    def parameter_handler(self):
        return self.M_fom_problem.parameter_handler

    @property
    def mesh_name(self):
        return self.M_mesh_name

    def import_snapshots_parameters(self, _ns=None):
        """Method which allows to import the parameters corresponding to the snapshots that are used to compute the
        Reduced Basis. The parameters are imported from an _input_file, provided that it represents a valid path, and it
        is as well possible to import just a subset of the parameters, by setting the input argument '_ns' to a value
        which is smaller than the total number of parameters stored at the target file.

        :param _ns: number of parameters which have to be imported. If None or if higher than the total number of
            parameters available, then all the parameters are imported. Defaults to None.
        :type _ns: int or NoneType
        :return: True if the importing has been successful, False otherwise
        :rtype: bool
        """

        assert self.M_import_snapshots, "Snapshots import disabled"

        try:
            self.M_offline_ns_parameters = np.loadtxt(self.M_parameters_path)
            if len(self.M_offline_ns_parameters.shape) == 1:
                self.M_offline_ns_parameters = self.M_offline_ns_parameters[None]
            if _ns is not None and self.M_offline_ns_parameters.shape[0] > _ns:
                self.M_offline_ns_parameters = self.M_offline_ns_parameters[:_ns]
            import_success = True

        except (IOError, OSError, FileNotFoundError) as e:
            logger.error(f"Error {e}: impossible to load the snapshots parameters")
            import_success = False

        return import_success

    def import_snapshots_matrix(self, _ns=None):
        """Method which allows to import the FOM snapshots (i.e. FOM solutions to the problem at hand, corresponding
        to different parameter values) that are later used to compute the Reduced Basis. The snapshots are imported
        from an _input_file, provided that it represents a valid path, and it is as well possible to import just a
        subset of the snapshots, by setting the input argument '_ns' to a value which is smaller than the total number
        of snapshots stored at the target file.

        :param _ns: number of snapshots which have to be imported. If None or if higher than the total number of
            snapshots available, then all the snapshots are imported. Defaults to None.
        :type _ns: int or NoneType
        :return: True if the importing has been successful, False otherwise
        :rtype: bool
        """

        assert self.M_import_snapshots, "Snapshots import disabled"

        try:
            self.M_snapshots_matrix = np.load(self.M_snapshots_path)
            if len(self.M_snapshots_matrix.shape) == 1:
                self.M_snapshots_matrix = self.M_snapshots_matrix[:, None]
            self.M_Nh = self.M_snapshots_matrix.shape[0]
            self.M_ns = self.M_snapshots_matrix.shape[1]
            if _ns is not None and _ns < self.M_ns:
                self.M_snapshots_matrix = self.M_snapshots_matrix[:, :_ns]
                self.M_ns = _ns
            import_success = True

        except (IOError, OSError, FileNotFoundError) as e:
            logger.error(f"Error {e}: impossible to load the snapshots matrix")
            import_success = False

        return import_success

    def import_snapshots(self):
        """Method which allow to import both the FOM snapshots and the corresponding parameter values for given input
        files, provided that they represent valid paths. It is basically a wrapper around the functions:

            * :func:`~rb_manager.RbManager.import_snapshots_matrix`
            * :func:`~rb_manager.RbManager.import_snapshots_parameters`

        :return: True if the importing has been successful, False otherwise
        :rtype: bool
        """

        assert self.M_import_snapshots, "Snapshots import disabled"

        import_success_snapshots = self.import_snapshots_matrix()
        import_success_parameters = self.import_snapshots_parameters()

        return bool(import_success_parameters and import_success_snapshots)

    def import_basis_matrix(self,):
        """Method which allows to import the matrix encoding the Reduced Basis, which has been constructed
        performing a POD over a certain set of snapshots, computed as solutions to the parametric FOM problem
        for several distinct parameter values.

        :return: True if the importing has been successful, False otherwise
        :rtype: bool
        """

        assert self.M_import_offline_structures, "Offline structures import is disabled"

        try:
            self.M_basis = np.load(self.M_basis_path)
            self.M_Nh = self.M_basis.shape[0]
            self.M_N = self.M_basis.shape[1]
            import_success = True

        except (IOError, OSError, FileNotFoundError) as e:
            logger.error(f"Error {e}: impossible to load the basis matrix")
            import_success = False

        return import_success

    @property
    def get_test(self):
        return self.M_get_test

    @get_test.setter
    def get_test(self, _get_test):
        self.M_get_test = _get_test

    def import_test_parameters(self, _ns=None):
        """Method which allows to import the parameters corresponding to the snapshots that are used as test dataset in
        Deep Learning applications. The parameters are imported from an _input_file, provided that it represents a valid
        path.

        :return: True if the importing has been successful, False otherwise
        :rtype: bool
        """

        try:
            self.M_test_offline_ns_parameters = np.loadtxt(self.M_test_parameters_path)
            if len(self.M_test_offline_ns_parameters.shape) == 1:
                self.M_test_offline_ns_parameters = self.M_test_offline_ns_parameters[None]
            if _ns is not None and self.M_test_offline_ns_parameters.shape[0] > _ns:
                self.M_test_offline_ns_parameters = self.M_test_offline_ns_parameters[:_ns]
            import_success = True
        except (IOError, OSError, FileNotFoundError, TypeError) as e:
            logger.error(f"Error {e}: impossible to load the test snapshots parameters")
            import_success = False

        return import_success

    def import_test_snapshots_matrix(self, _ns=None):
        """Method which allows to import the FOM snapshots (i.e. FOM solutions to the problem at hand, corresponding
        to different parameter values) that are later used as test dataset in Deep Learning applications. The snapshots
        are imported from an _input_file, provided that it represents a valid path.

        :return: True if the importing has been successful, False otherwise
        :rtype: bool
        """

        try:
            self.M_test_snapshots_matrix = np.load(self.M_test_snapshots_path)
            if len(self.M_test_snapshots_matrix.shape) == 1:
                self.M_test_snapshots_matrix = self.M_test_snapshots_matrix[:, None]
            if _ns is not None and self.M_test_snapshots_matrix.shape[1] > _ns:
                self.M_test_snapshots_matrix = self.M_test_snapshots_matrix[:, :_ns]
            self.M_ns_test = self.M_test_snapshots_matrix.shape[1]
            import_success = True

        except (IOError, OSError, FileNotFoundError, TypeError) as e:
            logger.error(f"Error {e}: impossible to load the test snapshots matrix")
            import_success = False

        return import_success

    def import_rb_affine_components(self):
        """Method which allows to import the RB affine components for the stiffness matrix A and for the right-hand side
        vector f; the importing is from the file whose path is given as input, provided that it represents a valid path.
        The RB affine components are computed by:

            * Pre-multiplying vectors by the transpose of the Reduced Basis matrix
            * Pre-multiplying matrices by the transpose of the Reduced Basis matrix and post-multiplying such result by
              the Reduced Basis matrix

        :return: set containing the name of the operators whose import of the RB affine components has failed
        :rtype: set or NoneType
        """

        assert self.M_import_offline_structures, "Offline structures import is disabled"

        return self.M_affine_decomposition.import_rb_affine_components(self.M_affine_components_path)

    def import_offline_structures(self):
        """Method which allows to import the most relevant offline structures needed in the resolution of the parametric
        problem at hand, using the RB method, i.e.

            * parameters
            * snapshots
            * Reduced Basis matrix
            * affine components, projected onto the Reduced Basis space

        """

        assert self.M_import_offline_structures, "Offline structures import is disabled"

        self.import_snapshots_parameters()
        self.import_snapshots_matrix()
        self.import_basis_matrix()
        self.M_affine_decomposition.import_affine_components()

        return

    def get_offline_parameter(self, _iP):
        """Getter method, which returns the parameter corresponding to the snapshot with index '_iP'

        :param _iP: index of the desired parameter
        :type _iP: int
        :return: value of the parameter with index '_iP'
        :rtype: numpy.ndarray
        """
        assert _iP >= 0
        return self.M_offline_ns_parameters[_iP, :]

    @property
    def offline_parameters(self):
        """Getter method, which allows to get all the parameters corresponding to the stored snapshots

        :return: values of all the parameters
        :rtype: numpy.ndarray
        """
        return self.M_offline_ns_parameters

    def get_test_parameter(self, _iP):
        """Getter method, which returns the parameter corresponding to the test snapshot with index '_iP'

        :param _iP: index of the desired parameter
        :type _iP: int
        :return: value of the parameter with index '_iP'
        :rtype: numpy.ndarray
        """

        assert _iP >= 0
        return self.M_test_offline_ns_parameters[_iP, :]

    @property
    def test_parameters(self):
        """Getter method, which allows to get all the parameters corresponding to the stored test snapshots

        :return: values of all the test parameters
        :rtype: numpy.ndarray
        """

        return self.M_test_offline_ns_parameters

    def get_number_of_snapshots(self, is_test=False):
        """Getter function, which returns the number of train/test snapshots

        :param is_test: if True, the number of test snapshots is returned. Otherwise, the number of train ones.
           It defaults to False
        type is_test: bool
        :return: number of train/test snapshots
        :rtype: int
        """

        return self.M_ns_test if is_test else self.M_ns

    @property
    def snapshots_matrix(self):
        return self.M_snapshots_matrix

    def get_snapshots_matrix(self, _fom_coordinates=np.array([])):
        """Getter method, which returns the whole set of snapshots. If the input argument '_fom_coordinates' is not
        an empty array, the evaluation of the snapshots is restricted to the FOM dofs whose index is present in
        '_fom_coordinates'. If no snapshot is stored, ir raises a ValueError.

        :param _fom_coordinates: index of the FOM dofs at which the evaluation of the snapshots is desired. If empty,
            the evaluation occurs at all the FOM dofs. Defaults to an empty numpy array
        :type _fom_coordinates: numpy.ndarray
        :return: matrix of the snapshots, eventually evaluated in a subset of the FOM dofs
        :rtype: numpy.ndarray
        """

        try:
            assert self.M_snapshots_matrix.shape[0] > 0
        except AssertionError:
            logger.critical("Impossible to get the snapshots. You need to construct or import them before")
            raise ValueError

        if not _fom_coordinates.shape[0]:
            return self.M_snapshots_matrix
        else:
            return self.M_snapshots_matrix[_fom_coordinates.astype(int), :]

    @property
    def test_snapshots_matrix(self):
        return self.M_test_snapshots_matrix

    def get_test_snapshots_matrix(self, _fom_coordinates=np.array([])):
        """Getter method, which returns the whole set of test snapshots. If the input argument '_fom_coordinates' is not
        an empty array, the evaluation of the test snapshots is restricted to the FOM dofs whose index is present in
        '_fom_coordinates'. If no test snapshot is stored, ir raises a ValueError.

        :param _fom_coordinates: index of the FOM dofs at which the evaluation of the test snapshots is desired. If
            empty, the evaluation occurs at all the FOM dofs
        :type _fom_coordinates: numpy.ndarray
        :return: matrix of the test snapshots, eventually evaluated in a subset of the FOM dofs
        :rtype: numpy.ndarray
        """

        try:
            assert self.M_test_snapshots_matrix.shape[0] > 0
        except AssertionError:
            logger.critical("Impossible to get the test snapshots. You need to construct or import them before")
            raise ValueError

        if not _fom_coordinates.shape[0]:
            return self.M_test_snapshots_matrix
        else:
            return self.M_test_snapshots_matrix[_fom_coordinates.astype(int), :]

    def get_snapshot(self, _snapshot_number, _fom_coordinates=np.array([])):
        """Getter method, which returns the snapshot with index '_snapshot_number'. If the input argument
        '_fom_coordinates' is not an empty array, the evaluation of the snapshot is restricted to the FOM dofs
        whose index is present in '_fom_coordinates'. If no snapshot is stored or if the given index is not valid,
        it raises a ValueError.

        :param _snapshot_number: index of the desired snapshot
        :type _snapshot_number: int
        :param _fom_coordinates: index of the FOM dofs at which the evaluation of the snapshot is desired. If
          empty, the evaluation occurs at all the FOM dofs. Defaults to an empty numpy array
        :type _fom_coordinates: numpy.ndarray
        :return: desired snapshot, eventually evaluated in a subset of the FOM dofs
        :rtype: numpy.ndarray
        """

        try:
            assert self.M_snapshots_matrix.shape[0] > 0
        except AssertionError:
            logger.critical("Impossible to get the snapshots. You need to construct or import them before")
            raise ValueError

        try:
            assert 0 <= _snapshot_number < self.M_snapshots_matrix.shape[1]
        except AssertionError:
            logger.critical(f"{_snapshot_number} is not a valid index. Number of snapshots stored: "
                            f"{self.M_snapshots_matrix.shape[1]}")
            raise ValueError

        if not _fom_coordinates.shape[0]:
            return self.M_snapshots_matrix[:, _snapshot_number]
        else:
            return self.M_snapshots_matrix[_fom_coordinates.astype(int), _snapshot_number]

    def get_test_snapshot(self, _snapshot_number, _fom_coordinates=np.array([])):
        """Getter method, which returns the test snapshot with index '_snapshot_number'. If the input argument
        '_fom_coordinates' is not an empty array, the evaluation of the test snapshot is restricted to the FOM dofs
        whose index is present in '_fom_coordinates'. If no test snapshot is stored or if the given index is not valid,
        it raises a ValueError.

        :param _snapshot_number: index of the desired test snapshot
        :type _snapshot_number: int
        :param _fom_coordinates: index of the FOM dofs at which the evaluation of the test snapshot is desired. If
          empty, the evaluation occurs at all the FOM dofs. Defaults to an empty numpy array
        :type _fom_coordinates: numpy.ndarray
        :return: desired test snapshot, eventually evaluated in a subset of the FOM dofs
        :rtype: numpy.ndarray
        """

        try:
            assert self.M_test_snapshots_matrix.shape[0] > 0
        except AssertionError:
            logger.critical("Impossible to get the test snapshots. You need to construct or import them before")
            raise ValueError

        try:
            assert 0 <= _snapshot_number < self.M_test_snapshots_matrix.shape[1]
        except AssertionError:
            logger.critical(f"{_snapshot_number} is not a valid index. Number of test snapshots stored: "
                            f"{self.M_test_snapshots_matrix.shape[1]}")
            raise ValueError

        if not _fom_coordinates.shape[0]:
            return self.M_test_snapshots_matrix[:, _snapshot_number]
        else:
            return self.M_test_snapshots_matrix[_fom_coordinates.astype(int), _snapshot_number]

    def get_snapshot_function(self, _snapshot_number, _fom_coordinates=np.array([])):
        """Getter method, which allows to get either the test or the train snapshot corresponding to the index
        '_snapshot_number', depending on the value of the class attribute flag self.M_get_test. It is basically a
        wrapper around :func:`~rb_manager.RbManager.get_snapshot` and :func:`~rb_manager.RbManager.get_test_snapshot`.
        Additionally, the snapshots can be evaluated in a subset of the dofs, if the input argument '_fom_coordinates'
        is not an empty numpy array.

        :param _snapshot_number: index of the desired (train or test) snapshot
        :type _snapshot_number: int
        :param _fom_coordinates: index of the FOM dofs at which the evaluation of the (test) snapshot is desired. If
          empty, the evaluation occurs at all the FOM dofs. Defaults to an empty numpy array
        :type _fom_coordinates: numpy.ndarray
        :return: desired (train or test) snapshot, eventually evaluated in a subset of the FOM dofs
        :rtype: numpy.ndarray
        """
        if self.M_get_test:
            return self.get_test_snapshot(_snapshot_number, _fom_coordinates=_fom_coordinates)
        else:
            return self.get_snapshot(_snapshot_number, _fom_coordinates=_fom_coordinates)

    def get_parameters(self, _snapshot_number, num_parameters=None):
        """Method that returns the array of the parameter value corresponding to the given snapshot number. Depending on
        the class attribute flag self.M_get_test, either the training or the testing parameter values are returned.

        :param _snapshot_number: index of the desired (train or test) snapshot
        :type _snapshot_number: int
        :param num_parameters: number of parameters to be returned. If None, all parameters are returned.
            It defaults to None.
        :type num_parameters: int or NoneType
        :return: desired (train or test) parameter values corresponding to the given snapshot
        :rtype: numpy.ndarray
        """
        if self.M_get_test:
            if 0 <= _snapshot_number < self.M_test_offline_ns_parameters.shape[0]:
                return self.M_test_offline_ns_parameters[_snapshot_number, :num_parameters]
            else:
                logger.critical(f"{_snapshot_number} is not a valid index. Number of test snapshots stored: "
                                f"{self.M_test_offline_ns_parameters.shape[0]}")
                raise ValueError
        else:
            if 0 <= _snapshot_number < self.M_offline_ns_parameters.shape[0]:
                return self.M_offline_ns_parameters[_snapshot_number, :num_parameters]
            else:
                logger.critical(f"{_snapshot_number} is not a valid index. Number of train snapshots stored: "
                                f"{self.M_offline_ns_parameters.shape[0]}")
                raise ValueError

    @property
    def basis(self):
        return self.M_basis

    def get_basis(self, _fom_coordinates=np.array([])):
        """Getter method, which allows to get the matrix encoding the Reduced Basis. If the input argument
        '_fom_coordinates' is not an empty array, the evaluation of the reduced basis is restricted to the FOM dofs
        whose index is present in '_fom_coordinates'.

        :param _fom_coordinates: index of the FOM dofs at which the evaluation of the Reduced Basis is desired. If
          empty, the evaluation occurs at all the FOM dofs
        :type _fom_coordinates: numpy.ndarray
        :return: Reduced Basis matrix, eventually evaluated in a subset of the FOM dofs
        :rtype: numpy.ndarray
        """
        if not _fom_coordinates.shape[0]:
            return self.M_basis
        else:
            return self.M_basis[_fom_coordinates.astype(int), :]

    @property
    def N(self):
        """Getter method, which allows to get the dimensionality of the Reduced Basis

        :return: dimensionality of the Reduced Basis
        :rtype: int or dict
        """
        return self.M_N

    @property
    def N_fom(self):
        """Getter method, which allows to get the number of dofs of the FOM problem at hand

        :return: number of dofs of the FOM problem at hand
        :rtype: int or dict
        """
        return self.M_Nh

    @property
    def qa(self):
        """Getter method, which returns the number of affine components for the stiffness matrix. It is a wrapper around
        :func:`~affine_decomposition.AffineDecompositionHandler.get_Qa`

        :return: number of affine components for the stiffness matrix
        :rtype: int
        """
        return self.M_affine_decomposition.qa

    @property
    def qf(self):
        """Getter method, which returns the number of affine components for the right-hand side vector. It is a wrapper
        around :func:`~affine_decomposition.AffineDecompositionHandler.get_Qf`

        :return: number of affine components for the right-hand side vector
        :rtype: int
        """
        return self.M_affine_decomposition.qf

    def get_rb_functions_dict(self):
        """Method which returns the functions of the Reduced Basis inside a dictionary, with corresponding key defined
        as 'rb_func_{index}', being 'index' the index of the current function. Additionally, the field with key defined
        as 'range_rb_functions' stores the total number of functions in the Reduced Basis

        :return: dictionary, containing all the functions part of the Reduced Basis and the dimensionality of the basis
            itself
        :rtype: dict
        """

        rb_functions_dict = {'range_rb_functions': self.M_N}

        for iB in range(self.M_N):
            rb_functions_dict.update({'rb_func_' + str(iB): self.M_basis[:, iB]})

        return rb_functions_dict

    def get_rb_affine_matrix(self, _q):
        """Getter method, which returns the affine component of index _q among the FOM affine components of the
        stiffness matrix, projected over the RB-space. If _q exceeds the number of affine components for the stiffness
        matrix an IndexError is raised.

        :param _q: index of the desired affine component for the stiffness matrix
        :type _q: int
        :return: desired RB affine component of the stiffness matrix
        :rtype: int
        """
        return self.M_affine_decomposition.get_rb_affine_matrix(_q)

    def get_rb_affine_vector(self, _q):
        """Getter method, which returns the affine component of index _q among the FOM affine components of the
        right-hand side, projected over the RB-space. If _q exceeds the number of affine components for the right-hand
        side an IndexError is raised.

        :param _q: index of the desired affine component for the right-hand side vector
        :type _q: int
        :return: desired FOM affine component of the right-hand side vector
        :rtype: int
        """
        return self.M_affine_decomposition.get_rb_affine_vector(_q)

    def update_fom_specifics(self, _fom_specifics_update):
        """Method which allows to update the dictionary containing the specifics of the FOM problem at hand. If the
        input updating dictionary has fields with keys that are not present in the original dictionary, then such keys
        are added to the original dictionary; conversely, if the updating dictionary has keys in common with the
        original dictionary, the corresponding values of the original dictionary will be updated with the ones of the
        input dictionary.

        :param _fom_specifics_update: updating dictionary
        :type _fom_specifics_update: dict
        """

        self.M_fom_problem.update_fom_specifics(_fom_specifics_update)

        return

    def set_random_snapshots_matrix(self, _rows=4, _cols=2):
        """Method which builds a random snapshots matrix of dimension ('_rows', '_cols'), sampling from the
        standard normal distribution

        :param _rows: number of rows of the matrix, which stands for the number of FOM dofs. Defaults to 4.
        :type _rows: int
        :param _cols: number of columns of the matrix, which stand for the number of snapshots. Defaults to 2.
        :type _cols: int
        """

        self.M_snapshots_matrix = np.random.randn(_rows, _cols)

        return

    def build_snapshots(self, _new_snapshots, seed=0, extend_existing=True, prob=None, is_test=False):
        """Method which allows to construct new FOM snapshots interfacing, via the FomProblem class, to the external
        engine that has been chosen to perform the resolution of the FOM problems (either the Matlab 'feamat' library or
        the C++ 'LifeV' library). The number of snapshots is passed in input; if some snapshots are already stored, it
        is both possible to extend the existing snapshots with new ones, if the input 'extend_existing' is passed as
        True, or otherwise recompute all the snapshots from scratch. Additionally, the parameters sampling for the
        snapshots computation follows a  discrete uniform distribution in the default case, but other discrete
        distributions can be passed as well, via the input argument 'prob'. Finally, if the class attribute flag
        self.M_save_offline_structures is True, the computed snapshots and parameters are saved as .txt files; the
        saving path is known to the class if the class method :func:`~rb_manager.RbManager.save_offline_structures` has
        been previously called, with the input argument '_snapshot_matrix' defining such path.

        :param _new_snapshots: number of new desired snapshots
        :type _new_snapshots: int
        :param seed: seed for the random number generation. Defaults to 0
        :type seed: int
        :param extend_existing: if True, the new snapshots are added to the already existing ones, if any. If False, the
          snapshots are recomputed from scratch. Defaults to True
        :type extend_existing: bool
        :param prob: discrete probability distribution used for the random parameter sampling
        :type prob: list or tuple or numpy.ndarray
        :param is_test: True if the generated snapshots are to be used as test dataset, False otherwise.
           Defaults to False
        :type is_test: bool
        """

        if _new_snapshots <= 0:
            return

        if extend_existing:
            try:
                cond = self.M_snapshots_matrix.size and len(self.M_snapshots_matrix.shape) == 2 if not is_test else \
                    self.M_test_snapshots_matrix.size and len(self.M_test_snapshots_matrix.shape) == 2
                assert cond
            except AssertionError:
                extend_existing = False

        if extend_existing:
            current_snapshots_number = self.M_snapshots_matrix.shape[1] if not is_test else \
                                       self.M_test_snapshots_matrix.shape[1]
        else:
            current_snapshots_number = 0

        num_parameters = self.M_fom_problem.num_parameters
        params = np.zeros((_new_snapshots, num_parameters))
        snap_mat = np.zeros((self.M_Nh, _new_snapshots))

        if not self.M_fom_problem.is_linear():
            nl_term_mat = np.zeros_like(snap_mat)
            nl_term_jac_mat = []

        for iS in range(_new_snapshots):

            cur_seed = 201*(iS+1) + iS + current_snapshots_number + seed
            self.M_fom_problem.generate_parameter(prob=prob, seed=cur_seed)
            params[iS, :] = np.copy(self.M_fom_problem.param)

            logger.debug(f"Considering the parameter {iS + current_snapshots_number}: {params[iS, :]}")

            if self.M_fom_problem.is_linear():
                u, _ = self.M_fom_problem.solve_fom_problem(params[iS, :])
            else:
                u, _, nl_term, nl_term_jac = self.M_fom_problem.solve_fom_problem(params[iS, :],
                                                                                  _get_nonlinear_terms=True)
                nl_term_mat[:, iS] = nl_term
                nl_term_jac_mat.append(nl_term_jac)

            norm_of_snapshot = np.linalg.norm(u)
            logger.debug(f"Euclidean norm of the snapshot is {norm_of_snapshot}")
            snap_mat[:, iS] = u

        if not is_test:
            self.M_snapshots_matrix = snap_mat if not extend_existing else \
                                      np.append(self.M_snapshots_matrix, snap_mat, axis=1)
            self.M_offline_ns_parameters = params if not extend_existing else \
                                           np.append(self.M_offline_ns_parameters, params, axis=0)
            self.M_ns = current_snapshots_number + _new_snapshots
            logger.info(f"Current snapshots number: {self.M_ns}")
            if not self.M_fom_problem.is_linear():
                self.M_nl_term_snapshot_matrix = nl_term_mat if not extend_existing else \
                                                 np.append(self.M_nl_term_snapshot_matrix, nl_term_mat, axis=1)
                self.M_nl_term_jac_snapshot_matrix = np.array(nl_term_jac_mat) if not extend_existing else \
                                                     np.append(self.M_nl_term_jac_snapshot_matrix,
                                                               np.array(nl_term_jac_mat), axis=2)
        else:
            self.M_test_snapshots_matrix = snap_mat if not extend_existing else \
                                           np.append(self.M_test_snapshots_matrix, snap_mat, axis=1)
            self.M_test_offline_ns_parameters = params if not extend_existing else \
                                                np.append(self.M_test_offline_ns_parameters, params, axis=0)
            self.M_ns_test = current_snapshots_number + _new_snapshots
            logger.info(f"Current test snapshots number: {self.M_ns_test}")

        self.M_Nh = snap_mat.shape[0]

        if not self.M_fom_problem.is_linear():
            self.M_nl_term_jac_snapshot_matrix = np.swapaxes(self.M_nl_term_jac_snapshot_matrix, 0, 1)

        if self.M_save_offline_structures:
            snap_mat = self.M_snapshots_matrix if not is_test else self.M_test_snapshots_matrix
            snap_path = self.M_snapshots_path if not is_test else self.M_test_snapshots_path
            arr_utils.save_array(snap_mat, snap_path)

            params_mat = self.M_offline_ns_parameters if not is_test else self.M_test_offline_ns_parameters
            params_path = self.M_parameters_path if not is_test else self.M_test_parameters_path
            arr_utils.save_array(params_mat, params_path)

            if not is_test and not self.M_fom_problem.is_linear():
                # saving in .npy as there is no need to have a human-readable format!!
                np.save(self.M_nl_term_snapshot_path, self.M_nl_term_snapshot_matrix)
                np.save(self.M_nl_term_jac_snapshot_path, self.M_nl_term_jac_snapshot_matrix)

        return

    def perform_pod(self, _tol=1e-4):
        """Method which allows to perform the POD of the snapshots' matrix, so that the matrix encoding the Reduced
        Basis can be computed. Basically it is a wrapper around the __call__ method of the ProperOrthogonalDecomposition
        class. Finally, if the class attribute flag self.M_save_offline_structures is True, the computed reduced basis
        is saved as a .txt file; the saving path is known to the class if the class method
        :func:`~rb_manager.RbManager.save_offline_structures` has been previously called, via its input argument
        '_basis_matrix'

        :param _tol: tolerance for the relative energy of the already scanned singular values with respect to the energy
          of the full set of singular values. Energy is interpreted as the squared l2-norm of the singular values.
          The actual tolerance used in the algorithm is tolerance = 1.0 - _tol, thus '_tol' is intended to be close
          to 0. Defaults to 1e-4
        :type _tol: float
        """

        pod = podec.ProperOrthogonalDecomposition()

        self.M_basis = pod(self.M_snapshots_matrix, _tol)
        self.M_N = self.M_basis.shape[1]

        if self.M_save_offline_structures:
            arr_utils.save_array(self.M_basis, self.M_basis_path)

        return

    def print_rb_offline_summary(self):
        """Printing method, which prints useful info about the state of the RbManager class instance. In particular, it
        shows:

            * the number of snapshots
            * the dimensionality of the basis
            * the number of affine components of the stiffness matrix and of the right-hand side vector

        """

        logger.info(f"\n------------- RB SUMMARY -------------\n"
                    f"Number of snapshots: {self.M_ns}\n"
                    f"Number of selected RB functions: {self.M_N}")

        self.M_affine_decomposition.print_ad_summary()
        print('\n')
        return

    def print_rb_affine_components(self):
        """Method to print the RB affine components for both the stiffness matrix and the right-hand side vector
        """

        self.M_affine_decomposition.print_rb_affine_components()
        return

    def reset_rb_approximation(self):
        """Method to reset the RB affine components for both the stiffness matrix and the right-hand side vector by
        redefining the lists containing such elements as empty lists. Additionally, the basis and its dimensionality
        are also set to an empty matrix and to 0 respectively.
        """

        logger.debug("Resetting RB approximation")

        self.M_N = 0
        self.M_basis = np.zeros(0)

        self.M_affine_decomposition.reset_rb_approximation()

        return

    def build_rb_affine_decompositions(self, operators=None):
        """Method which constructs the RB affine components for the operators passed in input, suitably multiplying the
        considered affine component by the matrix encoding the reduced basis; in particular vectors are pre-multiplied
        by the transpose of such matrix, while matrices are pre-multiplied by the transpose basis matrix and
        post-multiplied by the basis matrix itself. If no operator is passed, the RB affine arrays are constructed for
        the stiffness matrix, the mass matrix and the right-hand side vector. If the FOM affine arrays are not yet
        stored in the AffineDecompositionHandler class attribute they are constructed via a call to
        :func:`~affine_decomposition.AffineDecompositionHandler.import_fom_affine_arrays`.

        :param operators: operators for which the RB affine components have to be constructed. Admissible values are 'A'
          for the stiffness matrix, 'M' for the mass matrix and 'f' for the right-hand side vector. Defaults to None.
        :type operators: set or NoneType
        """

        if operators is None:
            operators = {'A', 'f', 'M'}
        self.M_affine_decomposition.build_rb_affine_decompositions(self.M_basis, self.M_fom_problem,
                                                                   operators=operators)
        return

    def save_rb_affine_decomposition(self, operators=None):
        """Method which saves the RB affine components of the operators passed in input to text files, whose path has
        been specified by the input argument '_file_name'. If no operator is passed, the RB affine arrays are saved for
        both the stiffness matrix and the right-hand side vector. The final file name is constructed by adding to the
        input file name the operator name and the index of the affine components which is currently saved.

        :param operators: operators for which the RB affine components have to be saved. Admissible values are 'A'
          for the stiffness matrix, 'M' for the mass matrix and 'f' for the right-hand side vector. Defaults to None
        :type operators: set or NoneType
        """

        self.M_affine_decomposition.save_rb_affine_decomposition(self.M_affine_components_path, operators=operators)
        return

    def build_nonlinear_rb_affine_components(self):
        """
        MODIFY
        """

        self.M_affine_decomposition.build_nonlinear_rb_affine_components(self.M_basis)

        if self.M_save_offline_structures:
            self.M_affine_decomposition.save_nonlinear_rb_affine_decompositions(self.M_nl_term_affine_components_path,
                                                                                self.M_nl_term_jac_affine_components_path)

        return

    def build_rb_approximation(self, _ns, _tol=1e-4, prob=None):
        """Method which allows to build the RB approximation of the affine components of the stiffness matrix, of the
         mass matrix and of the right-hand side vector, which are needed to solve the RB system. In particular, the
         method is basically divided into 3 steps, where at each step it is possible either to import the desired
         quantities from ``.txt`` files (provided that a valid path to such files is given in input) or to construct the
         quantities from scratch. Such steps are:

            * importing or construction of the snapshots and of the corresponding parameters. If the number of the
              desired snapshots exceeds the number of the already stored ones, additional snapshots are anyway computed
            * importing or construction of the Reduced Basis
            * importing or construction of the RB approximation of the affine components for the stiffness matrix, for
              the mass matrix and for the right-hand side vector

        Notice that, while the importing of the Reduced Basis and of the RB-projected affine components is compulsory
        for the online execution of the RB method, the one of the snapshots it is not and it is also memory demanding;
        because of this, the method supports the possibility of importing all the quantities but the snapshots and
        proceeds to the building of new snapshots only if some of the other quantities have not been successfully
        loaded. Finally, if a quantity is constructed from scratch and the class attribute flag
        'self.M_save_offline_structures' is set to True, then such quantity is saved at a ``.txt`` file; the saving path
        is known to the class if the class method :func:`~rb_manager.RbManager.save_offline_structures` has been
        previously called via a proper input argument.

        :param _ns: number of snapshots
        :type _ns: int
        :param _tol: tolerance for the relative energy of the already scanned singular values with respect to the energy
          of the full set of singular values. Energy is interpreted as the squared l2-norm of the singular values.
          The actual tolerance used in the algorithm is tolerance = 1.0 - _tol, thus '_tol' is intended to be close
          to 0. Defaults to 1e-4
        :type _tol: float
        :param prob: discrete probability distribution used for the random parameter sampling. If None, a uniform
          distribution is used. Defaults to None
        :type prob: list or tuple or numpy.ndarray or NoneType
        """

        self.reset_rb_approximation()

        logger.info('Building RB approximation with %d snapshots and a tolerance %f' % (_ns, _tol))

        if self.M_import_snapshots:
            logger.info('Importing stored snapshots')
            import_success_snapshots = self.import_snapshots_matrix(_ns)
        else:
            import_success_snapshots = False
        if self.M_import_snapshots:
            logger.info('Importing stored parameters')
            import_success_parameters = self.import_snapshots_parameters(_ns)
        else:
            import_success_parameters = False

        if self.M_import_offline_structures:
            logger.info('Importing basis matrix')
            import_success_basis = self.import_basis_matrix()
        else:
            import_success_basis = False

        if (not import_success_snapshots or not import_success_parameters) and not import_success_basis:
            if _ns - self.M_ns > 0:
                logger.info("Snapshots importing has failed! We need to construct them from scratch")
                self.build_snapshots(_ns - self.M_ns, prob=prob)
                built_new_snapshots = True
        elif import_success_parameters and import_success_snapshots and self.M_ns < _ns:
            logger.info('We miss some snapshots! I have only %d in memory and I need to compute %d more.'
                        % (self.M_ns, _ns - self.M_ns))
            self.build_snapshots(_ns - self.M_ns, prob=prob)
            built_new_snapshots = True
        else:
            built_new_snapshots = False

        if not import_success_basis:
            self.perform_pod(_tol=_tol)

        if self.M_import_offline_structures and import_success_basis:
            logger.info('Importing RB affine components')
            import_failures_set = self.import_rb_affine_components()
            if import_failures_set:
                logger.info('Building RB affine components whose import has failed')
                self.build_rb_affine_decompositions(operators=import_failures_set)
        else:
            logger.info('Building all RB affine components')
            import_failures_set = {'A', 'f', 'M'}
            self.build_rb_affine_decompositions(operators=import_failures_set)

        if self.M_save_offline_structures and import_failures_set:
            self.save_rb_affine_decomposition()
        return

    def set_import_and_save(self, _import_snapshots, _import_offline,
                            _save_offline, _save_results):
        """
        MODIFY
        """

        self.M_import_snapshots = _import_snapshots
        self.M_import_offline_structures = _import_offline

        self.M_save_offline_structures = _save_offline
        self.M_save_results = _save_results

        return

    @property
    def do_import_snapshots(self):
        return self.M_import_snapshots

    @do_import_snapshots.setter
    def do_import_snapshots(self, _do_import_snapshots):
        self.M_import_snapshots = _do_import_snapshots
        return

    @property
    def save_results(self):
        return self.M_save_results

    @save_results.setter
    def save_results(self, _save_results):
        self.M_save_results = _save_results
        return

    def set_paths(self, _snapshot_matrix=None, _basis_matrix=None, _affine_components=None,
                  _offline_parameters=None, _generalized_coords=None, _results=None):
        """Method which allows to initialize the paths where to load/save the most relevant quantities needed
        in the resolution of the parametric problem at hand via the RB method; such quantities are:

            * snapshots
            * parameters
            * Reduced Basis matrix
            * affine components, projected onto the Reduced Basis space
            * solution to the problem for some test parameter values, projected onto the Reduced Basis space
            * solution to the problem for some test parameter values in the FOM space

        If the paths are passed as None, the corresponding quantities are not saved throughout the execution of the code

        :param _snapshot_matrix: path of the snapshots. Defaults to None
        :type _snapshot_matrix: str or NoneType
        :param _basis_matrix: path of the Reduced Basis. Defaults to None
        :type _basis_matrix: str or NoneType
        :param _affine_components: path of the affine components, projected onto the RB space. Defaults to None
        :type _affine_components: str or NoneType
        :param _offline_parameters: path of the parameters. Defaults to None
        :type: str or NoneType
        :param _results: path to the results folder. Defaults to None
        :type _results: str or NoneType
        """

        self.M_basis_path = _basis_matrix
        self.M_snapshots_path = _snapshot_matrix
        self.M_affine_components_path = _affine_components
        self.M_parameters_path = _offline_parameters
        self.M_gen_coords_path = _generalized_coords

        self.M_results_path = _results

        return

    def set_test_paths(self, _snapshot_matrix=None, _offline_parameters=None):
        """
        Set test data paths
        """

        self.M_test_snapshots_path = _snapshot_matrix
        self.M_test_parameters_path = _offline_parameters

        return

    def compute_theta_functions(self, _params):
        """Method which allows to evaluate the scalar parameter-dependent functions arising from the affine
        decomposition of the stiffness matrix and of the right-hand side vector of the FOM problem at hand, given a
        list of parameter values. The shape of such functions depends uniquely on the expression of the FOM problem
        at hand.

        :param _params: values of the parameters
        :type _params: numpy.ndarray
        :return: evaluation of the parameter-dependent functions of both A and f and the input parameter values
        :rtype: numpy.ndarray
        """

        Qa = self.qa
        Qf = self.qf

        return self.M_fom_problem.compute_theta_functions(_params, Qa, Qf)

    def build_reduced_problem(self, _param):
        """Method which assembles the left-hand side matrix and the right-hand side vector of the reduced problem, i.e.
        the problem whose solution is a vector in the RB space. Since, at least in principle, both the stiffness matrix
        and the right-hand side vector show a parametric dependency, the value of the parameter must be passed in input,
        so that both operators can be assembled correctly. In particular, the construction procedure assumes an affine
        decomposability of both A and f with respect to the parameters; if not so, DEIM-MDEIM techniques have to be
        adopted, in order to get an approximate affine decomposability.

        :param _param: value of the parameter
        :type _param: numpy.ndarray
        """

        N = self.M_N
        self.M_An = np.zeros((N, N))
        self.M_fn = np.zeros(N)
        self.M_un = np.zeros(N)

        theta_a = self.M_fom_problem.get_full_theta_a(_param)
        for iQa in range(self.M_used_Qa):
            self.M_An += theta_a[iQa] * self.get_rb_affine_matrix(iQa)

        theta_f = self.M_fom_problem.get_full_theta_f(_param)
        for iQf in range(self.M_used_Qf):
            self.M_fn += theta_f[iQf] * self.get_rb_affine_vector(iQf)

        return

    def solve_reduced_problem(self, _param, _used_Qa=0, _used_Qf=0):
        """Method which allows to solve the reduced problem, i.e. the problem whose solution is a vector in the RB
        space. This method, in particular, first assembles the reduced problem via a call to
        :func:`~rb_manager.RbManager.build_reduced_problem` and lately solves it in the Least-Squares sense, so that
        a solution can be provided even if the left-hand side matrix turns out to be singular.

        :param _param: value of the parameter
        :type _param: numpy.ndarray
        :param _used_Qa: number of affine components that are used to assemble the stiffness matrix. If equal to 0, all
          the components are used; values different from 0 make sense only if the stiffness matrix is not affine
          decomposable, so that it is necessary to resort to an approximate affine decomposition via the MDEIM
          algorithm. Defaults to 0
        :type _used_Qa: int
        :param _used_Qf: number of affine components that are used to assemble the right-hand side vector. If equal to
          0, all the components are used; values different from 0 make sense only if the right-hand side vector is not
          affine decomposable, so that it is necessary to resort to an approximate affine decomposition via the DEIM
          algorithm. Defaults to 0
        :type _used_Qf: int
        :return: solution to the reduced problem for the given parameter value
        :rtype: numpy.ndarray
        """

        logger.debug(f"Solving RB problem for parameter: {_param}")

        self.M_used_Qa = self.M_affine_decomposition.qa
        if _used_Qa > 0:
            self.M_used_Qa = _used_Qa

        self.M_used_Qf = self.M_affine_decomposition.qf
        if _used_Qf > 0:
            self.M_used_Qf = _used_Qf

        self.build_reduced_problem(_param)

        if self.M_fom_problem.is_linear():
            self.M_un = scipy.linalg.solve(self.M_An, self.M_fn)
        else:
            self.M_fn = np.expand_dims(self.M_fn, 1)
            my_newton = Newton(tol=1e-5, tol_abs=1e-8, max_err=1e-4, max_err_abs=1e-7, max_iter=10, jac_iter=1)

            def residual(u):
                return (self.M_An.dot(u) +
                        self.M_affine_decomposition.build_rb_nonlinear_term(u, self.M_basis, self.M_fom_problem) -
                        self.M_fn)

            def jacobian(u):
                jac = self.M_An
                if self.M_nonlinear_jacobian:
                    jac += self.M_affine_decomposition.build_rb_nonlinear_jacobian(u, self.M_basis,
                                                                                   self.M_fom_problem)
                return jac

            u0 = np.zeros((self.M_fn.shape[0], 1))

            self.M_un, self.M_solver_converged = my_newton(residual, jacobian, u0)
            self.M_un = self.M_un[:, 0]

        return self.M_un

    def reconstruct_fem_solution(self, _un, indices=None):
        """Method which reconstructs the FOM solution from the reduced one, by pre-multiplying it by the Reduced Basis
        matrix. More in general, it allows to pass from the RB space to the FOM space

        :param _un: RB vector, typically a solution to the reduced problem
        :type _un: numpy.ndarray
        :return: FOM vector, computed by extending the input RB vector onto the bigger FOM space
        :rtype: numpy.ndarray
        """

        assert _un.shape[0] == self.M_N

        spaces = np.arange(self.M_Nh) if indices is None else indices

        self.M_utildeh = self.M_basis[spaces].dot(_un)

        return

    def compute_rb_projection(self, _uh):
        """Method which computes the RB projection of a FOM vector, by pre-multiplying it by the transpose of the
        Reduced Basis matrix

        :param _uh: FOM vector, to be projected onto the RB space
        :type _uh: numpy.ndarray
        :return: RB vector, projection of the input FOM vector onto the RB space
        :rtype: numpy.ndarray
        """

        un = self.M_basis.T.dot(_uh)
        return un

    @property
    def un(self):
        """
        MODIFY
        """
        return self.M_un

    @property
    def utildeh(self):
        """Getter method, which allows to get the FOM expansion of the solution to the reduced problem for the current
        parameter value

        :return: FOM expansion of the solution to the reduced problem for the current parameter value
        :rtype: numpy.ndarray
        """
        return self.M_utildeh

    def compute_norm(self, vec, norm_types=None):
        """Method which allows to compute some different norms of a given vector vec. In particular, it allows to
        compute the l2, L2, H1 and H10 norms; the last three can readily be computed only if the input vector is a FOM
        vector, whose dimensions match with the ones of the mass matrix and of the stiffness matrix; if not so, a
        ValueError is raised. Also, the last three norms can be computed only if the mass and stiffness matrix are
        stored in the AffineDecompositionHandler class instance; if not so, the method constructs them via the external
        engine that has been initialized. Finally, if no norm type is specified, the L2 norm is computed.

        :param vec: vector whose norm(s) has(ve) to be computed
        :type vec: numpy.ndarray
        :param norm_types: type of the norms that have to be computed for the input vector 'vec'. Defaults to None.
        :type norm_types: set{str} or NoneType
        :return: norms that have been computed for the vector 'vec' in the for of a dictionary; the key represents the
          norm type, while the value represents the corresponding value of the norm.
        :rtype: dict{str: float}
        """

        if norm_types is None:
            norm_types = {'L2'}

        if "L2" in norm_types or "H1" in norm_types or "H10" in norm_types:
            return self.M_affine_decomposition.compute_norm(vec, self.M_fom_problem, norm_types=norm_types)
        else:
            return self.M_affine_decomposition.compute_norm(vec, norm_types=norm_types)

    def compute_rb_snapshots_error(self, _snapshot_number):
        """Testing method, which evaluates the behavior of the RB solver onto the same snapshots that have been used to
        construct the Reduced Basis. In particular, it solves both the FOM and the RB problem for the snapshot with
        index '_snapshot_number' (if it is a valid index!) and computes the relative error between the two in L2-norm.
        If the index is not valid it raises an IndexError.

        :param _snapshot_number: index of the selected snapshot
        :type _snapshot_number: int
        :return: L2-norm of the error on the selected snapshot, if available, else NaN
        :rtype: float or numpy.nan
        """

        goOn = True
        try:
            assert len(self.M_snapshots_matrix.shape) and len(self.M_offline_ns_parameters.shape)
        except AssertionError:
            logger.error("Impossible to compute the error over the pre-computed snapshots, "
                         "since no snapshot has been loaded\n\n")
            goOn = False

        if goOn:
            try:
                assert 0 <= _snapshot_number < self.M_offline_ns_parameters.shape[0]
            except AssertionError:
                logger.critical(f"Invalid index {_snapshot_number}. Number of stored snapshots: "
                                f"{self.M_offline_ns_parameters.shape[0]}")
                raise IndexError

            self.solve_reduced_problem(self.M_offline_ns_parameters[_snapshot_number, :])
            if self.M_solver_converged:
                self.reconstruct_fem_solution(self.M_un)

                error = self.M_utildeh
                error = error - self.M_snapshots_matrix[:, _snapshot_number]

                norm_of_error = self.compute_norm(error)["L2"] / \
                                self.compute_norm(self.M_snapshots_matrix[:, _snapshot_number])["L2"]
                logger.info("The L2-norm of the error for snapshot %d is %e" % (_snapshot_number, norm_of_error))
            else:
                norm_of_error = np.nan
        else:
            norm_of_error = np.nan

        return norm_of_error

    def compute_rb_test_snapshots_error(self, _snapshot_number):
        """Testing method, which evaluates the behavior of the RB solver onto the test snapshots, that do not correspond
         to any of the parameters that have been used to generate the Reduced Basis. In particular, it solves both the
         FOM and the RB problem for the test snapshot with index _snapshot_number (if it is a valid index!) and computes
         the relative error between the two in L2-norm. If the index is not valid it raises an IndexError.

        :param _snapshot_number: index of the selected test snapshot
        :type _snapshot_number: int
        :return: L2-norm of the error on the selected test snapshot, if available, else NaN
        :rtype: float or numpy.nan
        """

        goOn = True
        try:
            assert len(self.M_test_snapshots_matrix.shape) and len(self.M_test_offline_ns_parameters.shape)
        except AssertionError:
            logger.error("Impossible to compute the error over the pre-computed snapshots, "
                         "since no snapshot has been loaded")
            goOn = False

        if goOn:
            try:
                assert 0 <= _snapshot_number < self.M_test_offline_ns_parameters.shape[0]
            except AssertionError:
                logger.critical(f"Invalid index {_snapshot_number}. Number of stored test snapshots: "
                                f"{self.M_test_offline_ns_parameters.shape[0]}")
                raise IndexError

            self.solve_reduced_problem(self.M_test_offline_ns_parameters[_snapshot_number, :])
            if self.M_solver_converged:
                self.reconstruct_fem_solution(self.M_un)

                error = self.M_utildeh
                error = error - self.M_test_snapshots_matrix[:, _snapshot_number]

                norm_of_error = self.compute_norm(error)["L2"] / \
                                self.compute_norm(self.M_test_snapshots_matrix[:, _snapshot_number])["L2"]
                logger.info("The L2-norm of the error for TEST snapshot %d is %e" % (_snapshot_number, norm_of_error))
            else:
                norm_of_error = np.nan
        else:
            norm_of_error = np.nan

        return norm_of_error

    def test_rb_solver(self, _n_test, _noise=0.0, prob=None, make_plot=False):
        """Testing method, which evaluates the performance of the RB solver onto newly generated snapshots. Specifically,
        it generates '_n_test' new parameter values, for each of those it computes the solution of both the FOM
        problem and the RB problem and finally it calculates the relative error in L2 norm. As final result, the average
        L2-error is computed. Additionally, it is possible to add noise to the parameter values, by properly setting
        the input argument '_noise'; the added noise is a gaussian noise and the input argument identifies the SNR
        (Signal to Noise Ratio)

        :param _n_test: number of newly-generated parameters
        :type _n_test: int
        :param _noise: SNR of the noise which is added to the parameters. Defaults to 0.0
        :type _noise: float
        :param prob: discrete probability distribution used for the random parameter sampling
        :type prob: list or tuple or numpy.ndarray
        :param make_plot: if True, the solutions are plotted and saved. Defaults to False.
        :type make_plot: bool
        :return: average error in L2-norm and errors in L2-norm for all the newly-generated parameter values
        :rtype: tuple(float, list[float])
        """

        all_errors = 0.0
        all_errors_simulations = np.zeros(_n_test)

        execution_time_rb = 0
        execution_time_fem = 0
        speedup = 0
        num_converged_tests = 0

        for iP in range(_n_test):
            seed = 9001 * (iP + 1) + iP

            _ = self.M_fom_problem.generate_parameter(prob=prob, seed=seed)
            new_param = self.M_fom_problem.param

            logger.info(f"Solving for parameter {iP}: {new_param}")
            if _noise > 0:
                new_noised_param = new_param * (1 + _noise * np.random.normal(0, 1, size=new_param.shape))
                logger.debug(f"New NOISED parameter {iP}: {new_noised_param}")
            else:
                new_noised_param = new_param

            logger.debug("SOLVING THE RB PROBLEM ...")
            start_rb = time.perf_counter()
            self.solve_reduced_problem(new_noised_param)

            if self.M_solver_converged:
                execution_time_rb += (time.perf_counter() - start_rb)
                num_converged_tests += 1
                self.reconstruct_fem_solution(self.M_un)

                logger.debug("SOLVING THE FOM PROBLEM ...")
                uh, time_fem = self.M_fom_problem.solve_fom_problem(new_param)
                execution_time_fem += time_fem

                speedup += (execution_time_fem / execution_time_rb)

                error = self.M_utildeh - uh

                norm_of_error = self.compute_norm(error)['L2'] / self.compute_norm(uh)['L2']
                all_errors = all_errors + norm_of_error

                all_errors_simulations[iP] = norm_of_error
                logger.debug("The L2-norm of the error is %e \n\n" % norm_of_error)

                if self.M_save_results and self.M_results_path is not None:
                    logger.debug("SAVING THE RESULTS ...")
                    comm_path = os.path.join(self.M_results_path, os.path.normpath('test_param' + str(iP)))
                    gen_utils.create_dir(comm_path)

                    RB_path = os.path.join(comm_path, 'RB.npy')
                    arr_utils.save_array(self.M_utildeh, RB_path)

                    FOM_path = os.path.join(comm_path, 'FOM.npy')
                    arr_utils.save_array(uh, FOM_path)

                    param_path = os.path.join(comm_path, 'param.txt')
                    arr_utils.save_array(new_noised_param, param_path)

                    if make_plot:
                        self.plot_fe_solution(self.M_utildeh, uh, folder=comm_path, name="Solution")

        if num_converged_tests:
            avg_error = all_errors / num_converged_tests
            execution_time_fem /= num_converged_tests
            execution_time_rb /= num_converged_tests
            speedup /= num_converged_tests

            print('\n')
            logger.info("RB SOLVER CONVERGENCE")
            logger.info(f"Number of tests where the RB solver converged: {num_converged_tests}/{_n_test}")
            logger.info(f"RB solver convergence rate: {num_converged_tests / _n_test * 100} %")
            print('\n')

            logger.info("AVERAGE ERRORS")
            logger.info("The average L2-norm of the error is %e" % avg_error)
            print('\n')

            logger.info("AVERAGE EXECUTION TIMES AND SPEEDUP")
            logger.info("The average FEM execution time is %f s" % execution_time_fem)
            logger.info("The average RB  execution time is %f s" % execution_time_rb)
            logger.info("The average speedup FEM/RB is %f " % speedup)
            print('\n\n')
        else:
            logger.critical("None of the tests converged!")
            avg_error = np.nan

        return avg_error, all_errors_simulations

    def solver_converged(self):
        return self.M_solver_converged

    def plot_fe_solution(self, rb_solution, fe_solution=None, folder=None, name=None):
        """Method which allows to plot a FOM vector over the corresponding FE space. If two FOM vectors are passed, a
        subplot of size (1,2) is built; otherwise a single plot is realized. The plots are Matlab 'surf' plots and are
        saved in .eps format if a valid saving path is passed to the function, via the input argument 'folder'. In the
        general use-case, the first input is assumed to be the FOM expansion of an RB solution (or error), while the
        second input is intended to be the FOM solution to the same problem.

        :param rb_solution: first (and eventually only) FOM vector to be plotted over the corresponding FE space
        :type rb_solution: numpy.ndarray
        :param fe_solution: second FOM vector to be plotted over the corresponding FE space. If None, just a single plot
          is realized. Defaults to None.
        :type fe_solution: numpy.ndarray or NoneType
        :param folder: path to the directory where the plots has to be saved. Defaults to None
        :type: str or NoneType
        :param name: name of the plot to be saved in 'folder'. Defaults to None
        :type name: str or NoneType
        """

        try:
            assert rb_solution.shape[0] != self.M_N
        except AssertionError:
            logger.warning("RB-array is converted to FE-array to get the desired plot")
            rb_solution = self.M_basis.dot(rb_solution)

        if fe_solution is not None:
            try:
                assert fe_solution.shape[0] != self.M_N
            except AssertionError:
                logger.warning("RB-array is converted to FE-array to get the desired plot")
                fe_solution = self.M_basis.dot(fe_solution)

        if folder is None:
            folder = self.M_results_path

        self.M_fom_problem.plot_fe_solution(rb_solution, solution2=fe_solution, folder=folder, name=name)
        return


__all__ = [
    "RbManager"
]
