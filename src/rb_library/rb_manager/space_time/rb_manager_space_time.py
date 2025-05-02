#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 09:48:05 2019
@author: Riccardo Tenderini
@mail: riccardo.tenderini@epfl.ch
"""

import os
import numpy as np
import scipy.linalg

import src.rb_library.proper_orthogonal_decomposition as podec
import src.rb_library.rb_manager.rb_manager_unsteady as rbmu

import src.utils.array_utils as arr_utils
from src.utils.newton import Newton

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class RbManagerSpaceTime(rbmu.RbManagerUnsteady):
    """Class which handles the assembling and the resolution of unsteady parametrized PDE problems, performing
   dimensionality reduction in both space and time dimension via the employment of a ST-LSPG projection approach.
   The Space-Time Reduced Basis is  constructed as the outer product of a space basis and a time basis and it is built
   by performing two PODs (one over the space dimension and the other one over the time dimension) on the tensors
   assembled by computing different FOM solutions, for different values of the characteristic parameters. Such solutions
   are furthermore computed interfacing to another third-party library, which could be either the 'feamat' Matlab
   library (https:\\github.com\lucapegolotti\feamat\tree\pyorb_wrappers_parabolic) or the 'LifeV' C++ library
   (https:\\bitbucket.org\lifev-dev\lifev-release\wiki\Home) in the context of this project.
   It inherits from :class:`~rb_manager_unsteady.RbManagerUnsteady`
   """

    def __init__(self, _fom_problem, _affine_decomposition=None):
        """Initialization of the RbManagerSpaceTime class

        :param _fom_problem: unsteady FOM problem at hand
        :type _fom_problem: FomProblemUnsteady
        :param _affine_decomposition: AffineDecompositionHandlerSpaceTime object, used to handle the affine
           decomposition of the unsteady parametric FOM problem at hand with respect to the characteristic parameters,
           according to the ST-LSPG projection approach. If None, it is not initialized. Defaults to None.
        :type _affine_decomposition: AffineDecompositionHandlerSpaceTime or NoneType
        """

        super().__init__(_fom_problem, _affine_decomposition)

        self.M_reduction_method = None

        self.M_basis_space = np.zeros(0)  # spatial basis
        self.M_sv_space = np.zeros(0)  # spatial singular values
        self.M_N_space = 0
        self.M_basis_time = np.zeros(0)  # temporal basis
        self.M_sv_time = np.zeros(0)  # temporal singular values
        self.M_N_time = 0

        self.M_x_hat = np.zeros(0)
        self.M_x_IG = np.zeros(0)
        self.M_IG_idxs = slice(None)

        self.M_M_matrix = np.zeros(0)
        self.M_A_matrices = np.zeros(0)

        self.M_basis_space_path = ""
        self.M_basis_time_path = ""
        self.M_gen_coords_path = ""
        self.M_reduced_solution_path = ""
        self.M_fom_solution_path = ""

        return

    def import_basis_space_matrix(self):
        """Method which allows to import from '_input_file' (provided that it represents a valid path) the matrix
        encoding the Reduced Basis in space, which has been constructed performing a POD over the mode-1 unfolding of
        the snapshots' tensor, obtained by solving the unsteady FOM problem at hand for different values of the
        characteristic parameters and using the backward Euler method.

        :return: True if the importing has been successful, False otherwise
        :rtype: bool
        """

        assert self.M_import_offline_structures, "Offline structures import is disabled"

        try:
            self.M_basis_space = np.load(self.M_basis_space_path)
            if len(self.M_basis_space.shape) == 1:
                self.M_basis_space = np.expand_dims(self.M_basis_space, axis=1)
            self.M_Nh = self.M_basis_space.shape[0]
            self.M_N_space = self.M_basis_space.shape[1]
            logger.debug(f"FOM spatial dimension: {self.M_Nh}")
            logger.debug(f"ROM spatial dimension: {self.M_N_space}")
            import_success = True
        except (IOError, OSError, FileNotFoundError) as e:
            logger.error(f"Error {e}: impossible to load the space basis matrix")
            import_success = False

        return import_success

    def import_basis_time_matrix(self):
        """Method which allows to import from '_input_file' (provided that it represents a valid path) the matrix
        encoding the Reduced Basis in time, which has been constructed performing a POD over the mode-2 unfolding of
        the snapshots' tensor, obtained by solving the unsteady FOM problem at hand for different values of the
        characteristic parameters and using the backward Euler method.

        :return: True if the importing has been successful, False otherwise
        :rtype: bool
        """

        assert self.M_import_offline_structures, "Offline structures import is disabled"

        try:
            self.M_basis_time = np.load(self.M_basis_time_path)
            if len(self.M_basis_time.shape) == 1:
                self.M_basis_time = np.expand_dims(self.M_basis_time, axis=1)
            self.M_Nt = self.M_basis_time.shape[0]
            self.M_N_time = self.M_basis_time.shape[1]
            logger.debug(f"FOM temporal dimension: {self.M_Nt}")
            logger.debug(f"ROM temporal dimension: {self.M_N_time}")
            import_success = True
        except (IOError, OSError, FileNotFoundError) as e:
            logger.error(f"Error {e}: impossible to load the time basis matrix")
            import_success = False

        return import_success

    def import_snapshots_quantities(self, _ns=None):
        """Method which allows to import the snapshots' tensor and all the related quantities. In particular, it allows
        to import:

            * the snapshots' tensor
            * the parameters' tensor
            * the Reduced Basis in space
            * the Reduced Basis in time

        Additionally, it is also possible to import a subset of snapshots and parameters (via the input argument '_ns'),
        but the imported basis refer anyway to the full set of snapshots. Also, it sets the dimensionality of the
        Space-Time Reduced Space to the product between the dimensionality of the Reduced  Basis in space and the one
        of the Reduced Basis in time.

        :param _ns: number of snapshots and parameters to be imported. If None or if higher than the total number of
           snapshots available, all the available snapshots and parameters are imported.
        :type _ns: int or NoneType
        :return: True if the importing has been successful, False otherwise
        :rtype: bool
        """

        import_success_snaphsots = self.import_snapshots_matrix(_ns)
        import_success_parameters = self.import_snapshots_parameters(_ns)
        import_success_basis_space = self.import_basis_space_matrix()
        import_success_basis_time = self.import_basis_time_matrix()

        self.M_N = self.M_N_space * self.M_N_time

        return bool(import_success_snaphsots and import_success_parameters
                    and import_success_basis_space and import_success_basis_time)

    def import_snapshots_basis(self):
        """Method which allows to import from files the Reduced Basis in both space and time. Additionally, it sets
        the dimensionality of the Space-Time Reduced Space to the product between the dimensionality of the Reduced
        Basis in space and the one of the Reduced Basis in time.

        :return: True if the importing has been successful, False otherwise
        :rtype: bool
        """

        import_success_basis_space = self.import_basis_space_matrix()
        import_success_basis_time = self.import_basis_time_matrix()

        self.M_N = self.M_N_space * self.M_N_time

        return import_success_basis_space and import_success_basis_time

    def perform_pod_space(self, _tol=1e-4):
        """Method which allows to perform the POD of the mode-1 unfolding of the snapshots' tensor (i.e. on the stored
        snapshots' matrix). Ultimately the matrix encoding the Reduced Basis in space is computed. Finally, if the
        class attribute flag self.M_save_offline_structures is True, the computed Reduced Basis in space is saved as a
        .txt file; the saving path is known to the class if the class method
        :func:`~rb_manager.RbManager.save_offline_structures` has been previously called, via its input argument
        '_basis_matrix_space'.

        :param _tol: tolerance for the relative energy of the already scanned singular values with respect to the energy
          of the full set of singular values. Energy is interpreted as the squared l2-norm of the singular values.
          The actual tolerance used in the algorithm is tolerance = 1.0 - _tol, thus '_tol' is intended to be close
          to 0. Defaults to 1e-5
        :type _tol: float
        """

        logger.info(f"Performing the POD in space, using a tolerance of {_tol:.2e}")

        pod = podec.ProperOrthogonalDecomposition()
        pod(self.M_snapshots_matrix, _tol)
        self.M_basis_space, self.M_sv_space = pod.basis, pod.singular_values
        self.M_N_space = self.M_basis_space.shape[1]

        if self.M_save_offline_structures and self.M_basis_space_path is not None:
            arr_utils.save_array(self.M_basis_space, self.M_basis_space_path)

        return

    def perform_pod_time(self, _tol=1e-4, method='reduced'):
        """Method which allows to perform the POD of the mode-2 unfolding of the space-reduced snapshots' tensor.
        Ultimately the matrix encoding the Reduced Basis or the Residual Reduced Basis in time is computed. Finally,
        if the class attribute flag self.M_save_offline_structures is True, the computed Reduced Basis in time is saved
        as a .txt file; the saving path is known to the class if the class method
        :func:`~rb_manager.RbManager.save_offline_structures` has been previously called, via its input argument
        '_basis_matrix_time'.

        :param _tol: tolerance for the relative energy of the already scanned singular values with respect to the energy
          of the full set of singular values. Energy is interpreted as the squared l2-norm of the singular values.
          The actual tolerance used in the algorithm is tolerance = 1.0 - _tol, thus '_tol' is intended to be close
          to 0. Defaults to 1e-5
        :type _tol: float
        :param method: method to perform the temporal POD. If 'full' it is performed on the mode-2 unfolding of the
           snapshots tensor; if 'reduced' it is performed on the RB projection in space of the mode-2 unfolding of the
           snapshots tensor. It defaults to 'reduced'
       :type method: str
        """

        logger.info(f"Performing the POD in time, using a tolerance of {_tol:.2e} and with the {method} method")

        if method == 'full':
            time_unfold_snapshots_matrix = np.zeros((self.M_Nt, self.M_Nh * self.M_ns))
            for iNs in range(self.M_ns):
                time_unfold_snapshots_matrix[:, iNs * self.M_Nh:(iNs + 1) * self.M_Nh] = \
                    self.M_snapshots_matrix[:, iNs * self.M_Nt:(iNs + 1) * self.M_Nt].T
        elif method == 'reduced':
            time_unfold_snapshots_matrix = np.zeros((self.M_Nt, self.M_N_space * self.M_ns))
            for iNs in range(self.M_ns):
                time_unfold_snapshots_matrix[:, iNs * self.M_N_space:(iNs + 1) * self.M_N_space] = \
                    self.M_snapshots_matrix[:, iNs * self.M_Nt:(iNs + 1) * self.M_Nt].T.dot(self.M_basis_space)
        else:
            raise ValueError(f"Unrecognized method {method} for the temporal POD")

        pod = podec.ProperOrthogonalDecomposition()
        pod(time_unfold_snapshots_matrix, _tol)
        self.M_basis_time, self.M_sv_time = pod.basis, pod.singular_values
        self.M_N_time = self.M_basis_time.shape[1]

        if self.M_save_offline_structures and self.M_basis_time_path is not None:
            arr_utils.save_array(self.M_basis_time, self.M_basis_time_path)

        return

    @property
    def basis_space(self):
        return self.M_basis_space

    @property
    def basis_time(self):
        return self.M_basis_time

    @property
    def N_space(self):
        """Getter method, which allows to get the dimensionality of the spatial Reduced Basis

        :return: dimensionality of the spatial Reduced Basis
        :rtype: int or dict
        """
        return self.M_N_space

    @property
    def N_time(self):
        """Getter method, which allows to get the dimensionality of the temporal Reduced Basis

        :return: dimensionality of the temporal Reduced Basis
        :rtype: int or dict
        """
        return self.M_N_time

    def index_mapping(self, i, j):
        """Method that, given a couple of indices referred to the space basis and the time basis respectively,
        computes and returns the corresponding space-time index

        :param i: space basis index
        :type i: int
        :param j: time basis index
        :type j: int
        :return: space-time basis index
        :rtype: int
        """

        return int(i * self.M_N_time + j)

    def assemble_ST_basis(self):
        """Method that, assuming the Reduced Basis in space and time already computed and stored in the class, assembles
        the SpaceTime basis, by performing the outer products between the space basis elements and the time basis ones.

        :return: computed SpaceTime basis
        :rtype: numpy.ndarray
        """

        logger.info("Assembling the full SpaceTime basis")
        self.M_N = self.M_N_space * self.M_N_time
        M_basis = np.zeros((self.M_Nt * self.M_Nh, self.M_N))

        for i in range(self.M_N_space):
            for j in range(self.M_N_time):
                current_index = self.index_mapping(i, j)
                M_basis[:, current_index] = np.outer(self.M_basis_space[:, i], self.M_basis_time[:, j]).flatten()

        return M_basis

    def build_ST_basis(self, _tol_space=1e-5, _tol_time=1e-5):
        """Method that allows to build the Reduced Basis in both space and time. In particular the spatial
        basis are computed via POD on top of the mode-1 unfolding of the snapshots' tensor,
        while the time basis are computed via POD on top of the mode-2 unfolding of the space-projected snapshots'
        tensor.

        :param _tol_space: tolerance for the relative energy of the already scanned singular values with respect to the
          energy of the full set of singular values, employed for the POD in space. Energy is interpreted as the squared
          l2-norm of the singular values. The actual tolerance used in the algorithm is tolerance = 1.0 - _tol, thus
          '_tol_space' is intended to be close to 0. Defaults to 1e-5
        :type _tol_space: float
        :param _tol_time: tolerance for the relative energy of the already scanned singular values with respect to the
          energy of the full set of singular values, employed for the POD in time. Energy is interpreted as the squared
          l2-norm of the singular values. The actual tolerance used in the algorithm is tolerance = 1.0 - _tol, thus
          '_tol_time' is intended to be close to 0. Defaults to 1e-5
        :type _tol_time: float
        """

        self.perform_pod_space(_tol=_tol_space)
        self.perform_pod_time(_tol=_tol_time)

        self.M_N = self.M_N_space * self.M_N_time

        logger.debug('Finished snapshots PODs \n')

        return

    def compute_generalized_coordinates(self, x_fom):
        """Method that allows to compute the generalized coordinates  (i.e. the coefficients of the linear combination of
        the SpaceTime basis functions) associated to a given FOM solution.

        :param x_fom: FOM solution, given as a 2D array of shape (Ns, Nt)
        :type x_fom: numpy.ndarray
        :return: array of the generalized coordinates associated to the given FOM solution
        :rtype: numpy.ndarray
        """

        x_hat = self.M_affine_decomposition.compute_generalized_coordinates(x_fom, self.M_basis_space, self.M_basis_time)

        return x_hat

    def get_generalized_coordinates(self, n_snapshots=None):
        """Method that allows to compute the generalized coordinates (i.e. the coefficients of the linear combination of
        the SpaceTime basis functions) for the first 'n_snapshots' snapshots stored in the class. If 'n_snapshots' is
        either None or higher than the total number of stored snapshots, the generalized coordinates of all the
        snapshots are computed. The generalized coordinates are saved in the class attribute 'self.M_x_hat'.

        :param n_snapshots: number of snapshots for which the generalized coordinates have to be computed. If it is
          either None or higher than the total number of stored snapshots, the generalized coordinates of all the
          stored snapshots are computed. Defaults to None.
        :type n_snapshots: int or NoneType
        """

        if n_snapshots is None or n_snapshots > self.M_ns:
            logger.info(f"Number of considered snapshots clamped to {self.M_ns}, being it higher "
                        f"than the total number of snapshots considered or set to None")
            n_snapshots = self.M_ns
        else:
            logger.info(f"Number of considered snapshots: {n_snapshots}")

        self.M_x_hat = np.zeros((self.M_N, n_snapshots))

        for iP in range(n_snapshots):
            self.M_x_hat[:, iP] = self.compute_generalized_coordinates(self.M_snapshots_matrix[:,
                                                                       iP * self.M_Nt:(iP + 1) * self.M_Nt])

        if self.M_save_offline_structures:
            np.save(self.M_gen_coords_path, self.M_x_hat)

        return

    def import_generalized_coordinates(self):
        """Method to load the generalized coordinates from file. The path of the file is stored in the class attribute.
        """

        try:
            self.M_x_hat = np.load(self.M_gen_coords_path)
            logger.info(f"Generalized coordinates imported from {self.M_gen_coords_path}")
            self.import_snapshots_parameters()
            return True
        except (IOError, OSError, FileNotFoundError) as e:
            logger.error(f"Error {e}: impossible to load the generalized coordinates")
            return False

    def check_assembling_M_MA_matrices(self):
        """Method which checks whether the matrices 'M_M_matrix' (whose i-th column is M*phi_i) and 'M_A_matrices' (whose
        element with index 'i' in second dimension and 'q' in third dimension is A_q*phi_i) have been assembled (thus,
        being not empty!) or not.

        :return: True if the 'M_M_matrix' and the 'M_A_matrices' are not empty, False otherwise
        :rtype: bool
        """
        return self.M_M_matrix.shape[0] != 0 and self.M_A_matrices.shape[0] != 0

    def assemble_dtMA_matrix_mu(self, num_param):
        """Method that assembles the matrix whose i-th column is (M + dt*A(mu))*phi_i, being 'M' the FOM mass matrix,
        'A' the FOM stiffness matrix, 'mu' the parameter value for the 'num_param' snapshot in the snapshots' tensor,
        'dt' the timestep and 'phi_i' the i-th basis function in space. Such matrix is used to compute the residual
        snapshots.

        :param num_param: index of the snapshot to be considered
        :type num_param: int
        :return: matrix whose i-th column is (M + dt*A(mu))*phi_i, for the given snapshot
        :rtype: numpy.ndarray
        """

        if not self.M_affine_decomposition.check_set_fom_arrays():
            self.M_affine_decomposition.import_fom_affine_arrays(self.M_fom_problem)

        MA_matrix = np.copy(self.M_M_matrix)
        dt = self.dt

        theta_functions_a = self.M_fom_problem.M_full_theta_a(self.M_offline_ns_parameters[num_param, :])

        for index_a in range(theta_functions_a.shape[0]):
            MA_matrix += dt * theta_functions_a[index_a] * self.M_A_matrices[..., index_a]

        return MA_matrix

    def reset_reduced_structures(self):
        """Method which resets to empty arrays the matrices 'M_M_matrix' (i.e. M*phi_i) and 'M_A_matrices'
        (i.e. A_q*phi_i)
        """

        self.M_M_matrix = np.zeros(0)
        self.M_A_matrices = np.zeros(0)

        return

    def reset_rb_approximation(self):
        """Method to reset the RB affine components for both the stiffness matrix and the right-hand side vector by
        redefining the lists containing such elements as empty lists. Additionally, the bases and their dimensionalities
        are also set to an empty matrix and to 0 respectively.
        """

        logger.debug("Resetting RB approximation")

        super().reset_rb_approximation()

        self.M_N_space = 0
        self.M_N_time = 0

        self.M_basis_space = np.zeros(0)
        self.M_basis_time = np.zeros(0)

        self.M_affine_decomposition.reset_rb_approximation()

        return

    def build_nonlinear_rb_affine_components(self):
        """
        MODIFY
        """

        self.M_affine_decomposition.build_nonlinear_rb_affine_components(self.M_basis_space)

        if self.M_save_offline_structures:
            self.M_affine_decomposition.save_nonlinear_rb_affine_decompositions(self.M_nl_term_affine_components_path,
                                                                                self.M_nl_term_jac_affine_components_path)

        return

    def get_RB_nonlinear_affine_components(self, jacobian=True):
        """
        MODIFY
        """

        nl_term_components = self.M_affine_decomposition.M_rbAffineNLTerm
        if jacobian:
            nl_term_jac_components = self.M_affine_decomposition.M_rbAffineNLTermJac

        return (nl_term_components, nl_term_jac_components) if jacobian else nl_term_components

    @staticmethod
    def zigzag_indices(row, col):
        """
        MODIFY
        """

        result = []

        for line in range(1, (row + col)):
            start_col = max(0, line - row)
            count = min(line, (col - start_col), row)

            for j in range(count):
                result.append((min(row, line) - 1 - j, start_col + j))

        return result

    def print_rb_offline_summary(self):
        """Printing method, which prints useful info about the state of the RbManagerSpaceTime class instance. In
        particular, it shows:

            * the number of snapshots
            * the dimensionality of the basis (cumulative, in space and in time)
            * the number of residual snapshots
            * the dimensionality of the residual basis (cumulative, in space and in time)
            * the number of selected SpaceTime locations
            * the number of affine components of the stiffness matrix, the mass matrix and of the right-hand side vector

        """

        logger.info(f"\n ------------  RB SUMMARY  ------------\n"
                    f"Number of snapshots {self.M_ns}\n"                                     
                    f"Number of selected RB functions {self.M_N}\n"                        
                    f"Number of selected RB functions in space {self.M_N_space}\n"                
                    f"Number of selected RB functions in time {self.M_N_time}\n"
                    )

        self.M_affine_decomposition.print_ad_summary()

        return

    def set_paths(self, _snapshot_matrix=None, _offline_parameters=None,
                  _basis_matrix_space=None, _basis_matrix_time=None,
                  _affine_components=None, _generalized_coords=None, _results=None):
        """Method which allows to initialize the paths where to save the most relevant quantities needed
        in the resolution of the parabolic parametrized PDE problem at hand via the ST-LSPG projection approach;
        such quantities are:

            * snapshots
            * parameters
            * Reduced Basis matrices (in both space and time)
            * generalized_coordinates
            * reduced affine components for the left-hand side matrix and the right-hand side vector
            * solution to the problem for some test parameter values, projected onto the reduced space
            * solution to the problem for some test parameter values in the FOM space

        If the paths are passed as None, the corresponding quantities are not saved throughout the execution of the code

        :param _snapshot_matrix: path of the snapshot matrix. Defaults to None
        :type _snapshot_matrix: str or NoneType
        :param _offline_parameters: path of the parameter matrix. Defaults to None
        :type: str or NoneType
        :param _basis_matrix_space: path of the Reduced Basis in space. Defaults to None
        :type _basis_matrix_space: str or NoneType
        :param _basis_matrix_time: path of the Reduced Basis in time. Defaults to None
        :type _basis_matrix_time: str or NoneType
        :param _affine_components: path of the reduced affine components. Defaults to None
        :type _affine_components: str or NoneType
        :param _generalized_coords: path of the generalized coordinates. Defaults to None
        :type _generalized_coords: str or NoneType
        :param _results: path to the result folder. Defaults to None
        :type _results: str or NoneType
        """

        self.M_snapshots_path = _snapshot_matrix
        self.M_parameters_path = _offline_parameters
        self.M_basis_space_path = _basis_matrix_space
        self.M_basis_time_path = _basis_matrix_time
        self.M_affine_components_path = _affine_components
        self.M_gen_coords_path = _generalized_coords

        self.M_results_path = _results

        return

    def _compute_IG_with_zero(self):
        """
        MODIFY
        """

        self.M_u_IG = np.zeros(self.M_N)
        return

    def _compute_IG_with_average(self):
        """
        MODIFY
        """

        if arr_utils.is_empty(self.M_x_hat):
            self.import_generalized_coordinates()

        self.M_x_IG = np.mean(self.M_x_hat, axis=1)

        return

    def _compute_IG_with_NN(self, _param, k=1):
        """
        MODIFY
        """

        if arr_utils.is_empty(self.M_x_hat) or arr_utils.is_empty(self.M_offline_ns_parameters):
            self.import_generalized_coordinates()

        distances = np.array([np.linalg.norm(_param[self.M_IG_idxs] - self.M_offline_ns_parameters[j, self.M_IG_idxs])
                              for j in range(self.M_offline_ns_parameters.shape[0])])
        ref_value = np.mean([np.linalg.norm(self.M_offline_ns_parameters[j, self.M_IG_idxs])
                             for j in range(self.M_offline_ns_parameters.shape[0])])

        if any(distances < 1e-6 * ref_value):
            nearest_neigh_indices = [np.where(distances < 1e-6 * ref_value)[0][0]]
            weights = [1]
        else:
            nearest_neigh_indices = np.argpartition(distances, k)[:k]
            weights = 1 / distances[nearest_neigh_indices]

        self.M_x_IG = np.average(self.M_x_hat[:, nearest_neigh_indices], axis=1, weights=weights)

        # x_fom = np.average([self.M_snapshots_matrix[:, ind * self.M_Nt:(ind + 1) * self.M_Nt]
        #                     for ind in nearest_neigh_indices],
        #                    axis=0, weights=weights)
        #
        # self.M_x_IG = self.compute_generalized_coordinates(x_fom)

        return

    def compute_initial_guess(self, _param, method='NN', k=1):
        """Method that allows to compute the initial guess for the Newton method employed in case the problem at hand
        is non-linear.

        :param _param: value of the parameter
        :type _param: numpy.ndarray
        :param method: method use to determine the initial guess. It defaults to 'NN'.
        :type method: str
        :param k: number of nearest neighbours to be considered. It defaults to 1
        :type k: int
        """

        self._compute_IG_with_zero()

        if arr_utils.is_empty(self.M_offline_ns_parameters) or arr_utils.is_empty(self.M_x_hat):
            logger.warning("The initial guess is returned as an array of zeros, since no reduced snapshots and/or "
                           "parameter values have been uploaded!")
        else:
            if method == 'NN':
                self._compute_IG_with_NN(_param, k=k)
            elif method == 'zero':
                self._compute_IG_with_zero()
            elif method == 'average':
                self._compute_IG_with_average()
            else:
                logger.error(f"Unrecognized interpolation method {method}.")

        return

    @property
    def initial_guess(self):
        return self.M_x_IG

    def solve_reduced_problem(self, _param, _used_Qa=0, _used_Qf=0, _used_Qm=0):
        """Method that allows to build (by calling the
        :func:`~rb_manager_space_time.RbManagerSpaceTime.build_reduced_problem`  method) and solve in the Least Squares
        sense the linear system arising from the application of the ST-LSPG projection approach or the ST-RB approach
        to parabolic unsteady PDE problems.

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
        :param _used_Qm: number of affine components that are used to assemble the mass matrix. If equal to 0, all
          the components are used; values different from 0 make sense only if the mass matrix is not affine
          decomposable, so that it is necessary to resort to an approximate affine decomposition via the MDEIM
          algorithm. Defaults to 0
        :type _used_Qm: int
        :return: solution to the parabolic PDE problem for the parameter value given as input, approximated via the
          ST-LSPG projection approach or the ST-RB approach
        :rtype: numpy.ndarray
        """

        logger.debug(f"Solving RB problem for parameter: {_param}")

        self.M_used_Qa = self.M_affine_decomposition.qa
        if _used_Qa > 0:
            self.M_used_Qa = _used_Qa

        self.M_used_Qf = self.M_affine_decomposition.qf
        if _used_Qf > 0:
            self.M_used_Qf = _used_Qf

        self.M_used_Qm = self.M_affine_decomposition.qm
        if _used_Qm > 0:
            self.M_used_Qm = _used_Qm

        self.build_reduced_problem(_param)

        if self.M_fom_problem.is_linear():
            self.M_un = scipy.linalg.solve(self.M_An, self.M_fn)
            self.M_solver_converged = True
        else:
            my_newton = Newton(tol=1e-5, tol_abs=1e-8, max_err=1e-4, max_err_abs=1e-7, max_iter=10, jac_iter=1)

            def residual(u):
                return (self.M_An.dot(u) +
                        self.M_affine_decomposition.build_rb_nonlinear_term(u, self.M_basis_space,
                                                                            self.M_basis_time,
                                                                            self.M_fom_problem) * self.dt -
                        self.M_fn)

            def jacobian(u):
                jac = self.M_An
                if self.M_nonlinear_jacobian:
                    jac += self.M_affine_decomposition.build_rb_nonlinear_jacobian(u, self.M_basis_space,
                                                                                   self.M_basis_time,
                                                                                   self.M_fom_problem,
                                                                                   recompute_every=1) * self.dt
                return jac

            # define initial guess
            self.compute_initial_guess(_param, method='average', k=3)

            # call Newton method and compute solution
            self.M_un, self.M_solver_converged = my_newton(residual, jacobian, self.M_x_IG)

        return not self.M_solver_converged

    def reconstruct_fem_solution(self, _un, indices_space=None, indices_time=None):
        """Method which reconstructs the FOM solution (in both space and time) from the dimensionality reduced one,
        by linearly combining the elements of the SpaceTime Reduced Basis with weights given by the entries of the
        reduced solution itself. It is the inverse method of
        :func:`~rb_manager_space_time.RbManagerSpaceTime.get_generalized_coordinates`

        :param _un: dimensionality reduced vector, typically arising from the Least Squares solution of the reduced
          linear system
        :type _un: numpy.ndarray
        :param indices_space: indices of the spatial entries where to reconstruct the solution. If None, all entries
           are selected. It defaults to None.
        :type indices_space: list[int] or np.ndarray(int)
        :param indices_time: indices of the temporal entries where to reconstruct the solution. If None, all entries
           are selected. It defaults to None.
        :type indices_time: list[int] or np.ndarray(int)
        """

        self.M_utildeh = self.M_affine_decomposition.reconstruct_fem_solution(_un, self.M_basis_space, self.M_basis_time,
                                                                              _indices_space=indices_space,
                                                                              _indices_time=indices_time)

        return


__all__ = [
    "RbManagerSpaceTime"
]
