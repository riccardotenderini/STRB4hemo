#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:00:21 2022
@author: Riccardo Tenderini
@mail: riccardo.tenderini@epfl.ch
"""

import os
import json
import time
import numpy as np
from scipy.interpolate import RBFInterpolator

import src.rb_library.rb_manager.space_time.Stokes.rb_manager_space_time_Stokes as rbmstS
from src.utils.newton import Newton
import src.utils.array_utils as arr_utils

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../../../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class RbManagerSpaceTimeNavierStokes(rbmstS.RbManagerSpaceTimeStokes):
    """MODIFY
    """

    def __init__(self, _fom_problem, _affine_decomposition=None):
        """ MODIFY
        """

        super().__init__(_fom_problem, _affine_decomposition=_affine_decomposition)

        self.M_NLTerm_affine_components = []
        self.M_NLJacobian_affine_components = []
        self.M_N_components_NLTerm = 0
        self.M_N_components_NLJacobian = 0
        self.M_NLterm_offline_time_tensor_uuu = np.zeros(0)
        self.M_NLterm_offline_time_tensor_uuu_IC_1 = np.zeros(0)
        self.M_NLterm_offline_time_tensor_uuu_IC_2 = np.zeros(0)

        self.M_u_IG = np.zeros(0)
        self.M_newton_specifics = dict()
        self.M_u_IG_rbf = dict()

        return

    def import_NLTerm_affine_components(self, _tolerances,  N_components=None,
                                        _space_projection='standard'):
        """
        MODIFY
        """

        if not self.check_norm_matrices():
            self.get_norm_matrices()

        if len(self.M_NLTerm_affine_components) > 0:
            return

        logger.info("Importing non-linear term affine components")

        path = os.path.join(self.M_fom_structures_path,
                            os.path.join(f"NLterm",
                                         f"POD_tol_{_tolerances['velocity-space']:.0e}_"
                                         f"{_tolerances['pressure-space']:.0e}",
                                         _space_projection, "Vector"))
        Nmax = N_components if N_components is not None else self.M_N_space['velocity']
        if not os.path.isdir(path) and Nmax > 0:
            raise ValueError(f"Invalid path! No affine components for the NL term available at {path} !")
        if Nmax > self.M_N_space['velocity']:
            logger.warning(f"Setting the number of affine components for the NL term to {self.M_N_space['velocity']}, "
                           f"since the prescribed number {Nmax} exceeds the basis dimension {self.M_N_space['velocity']}")
            Nmax = self.M_N_space['velocity']

        count_i = 0
        while os.path.isfile(os.path.join(path, f"Vec_{count_i}_0.m")) and count_i < Nmax:
            self.M_NLTerm_affine_components.append([])
            count_j = 0
            while os.path.isfile(os.path.join(path, f"Vec_{count_i}_{count_j}.m")) and count_j < Nmax:
                self.M_NLTerm_affine_components[count_i].append(np.loadtxt(
                    os.path.join(path, f"Vec_{count_i}_{count_j}.m"), delimiter=',')[:self.M_N_space['velocity']])
                count_j += 1
            count_i += 1

        self.M_N_components_NLTerm = count_i

        logger.info(f"Loaded {self.M_N_components_NLTerm**2} non-linear term affine components")

        return

    @property
    def NLTerm_affine_components(self):
        return self.M_NLTerm_affine_components

    def import_NLJacobian_affine_components(self, _tolerances,  N_components=None,
                                            _space_projection='standard'):
        """
        MODIFY
        """

        assert 'use convective jacobian' in self.M_newton_specifics.keys()

        if len(self.M_NLJacobian_affine_components) > 0:
            return

        logger.info("Importing non-linear jacobian affine components")

        path = os.path.join(self.M_fom_structures_path,
                            os.path.join(f"NLterm",
                                         f"POD_tol_{_tolerances['velocity-space']:.0e}_"
                                         f"{_tolerances['pressure-space']:.0e}",
                                         _space_projection, "Matrix"))
        Nmax = (N_components if N_components is not None else self.M_N_space['velocity'])
        Nmax *= self.M_newton_specifics['use convective jacobian']
        if not os.path.isdir(path) and Nmax > 0:
            raise ValueError(f"Invalid path! No affine components for the NL term jacobian available at {path} !")
        if Nmax > self.M_N_space['velocity']:
            logger.warning(f"Setting the number of affine components for the NL term to {self.M_N_space['velocity']}, "
                           f"since the prescribed number {Nmax} exceeds the basis dimension {self.M_N_space['velocity']}")
            Nmax = self.M_N_space['velocity']

        count = 0
        while os.path.isfile(os.path.join(path, f"Mat_{count}.m")) and count < Nmax:
            self.M_NLJacobian_affine_components.append(np.loadtxt(
                os.path.join(path, f"Mat_{count}.m"), delimiter=',')[:self.M_N_space['velocity'], :self.M_N_space['velocity']])
            count += 1

        self.M_N_components_NLJacobian = count

        logger.info(f"Loaded {self.M_N_components_NLJacobian} non-linear jacobian affine components")

        return

    @property
    def NLJacobian_affine_components(self):
        return self.M_NLJacobian_affine_components

    def compute_NLterm_offline_quantities_time(self):
        """
        MODIFY
        """

        self.M_NLterm_offline_time_tensor_uuu = np.einsum('ai,aj,ak->ijk',
                                                          self.M_basis_time['velocity'],
                                                          self.M_basis_time['velocity'],
                                                          self.M_basis_time['velocity'])

        self.M_NLterm_offline_time_tensor_uuu_IC_1 = np.einsum('ai,aj,ak->ijk',
                                                               self.M_basis_time_IC.T,
                                                               self.M_basis_time['velocity'],
                                                               self.M_basis_time['velocity'])

        self.M_NLterm_offline_time_tensor_uuu_IC_2 = np.einsum('ai,aj,ak->ijk',
                                                               self.M_basis_time_IC.T,
                                                               self.M_basis_time_IC.T,
                                                               self.M_basis_time['velocity'])

        return

    @property
    def NLterm_offline_time_tensor_uuu(self):
        return self.M_NLterm_offline_time_tensor_uuu

    def compute_NLterm_offline_quantities_space(self):
        """
        MODIFY
        """

        return

    def compute_NLterm_offline_quantities(self):
        """
        MODIFY
        """

        self.compute_NLterm_offline_quantities_space()
        self.compute_NLterm_offline_quantities_time()

        return

    def import_NLterm_offline_quantities_space(self):
        """
        MODIFY
        """

        return

    def reset_reduced_structures(self):
        """
        MODIFY
        """

        super().reset_reduced_structures()

        self.M_NLTerm_affine_components = []
        self.M_NLJacobian_affine_components = []
        self.M_N_components_NLTerm = 0
        self.M_N_components_NLJacobian = 0

        return

    def import_reduced_convective_structures(self, _tolerances, N_components=None, _space_projection='standard'):
        """
        MODIFY
        """

        self.import_NLTerm_affine_components(_tolerances, N_components=N_components,
                                             _space_projection=_space_projection)
        self.import_NLJacobian_affine_components(_tolerances, N_components=N_components,
                                                 _space_projection=_space_projection)

        if N_components is not None and N_components > 0:
            self.compute_NLterm_offline_quantities()

        return

    def import_reduced_structures(self, _tolerances=None, N_components=None,
                                  _space_projection='standard'):
        """
        MODIFY
        """

        assert _tolerances is not None

        import_success = super().import_reduced_structures()

        self.import_reduced_convective_structures(_tolerances,
                                                  N_components=N_components, _space_projection=_space_projection)

        return import_success

    def assemble_reduced_structures(self, _space_projection='standard',
                                    _tolerances=None, N_components=None):
        """
        MODIFY
        """

        assert _tolerances is not None

        super().assemble_reduced_structures(_space_projection=_space_projection)

        self.import_reduced_convective_structures(_tolerances,
                                                  N_components=N_components, _space_projection=_space_projection)

        return

    def compute_initial_guess(self, _param, method='NN', k=1):
        """Method that allows to compute the initial guess for the Newton method employed in case the problem at hand
        is non-linear.

        :param _param: value of the parameter
        :type _param: numpy.ndarray
        :param method: method use to determine the initial guess. 'NN' uses weighted nearest neighbours interpolation.
           'PODI' uses RBF interpolation on the space-time reduced coefficients. It defaults to 'NN'.
        :type method: str
        :param k: number of nearest neighbours to be considered. It defaults to 1
        :type k: int
        """

        self.M_u_IG = self.get_zero_vector()

        if (arr_utils.is_empty(self.M_offline_ns_parameters) or
                'velocity' not in self.M_snapshots_hat or
                arr_utils.is_empty(self.M_snapshots_hat['velocity'])):
            logger.warning("The initial guess is returned as an array of zeros, since no reduced snapshots and/or "
                           "parameter values have been uploaded!")
            self._compute_IG_with_zero()

        else:
            if method == 'NN':
                self._compute_IG_with_NN(_param, k=k)
            elif method == 'PODI':
                self._compute_IG_with_PODI(_param)
            elif method == 'zero':
                self._compute_IG_with_zero()
            elif method == 'average':
                self._compute_IG_with_average()
            else:
                logger.error(f"Unrecognized interpolation method {method}.")

        return

    def _compute_IG_with_NN(self, _param, k=1, get_full_output=False):
        """ Compute the initial guess for the Newton method via nearest neighbours interpolation. The initial guess is
        computed just by finding the k parameter values (among the ones used in the offline phase) which are closer to
        the given one in l2-norm and by averaging the corresponding solutions. If the offline parameters are not
        available (because they have not been uploaded) a warning message is displayed and an array of zeros of the
        proper length is returned

        :param _param: value of the parameter
        :type _param: numpy.ndarray
        :param k: number of nearest neighbours to be considered. It defaults to 1
        :type k: int
        :param get_full_output: if true, also the indices and weights of the considered neighbours are returned.
           It defaults to False.
        :param get_full_output: bool
        :return: initial guess, in terms of its generalized coordinates, associated to the given parameter value
        :rtype: numpy.ndarray
        """

        # TODO: exploit some prior on the sensitivity of the solution with respect to the parameters ? For instance,
        #  for the clots test cases, we may restrict the search to the snapshots featuring clots in the same positions.

        if not self.M_snapshots_hat:
            self.import_generalized_coordinates()

        if k > self.M_offline_ns_parameters.shape[0]:
            k = self.M_offline_ns_parameters.shape[0]

        _param = self._normalize_parameter(_param[None], idxs=self.M_IG_idxs)[0]
        parameters = self._normalize_parameter(self.M_offline_ns_parameters, idxs=self.M_IG_idxs)

        distances = np.array([np.linalg.norm(_param - parameters[j]) for j in range(parameters.shape[0])])
        ref_value = np.mean([np.linalg.norm(parameters[j]) for j in range(parameters.shape[0])])

        if any(distances <= 1e-6 * ref_value):
            nn_indices = [np.where(distances <= 1e-6 * ref_value)[0][0]]
            weights = np.array([1.0])
        else:
            nn_indices = np.argpartition(distances, k - 1)[:k]
            weights = 1 / distances[nn_indices]
        weights /= np.sum(weights)

        logger.debug(f"Number of neighbours considered: {len(weights)}. "
                     f"Neighbours distances: {distances[nn_indices]}")

        u_hat = np.average(self.M_snapshots_hat['velocity'][:, nn_indices], axis=1, weights=weights)
        p_hat = np.average(self.M_snapshots_hat['pressure'][:, nn_indices], axis=1, weights=weights)
        l_hat = [np.zeros(0)] * self.M_n_coupling
        for n in range(self.M_n_coupling):
            l_hat[n] = np.average(self.M_snapshots_hat['lambda'][n][:, nn_indices], axis=1, weights=weights)
        l_hat = np.hstack([elem for elem in l_hat])

        self.M_u_IG = np.concatenate([u_hat, p_hat, l_hat], axis=0)

        return None if not get_full_output else (nn_indices, weights)

    def _compute_IG_with_PODI(self, _param):
        """Compute the initial guess for the Newton method via proper orthogonal decomposition interpolation.

        :param _param: value of the parameter
        :type _param: numpy.ndarray
        :return: initial guess, in terms of its generalized coordinates, associated to the given parameter value
        :rtype: numpy.ndarray
        """

        if not self.M_snapshots_hat:
            self.import_generalized_coordinates()

        if not self.M_u_IG_rbf:
            logger.info("Initializing RBF interpolation")

            params = self._normalize_parameter(self.M_offline_ns_parameters, idxs=self.M_IG_idxs)
            rbf_specs = {'kernel': 'thin_plate_spline', 'smoothing': 1e-2, 'degree': 1, 'epsilon': 1.0}

            self.M_u_IG_rbf['velocity'] = RBFInterpolator(params, self.M_snapshots_hat['velocity'].T, **rbf_specs)
            self.M_u_IG_rbf['pressure'] = RBFInterpolator(params, self.M_snapshots_hat['pressure'].T, **rbf_specs)
            self.M_u_IG_rbf['lambda'] = [None] * self.M_n_coupling
            for n in range(self.M_n_coupling):
                self.M_u_IG_rbf['lambda'][n] = RBFInterpolator(params, self.M_snapshots_hat['lambda'][n].T, **rbf_specs)

        _param = self._normalize_parameter(_param[None], idxs=self.M_IG_idxs)
        u_hat = self.M_u_IG_rbf['velocity'](_param)[0]
        p_hat = self.M_u_IG_rbf['pressure'](_param)[0]
        l_hat = np.hstack([self.M_u_IG_rbf['lambda'][n](_param) for n in range(self.M_n_coupling)])[0]

        self.M_u_IG = np.hstack([u_hat, p_hat, l_hat])

        return

    def _compute_IG_with_average(self):
        """
        MODIFY
        """

        if not self.M_snapshots_hat:
            self.import_generalized_coordinates()

        u_hat = np.mean(self.M_snapshots_hat['velocity'], axis=1)
        p_hat = np.mean(self.M_snapshots_hat['pressure'], axis=1)
        l_hat = np.hstack([np.mean(self.M_snapshots_hat['lambda'][n], axis=1) for n in range(self.M_n_coupling)])

        self.M_u_IG = np.hstack([u_hat, p_hat, l_hat])

        return

    def _compute_IG_with_zero(self):
        """
        MODIFY
        """

        self.M_u_IG = self.get_zero_vector()
        return

    def _setup_IG_indices(self):
        """
        Specify which parameters should be considered to compute the initial guess.
        """

        # TODO: this is very simple... consider only inflow parameters, unless there are clots
        # if 'inflow' in self.M_parametrizations and 'clot' not in self.M_parametrizations:
        #     idxs = self.get_param_indices()['inflow']  # consider only inflow parameters for IG, if possible
        #     self.M_IG_idxs = np.arange(idxs[0], idxs[1]).astype(int)

        return

    @property
    def initial_guess(self):
        return self.M_u_IG

    def build_reduced_convective_term(self, x_hat):
        """
        MODIFY
        """

        raise NotImplementedError("This method is not implemented by this class")

    def build_reduced_convective_jacobian(self, x_hat):
        """
        MODIFY
        """
        raise NotImplementedError("This method is not implemented by this class")

    def assemble_reduced_structures_nlterm(self, x_hat, param=None):
        """
        Assemble the reduced structures needed to assemble the reduced non-linear term and its jacobian
        """

        raise NotImplementedError("This method is not implemented by this class")

    def reset_reduced_structures_nlterm(self):
        """
        Reset the reduced structures needed to assemble the reduced non-linear term and its jacobian
        """

        raise NotImplementedError("This method is not implemented by this class")

    def build_rb_approximation(self, _ns, _n_weak_io, _mesh_name, _tolerances,
                               _space_projection='standard', prob=None, ss_ratio=1, _N_components=None):
        """
        MODIFY
        """

        self.define_geometry_info(_n_weak_io[0], _n_weak_io[1], _mesh_name)
        self.reset_rb_approximation()

        logger.info(f"Building {self.M_reduction_method} approximation with {_ns} snapshots and tolerances "
                    f"{json.dumps(_tolerances, indent=1)}")

        if self.M_import_snapshots:
            logger.info('Importing stored snapshots')
            import_success_snapshots = self.import_snapshots_matrix(_ns, ss_ratio=ss_ratio)
        else:
            import_success_snapshots = False

        if self.M_import_offline_structures:
            logger.info('Importing space and time basis matrices\n')
            import_failures_basis = self.import_snapshots_basis()
        else:
            import_failures_basis = None

        if import_success_snapshots and self.M_ns < _ns:
            logger.warning(f"We miss some snapshots! I have only {self.M_ns} in memory and "
                           f"I would need to compute {_ns - self.M_ns} more!")

        if not import_success_snapshots and import_failures_basis:
            raise ValueError("Impossible to assemble the reduced problem if neither the snapshots nor the bases "
                             "can be loaded!")

        import_reduced_ac = self.M_import_offline_structures
        if import_success_snapshots and (import_failures_basis is None or len(import_failures_basis)):
            logger.info("Basis importing failed. We need to construct it via POD")
            self.build_ST_basis(_tolerances, which=import_failures_basis)
            import_reduced_ac = False

        logger.info('Building RB affine decomposition')
        if import_reduced_ac:
            logger.info('Importing reduced structures')
            import_success_reduced_structures = self.import_reduced_structures(_tolerances=_tolerances,
                                                                               N_components=_N_components,
                                                                               _space_projection=_space_projection)
            if not import_success_reduced_structures:
                logger.info('Building space reduced structures whose import has failed')
                start = time.time()
                self.assemble_reduced_structures(_space_projection=_space_projection,
                                                 _tolerances=_tolerances, N_components=_N_components)
                logger.debug(f"Assembling of space-reduced structures performed in {(time.time()-start):.4f} s")

            if 'ST' in self.M_reduction_method:  # avoid construction of ST blocks in SRB-TFO
                import_failures_ac = self.import_rb_affine_components()
                import_success_generalized_coords = self.import_generalized_coordinates()
                if not import_success_snapshots:
                    self._import_parameters(_ns=_ns)
                if import_failures_ac:
                    start = time.time()
                    logger.info('Building affine space-time reduced structures whose import has failed')
                    self.build_rb_affine_decompositions()
                    logger.debug(f"Assembling of space-time-reduced structures performed in {(time.time()-start):.4f} s")
                if not import_success_generalized_coords:
                    if not import_success_snapshots:
                        raise ValueError("Impossible to compute the generalized coordinates without FOM snapshots.")
                    start = time.time()
                    logger.info('Computing space-time generalized coordinates of FOM snapshots')
                    self.get_generalized_coordinates(n_snapshots=_ns, ss_ratio=ss_ratio, save_flag=True)
                    logger.debug(f"Computing space-time generalized coordinates performed in {(time.time()-start):.4f} s")

        else:
            start = time.time()
            logger.info('Building space-reduced structures')
            self.assemble_reduced_structures(_space_projection=_space_projection,
                                             _tolerances=_tolerances, N_components=_N_components)
            logger.debug(f"Assembling of space-reduced structures performed in {(time.time()-start):.4f} s")

            if 'ST' in self.M_reduction_method:
                start = time.time()
                self.build_rb_affine_decompositions()
                logger.debug(f"Assembling of space-time-reduced structures performed in {(time.time()-start):.4f} s")

                if not import_success_snapshots:
                    raise ValueError("Impossible to compute the generalized coordinates without FOM snapshots.")
                start = time.time()
                logger.info('Computing space-time generalized coordinates of FOM snapshots')
                self.get_generalized_coordinates(n_snapshots=_ns, ss_ratio=ss_ratio, save_flag=True)
                logger.debug(f"Computing space-time generalized coordinates performed in {(time.time()-start):.4f} s")

        return

    def set_newton_specifics(self, newton_specs):
        """
        MODIFY
        """

        self.M_newton_specifics = newton_specs

        return

    def _reset_solution(self):
        """
        MODIFY
        """
        super()._reset_solution()
        self.M_u_IG = np.zeros(0)
        return

    def _reset_errors(self):
        """
        MODIFY
        """

        super()._reset_errors()

        self.M_relative_error['IG-velocity'], self.M_relative_error['IG-velocity-l2'] = np.zeros(self.M_Nt), 0.0
        self.M_relative_error['IG-pressure'], self.M_relative_error['IG-pressure-l2'] = np.zeros(self.M_Nt), 0.0

        return

    def _update_errors(self, N=None, is_IG=False):
        """
        MODIFY
        """

        if N is None:
            N = 1

        if not is_IG:
            super()._update_errors(N=N)
        else:
            self.M_relative_error['IG-velocity'] += self.M_cur_errors['velocity'] / N
            self.M_relative_error['IG-velocity-l2'] += self.M_cur_errors['velocity-l2'] / N
            self.M_relative_error['IG-pressure'] += self.M_cur_errors['pressure'] / N
            self.M_relative_error['IG-pressure-l2'] += self.M_cur_errors['pressure-l2'] / N

        return

    def _solve_lu(self, **kwargs):
        """
        MODIFY
        """
        raise NotImplementedError("`_solve_lu` method is directly embedded in `_solve` method "
                                  "for the Navier-Stokes solver.")

    def _solve(self, _param=None):
        """
        MODIFY
        """

        assert _param is not None, "Parameter value not provided in input"

        # self._setup_IG_indices()

        tol = (self.M_newton_specifics['tolerance'] ** 2 if self.M_reduction_method == "ST-PGRB"
               else self.M_newton_specifics['tolerance'])
        tol_abs = self.M_newton_specifics['absolute tolerance']
        max_err = (self.M_newton_specifics['max error'] ** 2 if self.M_reduction_method == "ST-PGRB"
                   else self.M_newton_specifics['max error'])
        max_err_abs = self.M_newton_specifics['absolute max error']
        max_iter = self.M_newton_specifics['max iterations']

        my_newton = Newton(tol=tol, tol_abs=tol_abs, max_err=max_err, max_err_abs=max_err_abs,
                           max_iter=max_iter, jac_iter=1, alpha=1.0)

        if self.M_newton_specifics['use convective jacobian'] and self.M_use_LU:
            logger.warning("LU factorization cannot be exploited if affine components for the "
                           "convective jacobian are considered!")
            self.M_use_LU = False

        def residual(u):
            nl_term_rb = self.build_reduced_convective_term(u)
            res = self.M_Block.dot(u) + nl_term_rb - self.M_f_Block
            return res

        def jacobian(u):
            if self.M_use_LU:
                return self.M_Block_LU
            else:
                jac = np.copy(self.M_Block)
                if self.M_newton_specifics['use convective jacobian']:
                    nl_jac_rb = self.build_reduced_convective_jacobian(u)
                    jac += nl_jac_rb
                return jac

        def pre_iter(u=None):
            self.assemble_reduced_structures_nlterm(u, param=_param)
            return

        def post_iter(u=None):
            self.reset_reduced_structures_nlterm()
            return

        # define initial guess
        # if self.M_un.size > 0:
        #     self.M_u_IG = self.M_un  # use solution at previous cycle as IG for the next cycle
        # else:
        self.compute_initial_guess(_param,
                                   method=self.M_newton_specifics['IG mode'],
                                   k=self.M_newton_specifics['neighbors number'])

        # call Newton method and compute solution
        self.M_un, self.M_solver_converged = my_newton(residual, jacobian, self.M_u_IG.copy(),
                                                       pre_iter=pre_iter,
                                                       post_iter=post_iter,
                                                       use_lu=self.M_use_LU)

        status = 0 if self.M_solver_converged else 1

        return status

    def _fill_errors(self, errors, data=None, is_IG=False):
        """
        MODIFY
        """

        if data is None:
            data = dict()

        data[("IG-" if is_IG else "") + "velocity"] = errors['velocity-l2']
        data[("IG-" if is_IG else "") + "pressure"] = errors['pressure-l2']
        for n in range(self.M_n_coupling):
            data[("IG-" if is_IG else "") + f"Lagrange multipliers {n}"] = errors['lambda-l2'][n]

        return data

    def solve_pipeline(self, param_nb, is_test=False, ss_ratio=1, compute_IG_error=False, **kwargs):
        """
        MODIFY
        """

        status, elapsed_time = super().solve_pipeline(param_nb, is_test=is_test, ss_ratio=ss_ratio)

        if compute_IG_error:
            logger.debug("Computing the errors on the initial guess")
            self.reconstruct_fem_solution(self.M_u_IG)
            errors = self.compute_online_errors(param_nb, is_test=is_test, ss_ratio=ss_ratio)
            self._update_errors(is_IG=True)

            errors_folder_name = os.path.join(self.M_results_path, f'param{param_nb}', 'errors')
            data = json.load(open(os.path.join(errors_folder_name, 'errors.json'), 'r'))

            data = self._fill_errors(errors, data=data, is_IG=True)

            json.dump(data, open(os.path.join(errors_folder_name, 'errors.json'), "w"))

        return status, elapsed_time

    def check_dataset(self, _nsnap):
        """
        MODIFY
        """

        raise NotImplementedError("This method is not implemented for the Navier-Stokes problem")


__all__ = [
    "RbManagerSpaceTimeNavierStokes"
]





