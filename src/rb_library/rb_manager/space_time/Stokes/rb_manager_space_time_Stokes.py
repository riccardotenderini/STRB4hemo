#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 09:48:05 2021
@author: Riccardo Tenderini
@mail: riccardo.tenderini@epfl.ch
"""
import numpy as np
import os
import shutil
import warnings

import scipy.sparse
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix, lil_matrix, bmat, load_npz, save_npz
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import time
import json

import src.rb_library.proper_orthogonal_decomposition as podec
from src.utils.newton import Newton
import src.rb_library.rb_manager.space_time.rb_manager_space_time as rbmst
import src.utils.array_utils as arr_utils
import src.utils.general_utils as gen_utils

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../../../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class RbManagerSpaceTimeStokes(rbmst.RbManagerSpaceTime):
    """MODIFY
    """

    def __init__(self, _fom_problem, _affine_decomposition=None):
        """ MODIFY
        """

        super().__init__(_fom_problem, _affine_decomposition=_affine_decomposition)

        self.M_valid_meshes = {'tube_1x2_h0.12', 'tube_1x2_h0.15',
                               'tube_1x2_h0.20', 'tube_1x2_h0.30',
                               'tube_1x3_h0.20', 'tube_1x4_h0.15',
                               'bif_sym_alpha50_h0.10',
                               'bypass_coarse_fluid', 'bypass_BL',
                               'aorta_normal_rings'}

        self.M_valid_fields = ['velocity', 'pressure', 'lambda']

        self.M_snapshots_matrix = {'velocity': np.zeros(0), 'pressure': np.zeros(0), 'lambda': np.zeros(0)}
        self.M_snapshots_hat = {'velocity': np.zeros(0), 'pressure': np.zeros(0), 'lambda': np.zeros(0)}
        self.M_test_snapshots_matrix = {'velocity': np.zeros(0), 'pressure': np.zeros(0), 'lambda': np.zeros(0)}

        self.M_snapshots_IC = {'velocity': np.zeros(0), 'pressure': np.zeros(0), 'lambda': np.zeros(0)}
        self.M_test_snapshots_IC = {'velocity': np.zeros(0), 'pressure': np.zeros(0), 'lambda': np.zeros(0)}

        self.M_basis_space, self.M_sv_space = dict(), dict()
        self.M_basis_time, self.M_sv_time = dict(), dict()

        self.M_A_matrix = np.zeros(0)
        self.M_Bdiv_matrix = np.zeros(0)
        self.M_BdivT_matrix = np.zeros(0)
        self.M_B_matrix = np.zeros(0)
        self.M_BT_matrix = np.zeros(0)
        self.M_RHS_vector = np.zeros(0)
        self.M_Mclot_matrices = []
        self.M_q_vectors = {'in': [], 'out': []}
        self.M_R_matrix = np.zeros(0)
        self.M_Radd_matrix = np.zeros(0)

        self.M_has_resistance = False

        self.M_u_hat = dict()
        self.M_utildeh = dict()
        self.M_u0 = dict()

        self.M_shift_param = np.zeros(0)
        self.M_scale_param = np.ones(0)

        self.M_Nh = {'velocity': 0, 'pressure': 0, 'lambda': np.zeros(0, dtype=np.int16)}
        self.M_N = {'velocity': 0, 'pressure': 0, 'lambda': np.zeros(0, dtype=np.int16)}
        self.M_N_space = {'velocity': 0, 'pressure': 0, 'lambda': np.zeros(0, dtype=np.int16)}
        self.M_N_time = {'velocity': 0, 'pressure': 0, 'lambda': np.zeros(0, dtype=np.int16)}
        self.M_N_params = 0
        self.M_N_lambda_cumulative = np.zeros(0, dtype=np.int16)
        self.M_N_periods = 1
        self.M_Nt_IC = 0

        self.M_n_weak_inlets = 0
        self.M_n_weak_outlets = 0
        self.M_n_coupling = 0
        self.M_n_inlets = 0
        self.M_n_outlets = 0

        self.M_inflow_rate_function = None
        self.M_outflow_rate_function = None

        self.M_norm_matrices_bcs = dict()
        self.M_norm_matrices = dict()

        self.supr_primal = np.zeros(0)
        self.supr_dual = np.zeros(0)

        self.M_Blocks = [np.zeros(0)] * 9
        self.M_Blocks_param = dict()
        self.M_Blocks_param_affine = dict()
        self.M_Blocks_param_affine_fun = dict()
        self.M_Block = np.zeros(0)
        self.M_Block_LU = tuple()
        self.M_f_Block = np.zeros(0)
        self.M_f_Blocks_no_param = [np.zeros(0)] * 3
        self.M_f_Blocks_param = dict()
        self.M_f_Blocks_param_affine = dict()
        self.M_f_Blocks_param_affine_fun = dict()

        self.M_relative_error = dict()
        self.M_cur_errors = dict()
        self.M_online_mean_time = 0.0

        self.M_generalized_coords_path = ""
        self.M_fom_structures_path = ""
        self.M_reduced_structures_path = ""
        self.M_mesh_name = ""

        self.M_parametrizations = []
        self.M_valid_parametrizations = ['inflow', 'outflow', 'fluid', 'clot']
        self.M_n_inflow_params = 0
        self.M_n_outflow_params = 0

        self.M_use_LU = False

        # TODO: generalize for BDFs with different orders !
        self.M_bdf = np.array([-4/3, 1/3])
        self.M_bdf_rhs = 2/3
        self.M_extrap_coeffs = np.array([2.0, -1.0])

        return

    @property
    def use_LU(self):
        return self.M_use_LU

    @use_LU.setter
    def use_LU(self, _use_LU):
        self.M_use_LU = _use_LU
        return

    @property
    def N_lambda_cumulative(self):
        return self.M_N_lambda_cumulative

    @property
    def N_periods(self):
        return self.M_N_periods

    @N_periods.setter
    def N_periods(self, _N_periods):
        self.M_N_periods = _N_periods
        return

    @property
    def Nt_IC(self):
        return self.M_Nt_IC

    @Nt_IC.setter
    def Nt_IC(self, _Nt_IC):
        self.M_Nt_IC = _Nt_IC
        return

    def _load_snapshot_file(self, path, field, skip_IC=True, save=True, remove=True):
        """
        MODIFY
        """

        logger.debug(f"Loading snapshot file for field {field} at {path}")

        if 'lambda' in field:
            field_cnt = int(field[6:])
            fname = f"lagmult{field_cnt}"
        else:
            field_cnt = self.M_valid_fields.index(field)
            fname = f"field{field_cnt}"

        try:
            cur_data = np.load(os.path.join(path, fname + '.npy'))

        except (IOError, FileNotFoundError):

            if os.path.isfile(os.path.join(path, 'block0.h5')) and 'lambda' not in field:
                # parallel simulations: read .h5 file
                filename = os.path.join(path, 'block0.h5')
                cur_data = gen_utils.read_field_from_h5(filename, field)
            else:
                # serial simulations: read .txt files
                filename = os.path.join(path, fname + '.snap')
                cur_data = np.genfromtxt(filename, delimiter=',')

            # saving in .npy format
            if save:
                np.save(os.path.join(path, fname + '.npy'), cur_data)
            # deleting .txt / .h5 files
            if remove:
                os.remove(filename)

        if skip_IC and self.M_Nt_IC > 0:
            return cur_data[self.M_Nt_IC:], cur_data[self.M_Nt_IC-2:self.M_Nt_IC]
        else:
            return cur_data, None

    def _load_FEM_vector_file(self, fname):
        """
        Read a FEM vector. This method accounts for the fact that, when generated in parallel,
        the vectors may be unexpectedly split into multiple rows (one per proc).
        """

        with open(fname, "r") as file:
            lines = file.readlines()

        vector = np.hstack([np.fromstring(line.strip(), sep=' ') for line in lines])

        return vector

    def _load_FEM_matrix_file(self, fname, matrixtype=csc_matrix, save=True, remove=True):
        """
        Read a FEM matrix (sparse)
        """

        assert matrixtype in {lil_matrix, csr_matrix, csc_matrix, coo_matrix}, f"Invalid matrix type: {matrixtype}"

        try:
            cur_data = load_npz(fname + '.npz')
        except (IOError, FileNotFoundError):
            # loading .txt files
            cur_data = arr_utils.read_matrix(fname + '.m', matrixtype=matrixtype)
            # saving in .npy format
            if save:
                save_npz(fname + '.npz', cur_data, compressed=True)
            # deleting .txt files
            if remove:
                os.remove(fname + '.m')

        return cur_data

    def _import_parameters(self, _ns=None, is_test=False):
        """
        MODIFY
        """

        if not is_test:
            snap_params = self.M_offline_ns_parameters
            path = self.M_snapshots_path
        else:
            snap_params = self.M_test_offline_ns_parameters
            path = self.M_test_snapshots_path

        if not arr_utils.is_empty(snap_params):
            self._compute_parameter_bounds(is_test=is_test)
            logger.warning(f"{snap_params.shape[0]} parameters are already available. No more are imported.")
            return True

        count = 0
        if _ns is None:
            _ns = 0
            while os.path.isdir(os.path.join(path, f'param{_ns}')):
                _ns += 1

        for i in range(_ns):
            fname = os.path.join(path, f'param{i}', 'coeffile.txt')

            try:
                cur_param = np.genfromtxt(fname, delimiter=',')
                if arr_utils.is_empty(snap_params):
                    snap_params = np.expand_dims(cur_param, 0)
                else:
                    snap_params = np.concatenate((snap_params, np.expand_dims(cur_param, 0)), axis=0)
                count += 1
            except (OSError, IOError, FileNotFoundError):
                logger.warning(f"Impossible to load parameter values for snapshot number {i}")

        if count == 0:
            logger.warning("No snapshots available!")
            return False

        logger.info(f"Number of read parameter values : {count}")
        logger.debug(f"Size of parameters matrix: {snap_params.shape}")

        if not is_test:
            self.M_offline_ns_parameters = snap_params
        else:
            self.M_test_offline_ns_parameters = snap_params

        self._compute_parameter_bounds(is_test=is_test)

        return True

    def _compute_parameter_bounds(self, is_test=False):
        """
        MODIFY
        """

        if arr_utils.is_empty(self.M_offline_ns_parameters):
            logger.warning("Impossible to compute parameter bounds if offline parameter values are not available.")
            return

        parameters = self.M_test_offline_ns_parameters if is_test else self.M_offline_ns_parameters

        min_param, max_param = np.min(parameters, axis=0), np.max(parameters, axis=0)
        self.M_shift_param = np.array([min_param[k] if np.abs(max_param - min_param)[k] > 1e-10 else 0.0
                                       for k in range(min_param.shape[0])])
        self.M_scale_param = np.array([max_param[k] - min_param[k] if np.abs(max_param - min_param)[k] > 1e-10 else 1.0
                                       for k in range(min_param.shape[0])])

        return

    def _normalize_parameter(self, param, erase_constant=True, idxs=slice(None)):
        """
        MODIFY
        """

        assert len(param.shape) == 2, "The parameter in input should be two dimensional"

        new_param = (param[:, idxs] - self.M_shift_param[None, idxs]) / self.M_scale_param[None, idxs]

        if erase_constant:
            idx = np.where(self.M_scale_param == 1.0)[0]
            tmp = np.arange(self.M_scale_param.shape[0])
            if idx.shape != tmp.shape or np.any(idx != tmp):
                new_param = np.delete(new_param, idx, axis=1)

        return new_param

    def _import_snapshots(self, _ns=None, is_test=False, ss_ratio=1):
        """
        MODIFY
        """

        start1 = time.time()

        assert self.M_import_snapshots, "Snapshots import disabled"

        snap_mat = self.M_snapshots_matrix if not is_test else self.M_test_snapshots_matrix
        snap_mat_IC = self.M_snapshots_IC if not is_test else self.M_test_snapshots_IC
        path = self.M_snapshots_path if not is_test else self.M_test_snapshots_path

        logger.info(f"Reading {_ns} snapshots ...")

        count = 0
        if _ns is None:
            _ns = 0
            while os.path.isdir(os.path.join(path, f'param{_ns}')):
                _ns += 1

        snap_mat['velocity'], snap_mat['pressure'] = [], []
        snap_mat['lambda'] = {n: [] for n in range(self.M_n_coupling)}

        snap_mat_IC['velocity'], snap_mat_IC['pressure'] = [], []
        snap_mat_IC['lambda'] = {n: [] for n in range(self.M_n_coupling)}

        snap_params = []

        for i in range(_ns):

            cur_path = os.path.join(path, f'param{i}')
            is_h5 = os.path.isfile(os.path.join(cur_path, 'block0.h5'))

            if is_h5 and not os.path.isdir(os.path.join(path, 'Template')):
                gen_utils.create_dir(os.path.join(path, 'Template'))
                shutil.copy(os.path.join(cur_path, 'block0.h5'), os.path.join(path, 'Template', 'block0.h5'))
                shutil.copy(os.path.join(cur_path, 'block0.xmf'), os.path.join(path, 'Template', 'block0.xmf'))

            try:
                tmp_u, tmp_u_IC = self._load_snapshot_file(cur_path, 'velocity',save=True, remove=not is_h5)
                tmp_p, tmp_p_IC = self._load_snapshot_file(cur_path, 'pressure',save=True, remove=True)

                has_IC = tmp_u_IC is not None

                snap_mat['velocity'].append(tmp_u[ss_ratio-1::ss_ratio].T)
                snap_mat['pressure'].append(tmp_p[ss_ratio-1::ss_ratio].T)
                if has_IC:
                    snap_mat_IC['velocity'].append(tmp_u_IC[-1])
                    snap_mat_IC['pressure'].append(tmp_p_IC[-1])

                for n in range(self.M_n_coupling):
                    tmp_l, tmp_l_IC = self._load_snapshot_file(cur_path, f'lambda{n}', save=True, remove=True)

                    snap_mat['lambda'][n].append(tmp_l[ss_ratio-1::ss_ratio].T)
                    if has_IC:
                        snap_mat_IC['lambda'][n].append(tmp_l_IC[-1])

                snap_params.append(np.genfromtxt(os.path.join(cur_path, 'coeffile.txt'), delimiter=',')[None])

                count += 1

            except (OSError, IOError, FileNotFoundError):
                logger.warning(f"Impossible to load files for snapshot number {i}")

        if count == 0:
            logger.warning("No snapshots available!")
            return False

        assert snap_mat['velocity'][-1].shape[1] % self.M_N_periods == 0, \
            (f"The number of time instants ({snap_mat['velocity'][-1].shape[1]}) is not a multiple of "
             f"the number of periods {self.M_N_periods}")
        self.M_Nt = (snap_mat['velocity'][-1].shape[1]) // self.M_N_periods

        snap_params = np.vstack(snap_params)
        for field in snap_mat:
            if type(snap_mat[field]) is list:
                snap_mat[field] = np.hstack(snap_mat[field])
                if has_IC:
                    snap_mat_IC[field] = np.vstack(snap_mat_IC[field])
            elif type(snap_mat[field]) is dict:
                for n in snap_mat[field]:
                    snap_mat[field][n] = np.hstack(snap_mat[field][n])
                    if has_IC:
                        snap_mat_IC[field][n] = np.vstack(snap_mat_IC[field][n])

        if not has_IC:
            for field in set(self.M_valid_fields) - {'lambda'}:
                snap_mat_IC[field] = np.zeros(0)
            for n in range(self.M_n_coupling):
                snap_mat_IC['lambda'][n] = np.zeros(0)

        logger.info(f"Number of read snapshots : {count}")

        logger.debug(f"Size of velocity snapshots matrix : {snap_mat['velocity'].shape}")
        logger.debug(f"Size of pressure snapshots matrix : {snap_mat['pressure'].shape}")
        for n in range(self.M_n_coupling):
            logger.debug(f"Size of multipliers {n} snapshots matrix : {snap_mat['lambda'][n].shape}")
        logger.debug(f"Size of parameters matrix: {snap_params.shape}")

        if not is_test:
            self.M_offline_ns_parameters = snap_params
        else:
            self.M_test_offline_ns_parameters = snap_params

        self._compute_parameter_bounds(is_test=is_test)

        logger.debug(f"Snapshots importing duration: {(time.time() - start1):.4f} s")

        return True

    def import_snapshots_matrix(self, _ns=None, ss_ratio=1):
        """
        MODIFY
        """

        logger.info("Importing snapshots")

        import_success = self._import_snapshots(_ns=_ns, is_test=False, ss_ratio=ss_ratio)

        self.M_Nh['velocity'] = self.M_snapshots_matrix['velocity'].shape[0]
        self.M_ns = self.M_snapshots_matrix['velocity'].shape[1] // (self.M_Nt * self.M_N_periods)
        self.M_Nh['pressure'] = self.M_snapshots_matrix['pressure'].shape[0]
        self.M_Nh['lambda'] = np.array([self.M_snapshots_matrix['lambda'][n].shape[0]
                                        for n in range(self.M_n_coupling)]).astype(int)
        self.M_N_params = self.M_offline_ns_parameters.shape[1]

        return import_success

    def import_test_snapshots_matrix(self, _ns=None, ss_ratio=1):
        """
        MODIFY
        """

        logger.info("Importing test snapshots")

        import_success = self._import_snapshots(_ns=_ns, is_test=True, ss_ratio=ss_ratio)

        self.M_Nh['velocity'] = self.M_test_snapshots_matrix['velocity'].shape[0]
        self.M_ns_test = int(self.M_test_snapshots_matrix['velocity'].shape[1] / (self.M_Nt * self.M_N_periods))
        self.M_Nh['pressure'] = self.M_test_snapshots_matrix['pressure'].shape[0]
        self.M_Nh['lambda'] = np.array([self.M_test_snapshots_matrix['lambda'][n].shape[0]
                                        for n in range(self.M_n_coupling)]).astype(int)
        self.M_N_params = self.M_test_offline_ns_parameters.shape[1]

        return import_success

    def get_snapshot(self, _snapshot_number, _fom_coordinates=np.array([]), timesteps=None,
                     field='velocity', n=0):
        """
        MODIFY
        """

        if field == 'lambda':
            assert n <= self.M_n_coupling, f"Invalid coupling index {n} for Lagrange multiplier snapshot"
            return self._get_snapshot(self.M_snapshots_matrix['lambda'][n], _snapshot_number,
                                      _fom_coordinates=_fom_coordinates, timesteps=timesteps)
        else:
            return self._get_snapshot(self.M_snapshots_matrix[field], _snapshot_number,
                                      _fom_coordinates=_fom_coordinates, timesteps=timesteps)

    def get_test_snapshot(self, _snapshot_number, _fom_coordinates=np.array([]), timesteps=None,
                          field='velocity', n=0):
        """
        MODIFY
        """

        if field == 'lambda':
            assert n <= self.M_n_coupling, f"Invalid coupling index {n} for Lagrange multiplier snapshot"
            return self._get_snapshot(self.M_test_snapshots_matrix['lambda'][n], _snapshot_number,
                                      _fom_coordinates=_fom_coordinates, timesteps=timesteps)
        else:
            return self._get_snapshot(self.M_test_snapshots_matrix[field], _snapshot_number,
                                      _fom_coordinates=_fom_coordinates, timesteps=timesteps)

    def get_snapshot_function(self, _snapshot_number, _fom_coordinates=np.array([]), timesteps=None,
                              field='velocity', n=0):
        """
        MODIFY
        """

        if self.M_get_test:
            return self.get_test_snapshot(_snapshot_number,
                                          _fom_coordinates=_fom_coordinates, timesteps=timesteps, field=field, n=n)
        else:
            return self.get_snapshot(_snapshot_number,
                                     _fom_coordinates=_fom_coordinates, timesteps=timesteps, field=field, n=n)

    def get_norm_matrices(self, matrixtype=csc_matrix, check_spd=False):
        """
        MODIFY
        """
        logger.debug("Importing the norm matrices")

        self.M_norm_matrices_bcs, self.M_norm_matrices = dict(), dict()

        self.M_norm_matrices_bcs['velocity'] = self._load_FEM_matrix_file(os.path.join(self.M_fom_structures_path,
                                                                                       'norm0_bcs'),
                                                                          matrixtype=matrixtype,
                                                                          save=True, remove=True)

        self.M_norm_matrices['velocity'] = self._load_FEM_matrix_file(os.path.join(self.M_fom_structures_path,
                                                                                   'norm0'),
                                                                      matrixtype=matrixtype,
                                                                      save=True, remove=True)
        self.M_norm_matrices['pressure'] = self._load_FEM_matrix_file(os.path.join(self.M_fom_structures_path,
                                                                                   'norm1'),
                                                                      matrixtype=matrixtype,
                                                                      save=True, remove=True)

        if check_spd:
            for field in self.M_norm_matrices:
                if not (arr_utils.is_symmetric(self.M_norm_matrices[field]) and
                        arr_utils.is_positive_definite(self.M_norm_matrices[field])):
                    logger.warning(f"The {field} norm matrix is not SPD!")

        return

    def check_norm_matrices(self):
        """
        MODIFY
        """
        return ('velocity' in self.M_norm_matrices.keys()) and \
               (self.M_norm_matrices['velocity'].shape[0] > 0) and \
               ('pressure' in self.M_norm_matrices.keys()) and \
               (self.M_norm_matrices['pressure'].shape[0] > 0) and \
               ('velocity' in self.M_norm_matrices_bcs.keys()) and \
               (self.M_norm_matrices_bcs['velocity'].shape[0] > 0)

    def compute_norm(self, vec, field='velocity'):
        """
        MODIFY
        """
        if not self.check_norm_matrices():
            self.get_norm_matrices()

        vec = vec.squeeze()
        if len(vec.shape) == 1:
            return arr_utils.mynorm(vec, self.M_norm_matrices[field] if field in self.M_norm_matrices else None)
        elif len(vec.shape) == 2:
            return arr_utils.mynorm(vec.T, self.M_norm_matrices[field] if field in self.M_norm_matrices else None)
        else:
            raise ValueError(f"Invalid shape for the input vector: {vec.shape}")

    def __compute_IC_idxs(self):
        """Compute the indices of the initial conditions in the snapshots matrix."""

        M = self.M_snapshots_matrix['velocity'].shape[1]
        idxs = np.arange(0, M + 1, self.M_Nt)
        idxs = np.array([idx for idx in idxs if idx % (self.M_Nt * self.M_N_periods)])

        return idxs

    def _subtract_IC(self, field="velocity", n=0, include_IC=False):
        """
        Subtract initial condition from snapshots spanning more than one period.
        """

        snap_mat = self.M_snapshots_matrix[field] if field != "lambda" else self.M_snapshots_matrix['lambda'][n]

        idxs = self.__compute_IC_idxs()
        ICs = snap_mat[:, idxs - 1] if idxs.size else []

        if include_IC:
            snap_mat_IC = self.M_snapshots_IC[field] if field != "lambda" else self.M_snapshots_IC['lambda'][n]
            if snap_mat_IC.size:
                idxs_IC = np.array([k * self.M_Nt * self.M_N_periods + np.arange(self.M_Nt) for k in range(self.M_ns)])
                for k,idx_IC in enumerate(idxs_IC):
                    snap_mat[:, idx_IC] -= snap_mat_IC[k, :, None]

        if self.M_N_periods == 1:
            return None

        for (idx, IC) in zip(idxs, ICs.T):
            snap_mat[:, idx:idx + self.M_Nt] -= IC[:, None]

        return ICs

    def _restore_IC(self, ICs, field="velocity", n=0, include_IC=False):
        """
        Restore initial condition from snapshots spanning more than one period.
        """

        snap_mat = self.M_snapshots_matrix[field] if field != "lambda" else self.M_snapshots_matrix['lambda'][n]

        if include_IC:
            snap_mat_IC = self.M_snapshots_IC[field] if field != "lambda" else self.M_snapshots_IC['lambda'][n]
            if snap_mat_IC.size:
                idxs_IC = np.array([k * self.M_Nt * self.M_N_periods + np.arange(self.M_Nt) for k in range(self.M_ns)])
                for k,idx_IC in enumerate(idxs_IC):
                    snap_mat[:, idx_IC] += snap_mat_IC[k, :, None]

        if self.M_N_periods == 1 or ICs is None:
            return

        idxs = self.__compute_IC_idxs()

        for (idx, IC) in zip(idxs, ICs.T):
            snap_mat[:, idx:idx + self.M_Nt] += IC[:, None]

        return

    def perform_pod_space(self, _tol=1e-3, field="velocity"):
        """
        Perform the POD in space for the given field.
        """

        logger.info(f"Performing the POD in space for {field}, using a tolerance of {_tol:.2e}")

        if not self.check_norm_matrices():
            self.get_norm_matrices()

        pod = podec.ProperOrthogonalDecomposition()

        if field == 'lambda':
            self.M_N_space['lambda'] = np.zeros(self.M_n_coupling, dtype=int)
            self.M_basis_space['lambda'] = [np.zeros(0)] * self.M_n_coupling
            self.M_sv_space['lambda'] = [np.zeros(0)] * self.M_n_coupling
            for n in range(self.M_n_coupling):
                ICs = self._subtract_IC(field='lambda', n=n, include_IC=True)
                snap_mat = (self.M_snapshots_matrix['lambda'][n] if self.M_Nt_IC == 0 else
                            np.hstack([self.M_snapshots_matrix['lambda'][n], self.M_snapshots_IC['lambda'][n].T]))
                pod(snap_mat, _tol=_tol, _norm_matrix=None)
                self._restore_IC(ICs, field='lambda', n=n, include_IC=True)
                self.M_basis_space['lambda'][n], self.M_sv_space['lambda'][n] = pod.basis, pod.singular_values
                self.M_N_space['lambda'][n] = self.M_basis_space['lambda'][n].shape[1]
        else:
            ICs = self._subtract_IC(field=field, include_IC=True)
            snap_mat = (self.M_snapshots_matrix[field] if self.M_Nt_IC == 0 else
                        np.hstack([self.M_snapshots_matrix[field], self.M_snapshots_IC[field].T]))
            pod(snap_mat, _tol=_tol, _norm_matrix=self.M_norm_matrices[field])
            self._restore_IC(ICs, field=field, include_IC=True)
            self.M_basis_space[field], self.M_sv_space[field] = pod.basis, pod.singular_values
            self.M_N_space[field] = self.M_basis_space[field].shape[1]

        return

    def __time_unfold_matrix(self, field='velocity', n=0, method='reduced'):
        """Unfold the snapshots' matrix to enable POD in time."""

        Nh = self.M_Nh[field][n] if field == 'lambda' else self.M_Nh[field]
        N_space = self.M_N_space[field][n] if field == 'lambda' else self.M_N_space[field]
        snap_mat = self.M_snapshots_matrix[field][n] if field == 'lambda' else self.M_snapshots_matrix[field]

        _Nt = self.M_Nt * self.M_N_periods

        if method == 'full':
            _Nh = Nh * self.M_N_periods
            ret_mat = np.zeros((self.M_Nt, _Nh * self.M_ns))
            for iNs in range(self.M_ns):
                tmp = snap_mat[:, iNs * _Nt:(iNs + 1) * _Nt].T
                ret_mat[:, iNs * _Nh:(iNs + 1) * _Nh] = np.reshape(tmp, (self.M_Nt, _Nh), order='F')
        elif method == 'reduced':
            _Nh = N_space * self.M_N_periods
            ret_mat = np.zeros((self.M_Nt, _Nh * self.M_ns))
            for iNs in range(self.M_ns):
                tmp = snap_mat[:, iNs * _Nt:(iNs + 1) * _Nt].T.dot(self.M_basis_space[field])
                ret_mat[:, iNs * _Nh:(iNs + 1) * _Nh] = np.reshape(tmp, (self.M_Nt, _Nh), order='F')
        else:
            raise ValueError(f"Unrecognized method {method} for the temporal POD")

        return ret_mat

    def perform_pod_time(self, _tol=1e-3, method='reduced', field="velocity"):
        """
        Perform the POD in time for the given field.
        """

        logger.info(f"Performing the POD in time for {field}, using a tolerance of {_tol:.2e} "
                    f"and with the {method} method")

        _Nt = self.M_Nt * self.M_N_periods

        if field == 'lambda':
            self.M_N_time['lambda'] = np.zeros(self.M_n_coupling, dtype=int)
            self.M_basis_time['lambda'] = [np.zeros(0)] * self.M_n_coupling
            self.M_sv_time['lambda'] = [np.zeros(0)] * self.M_n_coupling
            for n in range(self.M_n_coupling):
                ICs = self._subtract_IC(field='lambda', n=n, include_IC=True)

                time_unfold_snapshots_matrix = self.__time_unfold_matrix(field='lambda', n=n, method=method)

                pod = podec.ProperOrthogonalDecomposition()
                pod(time_unfold_snapshots_matrix, _tol)

                self._restore_IC(ICs, field='lambda', n=n, include_IC=True)

                self.M_basis_time['lambda'][n], self.M_sv_time['lambda'][n] = pod.basis, pod.singular_values
                self.M_N_time['lambda'][n] = self.M_basis_time['lambda'][n].shape[1]

        else:
            ICs = self._subtract_IC(field=field, include_IC=True)

            time_unfold_snapshots_matrix = self.__time_unfold_matrix(field=field, method=method)

            pod = podec.ProperOrthogonalDecomposition()
            pod(time_unfold_snapshots_matrix, _tol)

            self._restore_IC(ICs, field=field, include_IC=True)

            self.M_basis_time[field], self.M_sv_time[field] = pod.basis, pod.singular_values
            self.M_N_time[field] = self.M_basis_time[field].shape[1]

        return

    @staticmethod
    def real_fourier_basis(N, Nt, add_constant=False, only_sin=False):
        """Build a discrete real-valued Fourier basis in 1D."""

        basis = []

        if only_sin:
            for k in range(1, N+1):
                sin_vector = np.sqrt(2 / Nt) * np.sin(np.pi * k * np.arange(Nt) / Nt)
                basis.append(sin_vector)

        else:

            if add_constant:
                basis.append(np.ones(Nt) / np.sqrt(Nt))
                N -= 1

            for k in range(1, N // 2 + 1):
                cos_vector = np.sqrt(2 / Nt) * np.cos(2 * np.pi * k * np.arange(Nt) / Nt)
                basis.append(cos_vector)
                sin_vector = np.sqrt(2 / Nt) * np.sin(2 * np.pi * k * np.arange(Nt) / Nt)
                basis.append(sin_vector)

            if N % 2:
                cos_vector = np.sqrt(2 / Nt) * np.cos(2 * np.pi * (N // 2 + 1) * np.arange(Nt) / Nt)
                basis.append(cos_vector)

        basis_matrix = np.vstack(basis).T

        return basis_matrix

    def time_basis_fourier(self, N, field="velocity"):
        """Use the discrete real-valued Fourier modes as reduced basis elements in time."""

        Vt = self.real_fourier_basis(N, self.M_Nt+1, add_constant=False, only_sin=True)[1:]  # keep only sines

        if field == "lambda":
            for n in range(self.M_n_coupling):
                self.M_basis_time[field][n], self.M_sv_time[field][n] = Vt, np.ones(N)
                self.M_N_time[field][n] = N
        else:
            self.M_basis_time[field], self.M_sv_time[field] = Vt, np.ones(N)
            self.M_N_time[field] = N

        return

    def primal_supremizers(self, stabilize=False):
        """
        MODIFY
        """

        if not self.check_norm_matrices():
            self.get_norm_matrices()

        logger.info("Adding primal supremizers in space")

        if stabilize:
            logger.debug("Computing the stabilized primal supremizers in space")
            FEM_matrices = self.import_FEM_structures(structures={'Bdiv', 'B'})
            primal_constraint_mat = FEM_matrices['BdivT']
            dual_constraint_mat_T = scipy.sparse.hstack([BTn for BTn in FEM_matrices['BT']], format='csc')
            dual_constraint_mat = scipy.sparse.vstack([Bn for Bn in FEM_matrices['B']], format='csc')

            lhs_mat = bmat([[self.M_norm_matrices_bcs['velocity'], dual_constraint_mat_T],
                            [dual_constraint_mat, None]],
                           format='csc')
            rhs_vec = np.vstack([primal_constraint_mat.dot(self.M_basis_space['pressure']),
                                 np.zeros((np.sum(self.M_Nh['lambda']), self.M_N_space['pressure']))])

        else:
            FEM_matrices = self.import_FEM_structures(structures={'Bdiv'})
            primal_constraint_mat = FEM_matrices['BdivT']
            lhs_mat = self.M_norm_matrices_bcs['velocity']
            rhs_vec = primal_constraint_mat.dot(self.M_basis_space['pressure'])

        supr_primal = arr_utils.solve_sparse_system(lhs_mat, rhs_vec)[:self.M_Nh['velocity']]
        if len(supr_primal.shape) == 1:
            supr_primal = supr_primal[..., None]

        for i in range(supr_primal.shape[1]):
            logger.debug(f"Normalizing primal supremizer {i}")

            for col in self.M_basis_space['velocity'].T:
                supr_primal[:, i] -= arr_utils.mydot(supr_primal[:, i], col, self.M_norm_matrices['velocity']) * col
            for j in range(i):
                supr_primal[:, i] -= arr_utils.mydot(supr_primal[:, i], supr_primal[:, j], self.M_norm_matrices['velocity']) * \
                                     supr_primal[:, j]

            supr_primal[:, i] /= self.compute_norm(supr_primal[:, i], field='velocity')

        supr_primal[np.abs(supr_primal) < 1e-15] = 0
        self.supr_primal = supr_primal

        return

    def time_supremizers(self, field='pressure', n=0, tol=5e-1):
        """
        MODIFY
        """

        if field == 'velocity':
            logger.warning("Invalid field 'velocity' for the temporal supremizers assembling")
            return
        elif field == 'pressure':
            basis_time = self.M_basis_time['pressure']
        elif field == 'lambda':
            if n in range(self.M_n_coupling):
                basis_time = self.M_basis_time['lambda'][n]
            else:
                raise ValueError(f"Invalid coupling index {n}. "
                                 f"The number of couplings for the current problem is {self.M_n_coupling}")
        else:
            raise ValueError(f"Invalid field {field}")

        prod_u_dual = np.dot(self.M_basis_time['velocity'].T, basis_time)
        prod_u_dual /= np.linalg.norm(prod_u_dual, axis=0)

        i = 1
        while i < prod_u_dual.shape[1]:
            original = np.copy(prod_u_dual[:, i])
            for j in range(i):
                prod_u_dual[:, i] -= np.dot(original, prod_u_dual[:, j]) * prod_u_dual[:, j]

            diff_norm = np.linalg.norm(prod_u_dual[:, i])
            if diff_norm <= tol:
                logger.warning(f"Critical index {i} for {field}{n if field == 'lambda' else ''} constraint - "
                               f"Norm: {diff_norm:.2e}. "
                               f"Adding the corresponding basis function to the velocity temporal basis.")
                self.M_basis_time['velocity'] = np.hstack((self.M_basis_time['velocity'], basis_time[:, i:i+1]))
                for k in range(self.M_basis_time['velocity'].shape[1] - 1):
                    self.M_basis_time['velocity'][:, -1] -= np.dot(self.M_basis_time['velocity'][:, -1],
                                                                   self.M_basis_time['velocity'][:, k]) * \
                                                self.M_basis_time['velocity'][:, k]
                self.M_basis_time['velocity'][:, -1] /= np.linalg.norm(self.M_basis_time['velocity'][:, -1])

                new_row = np.dot(self.M_basis_time['velocity'][:, -1], basis_time)
                prod_u_dual = np.vstack([prod_u_dual, new_row])
                prod_u_dual /= np.linalg.norm(prod_u_dual, axis=0)

                i = 1

            else:
                prod_u_dual[:, i] /= np.linalg.norm(prod_u_dual[:, i])
                i += 1

        self.M_N_time['velocity'] = self.M_basis_time['velocity'].shape[1]
        logger.debug(f"Size of velocity temporal basis after {field}{n if field == 'lambda' else ''} enrichment: "
                     f"{self.M_N_time['velocity']}")

        return

    def primal_supremizers_time(self, tol=5e-1):
        """
        MODIFY
        """
        logger.info("Adding primal stabilizers in time")
        logger.debug(f"Selected tolerance: {tol:.2f}")

        self.time_supremizers(field='pressure', tol=tol)

        return

    def dual_supremizers(self, stabilize=False):
        """
        MODIFY
        """

        if not self.check_norm_matrices():
            self.get_norm_matrices()

        logger.info("Adding dual supremizers in space")

        FEM_structures = {'Bdiv', 'B'} if stabilize else {'B'}
        FEM_matrices = self.import_FEM_structures(structures=FEM_structures)

        supr_dual = []
        for n in range(self.M_n_coupling):
            if stabilize:
                logger.debug("Computing the stabilized dual supremizers in space")
                lhs_mat = bmat([[self.M_norm_matrices_bcs['velocity'], FEM_matrices['BdivT']],
                                [FEM_matrices['Bdiv'], None]],
                               format='csc')
                rhs_vec = scipy.sparse.vstack([FEM_matrices['BT'][n].dot(self.M_basis_space['lambda'][n]),
                                               np.zeros((self.M_Nh['pressure'], self.M_N_space['lambda'][n]))])
            else:
                lhs_mat = self.M_norm_matrices_bcs['velocity']
                rhs_vec = FEM_matrices['BT'][n].dot(self.M_basis_space['lambda'][n])

            supr_dual_n = arr_utils.solve_sparse_system(lhs_mat, rhs_vec)[:self.M_Nh['velocity']]
            if len(supr_dual_n.shape) == 1:
                supr_dual_n = supr_dual_n[..., None]
            supr_dual.append(supr_dual_n)

        supr_dual = np.array(np.hstack([item for item in supr_dual]))

        for i in range(supr_dual.shape[1]):
            logger.debug(f"Normalizing dual supremizer {i}")

            for col in self.M_basis_space['velocity'].T:
                supr_dual[:, i] -= arr_utils.mydot(supr_dual[:, i], col, self.M_norm_matrices['velocity']) * col

            for col in self.supr_primal.T:
                supr_dual[:, i] -= arr_utils.mydot(supr_dual[:, i], col, self.M_norm_matrices['velocity']) * col

            for j in range(i):
                supr_dual[:, i] -= arr_utils.mydot(supr_dual[:, i], supr_dual[:, j], self.M_norm_matrices['velocity']) * \
                                   supr_dual[:, j]

            supr_dual[:, i] /= self.compute_norm(supr_dual[:, i],field='velocity')

        supr_dual[np.abs(supr_dual) < 1e-15] = 0
        self.supr_dual = supr_dual

        return

    def dual_supremizers_time(self, tol=5e-1):
        """
        MODIFY
        """
        logger.debug("Adding dual stabilizers in time")
        logger.debug(f"Selected tolerance: {tol:.2f}")

        for n in range(self.M_n_coupling):
            self.time_supremizers(field='lambda', n=n, tol=tol)

        return

    def build_IC_basis_elements(self):
        """
        Build quantities needed to enforce initial conditions.
        """
        return

    def build_ST_basis(self, *args, **kwargs):
        """
        MODIFY
        """
        raise NotImplementedError("This method is not implemented in this class!")

    def check_build_ST_basis(self):
        """
        MODIFY
        """

        return (self.M_basis_space['velocity'].shape[0] != 0 and self.M_basis_time['velocity'].shape[0] != 0 and
                self.M_basis_space['pressure'].shape[0] != 0 and self.M_basis_time['pressure'].shape[0] != 0 and
                all([self.M_basis_space['lambda'][n].shape[0] != 0 and self.M_basis_time['lambda'][n].shape[0] != 0
                    for n in range(self.M_n_coupling)]))

    def _save_bases(self, _field, _type, n=0):
        """
        MODIFY
        """

        assert _type in {'space', 'time'}

        basis = self.M_basis_space if _type == 'space' else self.M_basis_time if _type == 'time' else dict()
        sv = self.M_sv_space if _type == 'space' else self.M_sv_time if _type == 'time' else dict()

        assert _field in basis and _field in sv

        path = os.path.join(self.M_basis_path, f"{_field}{n}" if _field == 'lambda' else f"{_field}")

        if _field == 'lambda':
            np.save(os.path.join(path, f'{_type}_basis.npy'), basis[_field][n])
            np.save(os.path.join(path, f'{_type}_sv.npy'), sv[_field][n])
        else:
            np.save(os.path.join(path, f'{_type}_basis.npy'), basis[_field])
            np.save(os.path.join(path, f'{_type}_sv.npy'), sv[_field])

        # these are necessary to assemble the affine components for the convective term in NS
        if _type == 'space' and _field in {'velocity', 'pressure'}:
            os.makedirs(os.path.join(self.M_basis_path, self.M_mesh_name), exist_ok=True)
            cnt = 0 if _field == 'velocity' else 1
            np.savetxt(os.path.join(self.M_basis_path, self.M_mesh_name, f'field{cnt}.basis'),
                       self.M_basis_space[_field].T, fmt='%.16g', delimiter=',')
            np.savetxt(os.path.join(self.M_basis_path, self.M_mesh_name, f'svd{cnt}.txt'),
                       self.M_sv_space[_field], fmt='%.16g', delimiter=',')

        return

    def save_ST_basis(self, which=None):
        """
        MODIFY
        """

        if which is None:
            which = {'velocity-space', 'velocity-time',
                     'pressure-space', 'pressure-time',
                     'lambda-space', 'lambda-time'}

        if self.M_save_offline_structures and which:
            logger.debug("Dumping ST bases to file ...")

            gen_utils.create_dir(os.path.join(self.M_basis_path, 'velocity'))
            gen_utils.create_dir(os.path.join(self.M_basis_path, 'pressure'))
            gen_utils.create_dir(os.path.join(self.M_basis_path, self.M_mesh_name))

            if 'velocity-space' in which:
                self._save_bases('velocity', 'space')
            if 'pressure-space' in which:
                self._save_bases('pressure', 'space')
            if 'velocity-time' in which:
                self._save_bases('velocity', 'time')
            if 'pressure-time' in which:
                self._save_bases('pressure', 'time')
            if {'lambda-space', 'lambda-time'} & which:
                for n in range(self.M_n_coupling):
                    cur_path = os.path.join(self.M_basis_path, os.path.normpath('lambda' + str(n) + '/'))
                    gen_utils.create_dir(cur_path)
                    if 'lambda-space' in which:
                        self._save_bases('lambda', 'space', n=n)
                    if 'lambda-time' in which:
                        self._save_bases('lambda', 'time', n=n)

        return

    def import_basis_space_matrix(self, field="velocity"):
        """MODIFY
        """

        assert self.M_import_offline_structures, "Offline structures import is disabled"

        logger.info(f"Importing reduced basis in space for {field}")
        try:

            if field == 'lambda':
                self.M_N_space['lambda'] = np.zeros(self.M_n_coupling, dtype=int)
                self.M_basis_space['lambda'] = [np.zeros(0)] * self.M_n_coupling
                self.M_sv_space['lambda'] = [np.zeros(0)] * self.M_n_coupling
                for n in range(self.M_n_coupling):
                    path = os.path.join(self.M_basis_path, f"{field}{n}")
                    self.M_basis_space['lambda'][n] = np.load(os.path.join(path, 'space_basis.npy'))
                    self.M_sv_space['lambda'][n] = np.load(os.path.join(path, 'space_sv.npy'))
                    if len(self.M_basis_space['lambda'][n].shape) == 1:
                        self.M_basis_space['lambda'][n] = np.expand_dims(self.M_basis_space['lambda'][n], axis=1)
                    if self.M_Nh['lambda'].shape[0] < self.M_n_coupling:
                        self.M_Nh['lambda'] = np.append(self.M_Nh['lambda'], self.M_basis_space['lambda'][n].shape[0])
                    self.M_N_space['lambda'][n] = self.M_basis_space['lambda'][n].shape[1]
            else:
                path = os.path.join(self.M_basis_path, field)
                self.M_basis_space[field] = np.load(os.path.join(path, 'space_basis.npy'))
                self.M_sv_space[field] = np.load(os.path.join(path, 'space_sv.npy'))
                if len(self.M_basis_space[field].shape) == 1:
                    self.M_basis_space[field] = np.expand_dims(self.M_basis_space[field], axis=1)
                self.M_Nh[field] = self.M_basis_space[field].shape[0]
                self.M_N_space[field] = self.M_basis_space[field].shape[1]

            import_success = True

        except (IOError, OSError, FileNotFoundError) as e:
            logger.error(f"Error {e}: impossible to load the space basis matrix for {field}")
            import_success = False

        return import_success

    def import_basis_time_matrix(self, field="velocity"):
        """MODIFY
        """

        assert self.M_import_offline_structures, "Offline structures import is disabled"

        logger.info(f"Importing reduced basis in time for {field}")
        try:

            if field == 'lambda':
                self.M_N_time['lambda'] = np.zeros(self.M_n_coupling, dtype=int)
                self.M_basis_time['lambda'] = [np.zeros(0)] * self.M_n_coupling
                self.M_sv_time['lambda'] = [np.zeros(0)] * self.M_n_coupling
                for n in range(self.M_n_coupling):
                    path = os.path.join(self.M_basis_path, f"{field}{n}")
                    self.M_basis_time['lambda'][n] = np.load(os.path.join(path, 'time_basis.npy'))
                    self.M_sv_time['lambda'][n] = np.load(os.path.join(path, 'time_sv.npy'))
                    if len(self.M_basis_time['lambda'][n].shape) == 1:
                        self.M_basis_time['lambda'][n] = np.expand_dims(self.M_basis_time['lambda'][n], axis=1)
                    self.M_Nt = self.M_basis_time['lambda'][n].shape[0]
                    self.M_N_time['lambda'][n] = self.M_basis_time['lambda'][n].shape[1]
            else:
                path = os.path.join(self.M_basis_path, field)
                self.M_basis_time[field] = np.load(os.path.join(path, 'time_basis.npy'))
                self.M_sv_time[field] = np.load(os.path.join(path, 'time_sv.npy'))
                if len(self.M_basis_time[field].shape) == 1:
                    self.M_basis_time[field] = np.expand_dims(self.M_basis_time[field], axis=1)
                self.M_Nt = self.M_basis_time[field].shape[0]
                self.M_N_time[field] = self.M_basis_time[field].shape[1]

            import_success = True

        except (IOError, OSError, FileNotFoundError) as e:
            logger.error(f"Error {e}: impossible to load the time basis matrix for {field}")
            import_success = False

        return import_success

    def import_snapshots_basis(self):
        """
        MODIFY
        """

        assert self.M_import_offline_structures, "Offline structures import is disabled"

        import_failures_basis = set()

        import_failures_basis.add('velocity-space' if not self.import_basis_space_matrix(field="velocity") else None)
        import_failures_basis.add('velocity-time' if not self.import_basis_time_matrix(field="velocity") else None)
        import_failures_basis.add('pressure-space' if not self.import_basis_space_matrix(field="pressure") else None)
        import_failures_basis.add('pressure-time' if not self.import_basis_time_matrix(field="pressure") else None)
        import_failures_basis.add('lambda-space' if not self.import_basis_space_matrix(field="lambda") else None)
        import_failures_basis.add('lambda-time' if not self.import_basis_time_matrix(field="lambda") else None)

        import_failures_basis = set(filter(None, import_failures_basis))

        if not import_failures_basis:
            self.build_IC_basis_elements()

        if not({'velocity-space', 'velocity-time'} & import_failures_basis):
            self.M_N['velocity'] = self.M_N_space['velocity'] * self.M_N_time['velocity']
        if not ({'pressure-space', 'pressure-time'} & import_failures_basis):
            self.M_N['pressure'] = self.M_N_space['pressure'] * self.M_N_time['pressure']
        if not ({'lambda-space', 'lambda-time'} & import_failures_basis):
            self.M_N['lambda'] = self.M_N_space['lambda'] * self.M_N_time['lambda']
            self.M_N_lambda_cumulative = np.cumsum(np.vstack([self.M_N['lambda'][n] for n in range(self.M_n_coupling)]))
            self.M_N_lambda_cumulative = np.insert(self.M_N_lambda_cumulative, 0, 0)

        return import_failures_basis

    def import_FEM_structures(self, structures=None, matrixtype=csc_matrix):
        """
        MODIFY
        """

        if structures is None:
            structures = {'A', 'M', 'Bdiv', 'B', 'RHS', 'q', 'R'}
        if 'clot' in self.M_parametrizations:
            structures.add('Mclot')

        assert len(structures.intersection({'A', 'M', 'Bdiv'})) > 0 or self.M_Nh['velocity'] > 0

        matrices = dict()

        try:
            if 'A' in structures:
                logger.debug("Importing stiffness matrix...")
                matrices['A'] = self._load_FEM_matrix_file(os.path.join(self.M_fom_structures_path, "A"),
                                                           matrixtype=matrixtype,
                                                           save=True, remove=True)
                self.M_Nh['velocity'] = matrices['A'].shape[0]
            if 'M' in structures:
                logger.debug("Importing mass matrix...")
                matrices['M'] = self._load_FEM_matrix_file(os.path.join(self.M_fom_structures_path, "M"),
                                                           matrixtype=matrixtype,
                                                           save=True, remove=True)
                self.M_Nh['velocity'] = matrices['M'].shape[0]
            if 'Bdiv' in structures:
                logger.debug("Importing divergence matrix...")
                matrices['Bdiv'] = self._load_FEM_matrix_file(os.path.join(self.M_fom_structures_path, "Bdiv"),
                                                              matrixtype=matrixtype,
                                                              save=True, remove=True)
                logger.debug("Importing transposed divergence matrix...")
                matrices['BdivT'] = self._load_FEM_matrix_file(os.path.join(self.M_fom_structures_path, "BdivT"),
                                                               matrixtype=matrixtype,
                                                               save=True, remove=True)
                self.M_Nh['velocity'] = matrices['BdivT'].shape[0]
                self.M_Nh['pressure'] = matrices['Bdiv'].shape[0]

            if 'B' in structures:
                B = [np.zeros(0)] * self.M_n_coupling
                BT = [np.zeros(0)] * self.M_n_coupling
                for n in range(self.M_n_coupling):
                    logger.debug(f"Importing coupling matrix {n} ...")
                    B[n] = self._load_FEM_matrix_file(os.path.join(self.M_fom_structures_path, "B" + str(n)),
                                                      matrixtype=matrixtype,
                                                      save=True, remove=True)
                    logger.debug(f"Importing transposed coupling matrix {n} ...")
                    BT[n] = self._load_FEM_matrix_file(os.path.join(self.M_fom_structures_path, "BT" + str(n)),
                                                       matrixtype=matrixtype,
                                                       save=True, remove=True)

                    if self.M_Nh['velocity'] != BT[n].shape[0]:
                        pad = csr_matrix((self.M_Nh['velocity'] - BT[n].shape[0], BT[n].shape[1]))
                        BT[n] = scipy.sparse.vstack([BT[n], pad])
                    if self.M_Nh['velocity'] != B[n].shape[1]:
                        pad = csr_matrix((B[n].shape[0], self.M_Nh['velocity'] - B[n].shape[1]))
                        B[n] = scipy.sparse.hstack([B[n], pad])

                    if self.M_Nh['lambda'].shape[0] < self.M_n_coupling:
                        self.M_Nh['lambda'] = np.append(self.M_Nh['lambda'], B[n].shape[0])

                matrices['B'] = B
                matrices['BT'] = BT

            if 'RHS' in structures:
                RHS = [np.zeros(0)] * self.M_n_coupling
                for n in range(self.M_n_coupling):
                    logger.debug("Importing RHS vector...")
                    RHS[n] = self._load_FEM_vector_file(os.path.join(self.M_fom_structures_path, f"RHS{n}.m"))
                matrices['RHS'] = RHS

            if 'u0' in structures:
                logger.debug("Importing the velocity initial condition (inferred from the first snapshot)...")
                fname = os.path.join(self.M_snapshots_path, 'param0', 'field0_IC.snap')
                matrices['u0'] = np.genfromtxt(fname, delimiter=',')

            if 'Mclot' in structures:
                logger.debug("Importing clot reaction mass matrix ...")
                path = lambda cnt: os.path.join(self.M_fom_structures_path, f"Mclot{cnt}")
                matrices['Mclot'] = []
                k = 0
                assert os.path.isfile(path(k) + ".m") or os.path.isfile(path(k) + ".npz"), "No clot matrices available"
                while os.path.isfile(path(k) + ".m") or os.path.isfile(path(k) + ".npz"):
                    matrices['Mclot'].append(self._load_FEM_matrix_file(path(k), matrixtype=matrixtype,
                                                                        save=True, remove=True))
                    k += 1

            if 'R' in structures:
                path_R = lambda cnt: os.path.join(self.M_fom_structures_path, f"R{cnt}")
                path_Radd = lambda cnt: os.path.join(self.M_fom_structures_path, f"R_add{cnt}")

                tmp_matrices_R, tmp_matrices_Radd = [], []
                k = 0
                while os.path.isfile(path_R(k) + ".m") or os.path.isfile(path_R(k) + ".npz"):
                    tmp_matrices_R.append(self._load_FEM_matrix_file(path_R(k), matrixtype=matrixtype,
                                                                     save=True, remove=True))
                    tmp_matrices_Radd.append(self._load_FEM_matrix_file(path_Radd(k), matrixtype=matrixtype,
                                                                        save=True, remove=True))
                    k += 1

                if tmp_matrices_R:
                    matrices['R'] = sum(tmp_matrices_R)
                if tmp_matrices_Radd:
                    matrices['Radd'] = sum(tmp_matrices_Radd)

            if 'q' in structures:
                logger.debug("Importing flow rate vectors ...")
                path_in = lambda cnt: os.path.join(self.M_fom_structures_path, f"q_in{cnt}.m")
                matrices['q_in'] = []
                k_in = 0
                assert os.path.isfile(path_in(k_in)), "No inflow rate vectors available"
                while os.path.isfile(path_in(k_in)):
                    matrices['q_in'].append(self._load_FEM_vector_file(path_in(k_in)))
                    k_in += 1
                self.M_n_inlets = k_in

                path_out = lambda cnt: os.path.join(self.M_fom_structures_path, f"q_out{cnt}.m")
                matrices['q_out'] = []
                k_out = 0
                assert os.path.isfile(path_out(k_out)), "No outflow rate vectors available"
                while os.path.isfile(path_out(k_out)):
                    matrices['q_out'].append(self._load_FEM_vector_file(path_out(k_out)))
                    k_out += 1
                self.M_n_outlets = k_out

        except (IOError, OSError, FileNotFoundError, AssertionError) as e:
            raise ValueError(f"Error {e}: impossible to load the FEM matrices and FEM RHS")

        return matrices

    def define_geometry_info(self, _n_weak_inlets, _n_weak_outlets, _name_mesh, check_lambda=False):
        """
        MODIFY
        """

        self.M_n_weak_inlets = _n_weak_inlets
        self.M_n_weak_outlets = _n_weak_outlets
        self.M_n_coupling = self.M_n_weak_inlets + self.M_n_weak_outlets
        self.M_mesh_name = _name_mesh

        assert _name_mesh in self.M_valid_meshes, f"Invalid mesh name {self.M_mesh_name}"

        if not check_lambda:
            return

        _path = None
        for f in os.listdir(self.M_snapshots_path):
            if 'param' in f and os.path.isdir(os.path.join(self.M_snapshots_path, f)):
                _path = os.path.join(self.M_snapshots_path, f)
                _files = [_f for _f in os.listdir(_path) if os.path.isfile(os.path.join(_path, _f))]
                lagmult_files = [x for x in _files if 'lagmult' in x]
                lagmult_numbers = []
                for lagmult_n in lagmult_files:
                    try:
                        lagmult_numbers.append(int((lagmult_n.split('.snap')[0])[-1]))
                    except ValueError:
                        lagmult_numbers.append(int((lagmult_n.split('.npy')[0])[-1]))

                if np.max(lagmult_numbers) == self.M_n_coupling + 1:
                    raise ValueError(f"Invalid number of coupling: the value set in the config file "
                                     f"does not match the number of multipliers {np.max(lagmult_numbers)} in the "
                                     f"snapshots directory.")

                break

        if _path is None:
            logger.warning("Impossible to check if the number of Lagrange multipliers is correct, "
                           "since no snapshots are stored.")

        return

    def reset_reduced_structures(self):
        """
        MODIFY
        """

        super().reset_reduced_structures()

        if self.M_n_coupling == 0:
            raise ValueError("Number of coupling is set to zero: run the function define_geometry_info() before calling "
                             "reset_reduced_structures()")

        self.M_Bdiv_matrix = np.zeros(0)
        self.M_BdivT_matrix = np.zeros(0)
        self.M_B_matrix = [np.zeros(0)] * self.M_n_coupling
        self.M_BT_matrix = [np.zeros(0)] * self.M_n_coupling
        self.M_RHS_vector = [np.zeros(0)] * self.M_n_coupling

        if 'clot' in self.M_parametrizations:
            self.M_Mclot_matrices = []

        self.M_R_matrix = np.zeros(0)
        self.M_Radd_matrix = np.zeros(0)
        self.M_has_resistance = False

        return

    def reset_rb_approximation(self):
        """
        MODIFY
        """

        logger.debug("Resetting RB approximation")

        # super().reset_rb_approximation()

        if self.M_n_coupling == 0:
            raise ValueError("Number of coupling is set to zero: run the function define_geometry_info() before calling "
                             "reset_reduced_structures()")

        self.M_Nh = {'velocity': 0, 'pressure': 0, 'lambda': np.zeros(0, dtype=np.int16)}
        self.M_N = {'velocity': 0, 'pressure': 0, 'lambda': np.zeros(0, dtype=np.int16)}
        self.M_N_space = {'velocity': 0, 'pressure': 0, 'lambda': np.zeros(0, dtype=np.int16)}
        self.M_N_time = {'velocity': 0, 'pressure': 0, 'lambda': np.zeros(0, dtype=np.int16)}

        self.M_basis_space = {'velocity': np.zeros(0), 'pressure': np.zeros(0),
                              'lambda': [np.zeros(0)] * self.M_n_coupling}
        self.M_basis_time = {'velocity': np.zeros(0), 'pressure': np.zeros(0),
                             'lambda': [np.zeros(0)] * self.M_n_coupling}

        self.supr_primal = np.zeros(0)
        self.supr_dual = [np.zeros(0)] * self.M_n_coupling

        self.M_affine_decomposition.reset_rb_approximation()
        self.reset_reduced_structures()

        return

    @staticmethod
    def project_matrix(matrix, bases, norm_matrix=None):
        """
        MODIFY
        """

        assert len(bases) == 2

        if norm_matrix is not None:
            normed_basis_space = arr_utils.sparse_matrix_matrix_mul(norm_matrix, bases[0])
        else:
            normed_basis_space = bases[0]

        result = np.dot(normed_basis_space.T,
                        arr_utils.sparse_matrix_matrix_mul(matrix, bases[1]))

        return result

    @staticmethod
    def project_vector(vector, basis, norm_matrix=None):
        """
        MODIFY
        """

        if norm_matrix is not None:
            normed_basis_space = arr_utils.sparse_matrix_matrix_mul(norm_matrix, basis)
        else:
            normed_basis_space = basis

        result = np.dot(normed_basis_space.T, vector)

        return result

    @staticmethod
    def expand_vector(vector, basis_space, basis_time=None):
        """
        MODIFY
        """

        if basis_time is not None:
            return np.dot(basis_space, np.dot(vector, basis_time.T))
        else:
            return np.dot(basis_space, vector)

    def assemble_reduced_structures(self, _space_projection='standard'):
        """
        MODIFY
        """

        structures = {'A', 'M', 'Bdiv', 'B', 'RHS', 'q', 'R'}
        if 'clot' in self.M_parametrizations:
            structures.add('Mclot')

        FEM_matrices = self.import_FEM_structures(structures=structures)

        if _space_projection == 'natural':
            if not self.check_norm_matrices():
                self.get_norm_matrices()
            norms = [self.M_norm_matrices['velocity'], self.M_norm_matrices['pressure']]
        elif _space_projection == 'standard':
            norms = [None, None]
        else:
            raise ValueError(f"Unrecognized space projection {_space_projection}")

        logger.info("Projecting FEM structures onto the reduced subspace in space")

        logger.debug("Projecting the stiffness matrix")
        self.M_A_matrix = self.project_matrix(FEM_matrices['A'],
                                              [self.M_basis_space['velocity'], self.M_basis_space['velocity']],
                                              norm_matrix=norms[0])
        logger.debug("Projecting the mass matrix")
        self.M_M_matrix = self.project_matrix(FEM_matrices['M'],
                                              [self.M_basis_space['velocity'], self.M_basis_space['velocity']],
                                              norm_matrix=norms[0])
        logger.debug("Projecting the divergence transposed matrix")
        self.M_BdivT_matrix = self.project_matrix(FEM_matrices['BdivT'],
                                                  [self.M_basis_space['velocity'], self.M_basis_space['pressure']],
                                                  norm_matrix=norms[0])
        logger.debug("Projecting the divergence matrix")
        self.M_Bdiv_matrix = self.project_matrix(FEM_matrices['Bdiv'],
                                                 [self.M_basis_space['pressure'], self.M_basis_space['velocity']],
                                                 norm_matrix=norms[1])

        for n in range(self.M_n_coupling):
            logger.debug(f"Projecting the weak BC transposed matrix, block {n}")
            self.M_BT_matrix[n] = self.project_matrix(FEM_matrices['BT'][n],
                                                      [self.M_basis_space['velocity'], self.M_basis_space['lambda'][n]],
                                                      norm_matrix=norms[0])
            logger.debug(f"Projecting the weak BC matrix, block {n}")
            self.M_B_matrix[n] = self.project_matrix(FEM_matrices['B'][n],
                                                     [self.M_basis_space['lambda'][n], self.M_basis_space['velocity']],
                                                     norm_matrix=None)
            logger.debug(f"Projecting the weak BC vector, coupling block {n}")
            self.M_RHS_vector[n] = self.project_vector(FEM_matrices['RHS'][n], self.M_basis_space['lambda'][n],
                                                       norm_matrix=None)

        logger.debug("Projecting the initial condition")
        try:
            initial_condition = self.import_FEM_structures(structures={'u0'})
            self.M_u0['velocity'] = self.project_vector(initial_condition['u0'].T, self.M_basis_space['velocity'],
                                                        norm_matrix=self.M_norm_matrices['velocity']).T
        except ValueError:
            logger.warning("Impossible to load the initial condition. Proceeding with homogeneous initial condition")
            self.M_u0['velocity'] = np.zeros((2, self.M_N_space['velocity']))

        if 'clot' in self.M_parametrizations:
            logger.debug("Projecting the clot matrices")
            for (idx_clot, Mclot_matrix) in enumerate(FEM_matrices['Mclot']):
                self.M_Mclot_matrices.append(self.project_matrix(Mclot_matrix,
                                                                [self.M_basis_space['velocity'], self.M_basis_space['velocity']],
                                                                 norm_matrix=norms[0]))

        if 'R' in FEM_matrices:
            logger.debug("Projecting the resistance matrices")
            self.M_R_matrix = self.project_matrix(FEM_matrices['R'],
                                                  [self.M_basis_space['velocity'], self.M_basis_space['velocity']],
                                                  norm_matrix=norms[0])
            self.M_Radd_matrix = self.project_matrix(FEM_matrices['Radd'],
                                                     [self.M_basis_space['velocity'], self.M_basis_space['velocity']],
                                                     norm_matrix=norms[0])
            self.M_has_resistance = True

        logger.debug("Projecting the flow rate vectors")
        for (idx_q, q) in enumerate(FEM_matrices['q_in']):
            self.M_q_vectors['in'].append(self.project_vector(q, self.M_basis_space['velocity'], norm_matrix=norms[0]))
        for (idx_q, q) in enumerate(FEM_matrices['q_out']):
            self.M_q_vectors['out'].append(self.project_vector(q, self.M_basis_space['velocity'], norm_matrix=norms[0]))

        logger.info("Projection of FEM structures onto the reduced subspace in space complete!")

        if self.M_save_offline_structures:
            self.save_reduced_structures()

        return

    def save_reduced_structures(self):
        """
        MODIFY
        """

        logger.debug("Dumping space-reduced structures to file ...")

        gen_utils.create_dir(self.M_reduced_structures_path)

        np.save(os.path.join(self.M_reduced_structures_path, 'A_rb.npy'), self.M_A_matrix)
        np.save(os.path.join(self.M_reduced_structures_path, 'M_rb.npy'), self.M_M_matrix)
        np.save(os.path.join(self.M_reduced_structures_path, 'Bdiv_rb.npy'), self.M_Bdiv_matrix)
        np.save(os.path.join(self.M_reduced_structures_path, 'BdivT_rb.npy'), self.M_BdivT_matrix)
        for n in range(self.M_n_coupling):
            np.save(os.path.join(self.M_reduced_structures_path, f"B{n}_rb.npy"), self.M_B_matrix[n])
            np.save(os.path.join(self.M_reduced_structures_path, f"BT{n}_rb.npy"), self.M_BT_matrix[n])
            np.save(os.path.join(self.M_reduced_structures_path, f"RHS{n}_rb.npy"), self.M_RHS_vector[n])
        np.save(os.path.join(self.M_reduced_structures_path, 'u0_rb.npy'), self.M_u0['velocity'])
        if 'clot' in self.M_parametrizations:
            for k in range(len(self.M_Mclot_matrices)):
                np.save(os.path.join(self.M_reduced_structures_path, f"Mclot{k}_rb.npy"), self.M_Mclot_matrices[k])
        if np.size(self.M_R_matrix) > 0:
            np.save(os.path.join(self.M_reduced_structures_path, 'R_rb.npy'), self.M_R_matrix)
            np.save(os.path.join(self.M_reduced_structures_path, 'Radd_rb.npy'), self.M_Radd_matrix)
        for k_in in range(self.M_n_inlets):
            np.save(os.path.join(self.M_reduced_structures_path, f"q_in{k_in}_rb.npy"), self.M_q_vectors['in'][k_in])
        for k_out in range(self.M_n_outlets):
            np.save(os.path.join(self.M_reduced_structures_path, f"q_out{k_out}_rb.npy"), self.M_q_vectors['out'][k_out])

        return

    def import_reduced_structures(self):
        """
        MODIFY
        """
        logger.info("Importing FEM structures multiplied by RB in space")

        try:
            self.M_A_matrix = np.load(os.path.join(self.M_reduced_structures_path, "A_rb.npy"))
            self.M_M_matrix = np.load(os.path.join(self.M_reduced_structures_path, "M_rb.npy"))
            self.M_Bdiv_matrix = np.load(os.path.join(self.M_reduced_structures_path, "Bdiv_rb.npy"))
            self.M_BdivT_matrix = np.load(os.path.join(self.M_reduced_structures_path, "BdivT_rb.npy"))

            self.M_B_matrix = [np.zeros(0)] * self.M_n_coupling
            self.M_BT_matrix = [np.zeros(0)] * self.M_n_coupling
            self.M_RHS_vector = [np.zeros(0)] * self.M_n_coupling
            for n in range(self.M_n_coupling):
                self.M_B_matrix[n] = np.load(os.path.join(self.M_reduced_structures_path, f"B{n}_rb.npy"))
                self.M_BT_matrix[n] = np.load(os.path.join(self.M_reduced_structures_path, f"BT{n}_rb.npy"))
                self.M_RHS_vector[n] = np.load(os.path.join(self.M_reduced_structures_path, f"RHS{n}_rb.npy"))

            self.M_u0['velocity'] = np.load(os.path.join(self.M_reduced_structures_path, "u0_rb.npy"))

            if 'clot' in self.M_parametrizations:
                self.M_Mclot_matrices = []
                path = lambda cnt: os.path.join(self.M_reduced_structures_path, f"Mclot{cnt}_rb.npy")
                k = 0
                assert os.path.isfile(path(k)), "No reduced clot matrices available"
                while os.path.isfile(path(k)):
                    self.M_Mclot_matrices.append(np.load(path(k)))
                    k += 1

            if os.path.isfile(os.path.join(self.M_reduced_structures_path, f"R_rb.npy")):
                self.M_R_matrix = np.load(os.path.join(self.M_reduced_structures_path, "R_rb.npy"))
                self.M_Radd_matrix = np.load(os.path.join(self.M_reduced_structures_path, "Radd_rb.npy"))
                self.M_has_resistance = True

            path_in = lambda cnt: os.path.join(self.M_reduced_structures_path, f"q_in{cnt}_rb.npy")
            k_in = 0
            assert os.path.isfile(path_in(k_in)), "No reduced inflow vectors available"
            while os.path.isfile(path_in(k_in)):
                self.M_q_vectors['in'].append(np.load(path_in(k_in)))
                k_in += 1
            self.M_n_inlets = k_in

            path_out = lambda cnt: os.path.join(self.M_reduced_structures_path, f"q_out{cnt}_rb.npy")
            k_out = 0
            assert os.path.isfile(path_out(k_out)), "No reduced outflow vectors available"
            while os.path.isfile(path_out(k_out)):
                self.M_q_vectors['out'].append(np.load(path_out(k_out)))
                k_out += 1
            self.M_n_outlets = k_out

            import_success = True

        except (OSError, FileNotFoundError, AssertionError) as e:
            logger.error(f"Error {e}: failed to import the reduced structures!")
            import_success = False

        if import_success:
            self.M_N_space['velocity'] = self.M_A_matrix.shape[0]
            self.M_N_space['pressure'] = self.M_Bdiv_matrix.shape[0]
            self.M_N_space['lambda'] = np.array([self.M_B_matrix[n].shape[0] for n in range(self.M_n_coupling)])
            self.M_N['lambda'] = self.M_N_space['lambda'] * self.M_N_time['lambda']
            self.M_N_lambda_cumulative = np.cumsum(np.vstack([self.M_N['lambda'][n] for n in range(self.M_n_coupling)]))
            self.M_N_lambda_cumulative = np.insert(self.M_N_lambda_cumulative, 0, 0)

        return import_success

    def get_clots_number(self):
        """
        MODIFY
        """

        k = 0
        if 'clot' in self.M_parametrizations:
            path_m = lambda cnt: os.path.join(self.M_fom_structures_path, f"Mclot{cnt}.m")
            path_npz = lambda cnt: os.path.join(self.M_fom_structures_path, f"Mclot{cnt}.npz")
            while os.path.isfile(path_m(k)) or os.path.isfile(path_npz(k)):
                k += 1

        return k

    def index_mapping(self, i, j, field="velocity", n_coupl=0):
        """
        MODIFY
        """

        if field == 'lambda':
            assert 0 <= n_coupl < self.M_n_coupling
            return int(i * self.M_N_time['lambda'][n_coupl] + j)
        else:
            return int(i * self.M_N_time[field] + j)

    def compute_generalized_coordinates(self, x_fom, field="all"):
        """
        MODIFY
        """

        if not self.check_norm_matrices():
            self.get_norm_matrices()

        u_fom, p_fom, l_fom = None, None, None

        if field == "all":
            u_fom = x_fom[:self.M_Nh['velocity'], :self.M_Nt]
            p_fom = x_fom[self.M_Nh['velocity']:self.M_Nh['velocity']+self.M_Nh['pressure'], :self.M_Nt]
            NL_cum = np.hstack([np.array([0]), np.cumsum(self.M_Nh['lambda'])])
            l_fom = [x_fom[self.M_Nh['velocity']+self.M_Nh['pressure']+NL_cum[n]:
                           self.M_Nh['velocity']+self.M_Nh['pressure']+NL_cum[n+1], :self.M_Nt]
                     for n in range(self.M_n_coupling)]
        elif field == "velocity":
            u_fom = x_fom[:, :self.M_Nt]
        elif field == "pressure":
            p_fom = x_fom[:, :self.M_Nt]
        elif field == "lambda":
            l_fom = x_fom[:, :self.M_Nt]
        else:
            logger.warning(f"Unrecognized field {field}!")
            return None

        def _project(sol_fom, basis_space, basis_time=None):
            if basis_time is not None:
                return np.dot(basis_space.T, np.dot(sol_fom, basis_time)).flatten()
            else:
                return np.dot(basis_space.T, sol_fom)

        if field in {"all", "velocity"}:
            normed_basis_space = arr_utils.sparse_matrix_matrix_mul(self.M_norm_matrices['velocity'],
                                                                    self.M_basis_space['velocity'])
            u_rb = _project(u_fom, normed_basis_space,
                            self.M_basis_time['velocity'] if hasattr(self, 'M_basis_time') else None)

        if field in {"all", "pressure"}:
            normed_basis_space = arr_utils.sparse_matrix_matrix_mul(self.M_norm_matrices['pressure'],
                                                                    self.M_basis_space['pressure'])
            p_rb = _project(p_fom, normed_basis_space,
                            self.M_basis_time['pressure'] if hasattr(self, 'M_basis_time') else None)

        if field in {"all", "lambda"}:
            l_rb = [np.zeros(0)] * self.M_n_coupling
            for n in range(self.M_n_coupling):
                l_rb[n] = _project(l_fom[n], self.M_basis_space['lambda'][n],
                                   self.M_basis_time['lambda'][n] if hasattr(self, 'M_basis_time') else None)
            l_rb = np.hstack([l_rb[n] for n in range(self.M_n_coupling)])

        x_rb = (np.hstack([u_rb, p_rb, l_rb]) if field == "all" else
                u_rb if field == "velocity" else
                p_rb if field == "pressure" else
                l_rb)

        return x_rb

    def get_generalized_coordinates(self, n_snapshots=None, ss_ratio=1, save_flag=False):
        """
        MODIFY
        """

        assert self.M_import_snapshots, \
            f"Impossible to compute the generalized coordinates if the FOM snapshots are not loaded."

        if arr_utils.is_empty(self.M_snapshots_matrix['velocity']):
            self._import_snapshots(_ns=n_snapshots, is_test=False, ss_ratio=ss_ratio)

        if n_snapshots is None or n_snapshots > self.M_ns:
            logger.warning(f"Number of considered snapshots clamped to {self.M_ns}, being it either higher "
                           f"than the total number of snapshots or left to None")
            n_snapshots = self.M_ns
        else:
            logger.info(f"Number of considered snapshots: {n_snapshots}")

        self.M_snapshots_hat['velocity'] = np.zeros((self.M_N['velocity'], n_snapshots))
        self.M_snapshots_hat['pressure'] = np.zeros((self.M_N['pressure'], n_snapshots))
        self.M_snapshots_hat['lambda'] = [np.zeros(0)] * self.M_n_coupling
        for n in range(self.M_n_coupling):
            self.M_snapshots_hat['lambda'][n] = np.zeros((self.M_N['lambda'][n], n_snapshots))

        for iP in range(n_snapshots):

            logger.debug(f"Computing generalized coordinates of snapshot {iP}")

            _Nt = self.M_Nt * self.M_N_periods

            u_fom = self.M_snapshots_matrix['velocity'][:, iP * _Nt:iP * _Nt + self.M_Nt]
            p_fom = self.M_snapshots_matrix['pressure'][:, iP * _Nt:iP * _Nt + self.M_Nt]
            l_fom = np.vstack([self.M_snapshots_matrix['lambda'][n][:, iP * _Nt:iP * _Nt + self.M_Nt]
                               for n in range(self.M_n_coupling)])
            x_fom = np.vstack([u_fom, p_fom, l_fom])

            x_rb = RbManagerSpaceTimeStokes.compute_generalized_coordinates(self, x_fom)

            self.M_snapshots_hat['velocity'][:, iP] = x_rb[:self.M_N['velocity']]
            self.M_snapshots_hat['pressure'][:, iP] = x_rb[self.M_N['velocity']:self.M_N['velocity'] + self.M_N['pressure']]
            for n in range(self.M_n_coupling):
                self.M_snapshots_hat['lambda'][n][:, iP] = x_rb[self.M_N['velocity']+self.M_N['pressure']+self.M_N_lambda_cumulative[n]:
                                                                self.M_N['velocity']+self.M_N['pressure']+self.M_N_lambda_cumulative[n+1]]

        if self.M_save_offline_structures and save_flag:
            gen_utils.create_dir(self.M_generalized_coords_path)
            np.save(os.path.join(self.M_generalized_coords_path, 'parameters.npy'), self.M_offline_ns_parameters)
            np.save(os.path.join(self.M_generalized_coords_path, 'velocity.npy'), self.M_snapshots_hat['velocity'])
            np.save(os.path.join(self.M_generalized_coords_path, 'pressure.npy'), self.M_snapshots_hat['pressure'])
            for n in range(self.M_n_coupling):
                np.save(os.path.join(self.M_generalized_coords_path, f'lambda{n}.npy'), self.M_snapshots_hat['lambda'][n])

        return

    def import_generalized_coordinates(self):
        """
        MODIFY
        """

        logger.info("Importing the generalized coordinates")

        if ({'velocity', 'pressure', 'lambda'}.issubset(self.M_snapshots_hat) and
                not arr_utils.is_empty(self.M_snapshots_hat['velocity'])):
            return True

        try:
            self.M_snapshots_hat = dict()
            self.M_snapshots_hat['velocity'] = np.load(os.path.join(self.M_generalized_coords_path, 'velocity.npy'))
            self.M_snapshots_hat['pressure'] = np.load(os.path.join(self.M_generalized_coords_path, 'pressure.npy'))
            self.M_snapshots_hat['lambda'] = [np.zeros(0)] * self.M_n_coupling
            for n in range(self.M_n_coupling):
                self.M_snapshots_hat['lambda'][n] = np.load(os.path.join(self.M_generalized_coords_path, f'lambda{n}.npy'))

            if (arr_utils.is_empty(self.M_offline_ns_parameters)) or \
               (self.M_offline_ns_parameters.shape[0] < self.M_snapshots_hat['velocity'].shape[0]):
                self.M_offline_ns_parameters = np.load(os.path.join(self.M_generalized_coords_path, 'parameters.npy'))

            import_success = True

        except (FileNotFoundError, OSError):
            logger.warning("Failed to import the generalized coordinates")
            import_success = False

        return import_success

    def get_zero_vector(self):
        return np.zeros(self.M_N['velocity'] + self.M_N['pressure'] + self.M_N_lambda_cumulative[-1])

    def get_field(self, _wn, field, n=None, reshape=False):
        """
        Get target field from solution
        """

        if field == 'velocity':
            vec = _wn[:self.M_N['velocity']]
            if reshape:
                vec = np.reshape(vec, (self.M_N_space['velocity'], self.M_N_time['velocity']))
        elif field == 'pressure':
            vec = _wn[self.M_N['velocity']:self.M_N['velocity'] + self.M_N['pressure']]
            if reshape:
                vec = np.reshape(vec, (self.M_N_space['pressure'], self.M_N_time['pressure']))
        elif field == 'lambda':
            if n is None:
                vec = _wn[self.M_N['velocity'] + self.M_N['pressure']:]
            else:
                assert 0 <= n <= self.M_n_coupling, f"Invalid coupling index {n}"
                vec = _wn[self.M_N['velocity'] + self.M_N['pressure'] + self.M_N_lambda_cumulative[n]:
                          self.M_N['velocity'] + self.M_N['pressure'] + self.M_N_lambda_cumulative[n+1]]
                if reshape:
                    vec = np.reshape(vec, (self.M_N_space['lambda'][n], self.M_N_time['lambda'][n]))
        else:
            raise ValueError(f"Unrecognized field {field}")

        return vec

    def reconstruct_fem_solution(self, _w, fields=None, indices_space=None, indices_time=None):
        """
        MODIFY
        """

        logger.debug("Re-projecting the reduced solution onto the FE space")

        if fields is None:
            fields = ["velocity", "pressure", "lambda"]

        for field in fields:
            times = (slice(None) if (indices_time is None or field not in indices_time.keys())
                     else indices_time[field])

            if field == 'lambda':
                lambdatildeh = [np.zeros(0)] * self.M_n_coupling
                for n in range(self.M_n_coupling):
                    _ln = self.get_field(_w, field, n=n)
                    _ln = np.reshape(_ln, (self.M_N_space['lambda'][n], self.M_N_time['lambda'][n]))
                    lambdatildeh[n] = self.expand_vector(_ln, self.M_basis_space['lambda'][n],
                                                         self.M_basis_time['lambda'][n][times])

                    if self._has_IC(field='lambda', n=n):
                        _l0 = np.dot(self.M_basis_space['lambda'][n], self.M_u0['lambda'][n][-1])  # (N_l^s,)
                        lambdatildeh[n] += _l0[..., None] * np.ones(self.M_Nt)[None]

                self.M_utildeh['lambda'] = lambdatildeh

            else:
                spaces = (np.arange(self.M_Nh[field]) if (indices_space is None or field not in indices_space.keys())
                          else indices_space[field])

                _un = self.get_field(_w, field)
                _un = np.reshape(_un, (self.M_N_space[field], self.M_N_time[field]))
                self.M_utildeh[field] = self.expand_vector(_un, self.M_basis_space[field][spaces],
                                                           self.M_basis_time[field][times])

                if self._has_IC(field):
                    _u0 = np.dot(self.M_basis_space[field], self.M_u0[field][-1])  # (N_u^s,)
                    self.utildeh[field] += _u0[..., None] * np.ones(self.M_Nt)[None]

        return

    def set_empty_blocks(self):
        """
        MODIFY
        """

        self.M_Blocks[4] = np.zeros((self.M_N['pressure'], self.M_N['pressure']))
        self.M_Blocks[5] = np.zeros((self.M_N['pressure'], self.M_N_lambda_cumulative[-1]))
        self.M_Blocks[7] = np.zeros((self.M_N_lambda_cumulative[-1], self.M_N['pressure']))
        self.M_Blocks[8] = np.zeros((self.M_N_lambda_cumulative[-1], self.M_N_lambda_cumulative[-1]))

        self.M_f_Blocks_no_param[0] = np.zeros(self.M_N['velocity'])
        self.M_f_Blocks_no_param[1] = np.zeros(self.M_N['pressure'])
        self.M_f_Blocks_no_param[2] = np.zeros(self.M_N_lambda_cumulative[-1])

        return

    def build_rb_approximation(self, _ns, _n_weak_io, _mesh_name, _tolerances,
                               _space_projection='standard', prob=None, ss_ratio=1):
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
            import_success_reduced_structures = self.import_reduced_structures()
            if not import_success_reduced_structures:
                logger.info('Building space-reduced structures whose import has failed')
                start = time.time()
                self.assemble_reduced_structures(_space_projection=_space_projection)
                logger.debug(f"Assembling of space-reduced structures performed in {(time.time()-start):.4f} s")

            if 'ST' in self.M_reduction_method:  # avoid construction of ST blocks in SRB-TFO
                import_failures_ac = self.import_rb_affine_components()
                if import_failures_ac:
                    start = time.time()
                    logger.info('Building affine space-time reduced structures whose import has failed')
                    self.build_rb_affine_decompositions()
                    logger.debug(f"Assembling of space-time-reduced structures performed in {(time.time()-start):.4f} s")
        else:
            logger.info('Building space-reduced structures')
            start = time.time()
            self.assemble_reduced_structures(_space_projection=_space_projection)
            logger.debug(f"Assembling of space-reduced structures performed in {(time.time()-start):.4f} s")

            if 'ST' in self.M_reduction_method:
                logger.info('Building all reduced structures')
                start = time.time()
                self.build_rb_affine_decompositions()
                logger.debug(f"Assembling of space-time-reduced structures performed in {(time.time()-start):.4f} s")

        return

    def build_rb_affine_decompositions(self, operators=None):
        """
        MODIFY
        """

        if operators is None:
            operators = {'Mat', 'f'}

        if 'Mat' in operators:
            self.build_rb_nonparametric_LHS()

        if 'f' in operators:
            self.build_rb_nonparametric_RHS()

        self.set_param_functions()

        if self.M_save_offline_structures:
            self.save_rb_affine_decomposition(operators=operators)

        return

    def save_rb_affine_decomposition(self, operators=None, blocks=None, f_blocks=None):
        """
        Saving RB blocks for the LHS matrix
        """

        if operators is None:
            operators = {'Mat', 'f'}

        gen_utils.create_dir(self.M_affine_components_path)

        if 'Mat' in operators:
            logger.debug("Dumping RB LHS matrix parameter-independent blocks to file")

            blocks = [0, 1, 2, 3, 6] if blocks is None else blocks
            for iB in blocks:
                cur_path = os.path.join(self.M_affine_components_path, f"Block{iB}.npy")
                np.save(cur_path, self.M_Blocks[iB])

            for key in self.M_Blocks_param_affine:
                for iB in self.M_Blocks_param_affine[key]:
                    for (iC, elem) in enumerate(self.M_Blocks_param_affine[key][iB]):
                        cur_path = os.path.join(self.M_affine_components_path, f"{key}_Block{iB}_{iC}.npy")
                        np.save(cur_path, elem)

        if 'f' in operators:
            logger.debug("Dumping RB RHS vector parameter-independent blocks to file")

            f_blocks = [0] if f_blocks is None else f_blocks
            if self._has_IC():
                for iB in f_blocks:
                    cur_path = os.path.join(self.M_affine_components_path, f"F_Block{iB}.npy")
                    np.save(cur_path, self.M_f_Blocks_no_param[iB])

            for key in self.M_f_Blocks_param_affine:
                for iB in self.M_f_Blocks_param_affine[key]:
                    for (iC, elem) in enumerate(self.M_f_Blocks_param_affine[key][iB]):
                        cur_path = os.path.join(self.M_affine_components_path, f"{key}_F_Block{iB}_{iC}.npy")
                        np.save(cur_path, elem)

        return

    def import_rb_affine_components(self, operators=None, blocks=None, f_blocks=None):
        """
        Importing RB blocks for the LHS matrix
        """

        if operators is None:
            operators = {'Mat', 'f'}

        import_failures_ac = set()

        if 'Mat' in operators:
            logger.info("Importing RB LHS matrix parameter-independent blocks from file")

            self.M_Blocks = [np.zeros(0)] * 9
            blocks = [0, 1, 2, 3, 6] if blocks is None else blocks
            for iB in blocks:
                try:
                    cur_path = os.path.join(self.M_affine_components_path, f"Block{iB}.npy")
                    self.M_Blocks[iB] = np.load(cur_path)
                except (IOError, OSError, FileNotFoundError) as e:
                    logger.error(f"Error {e}. Impossible to open the desired file for block Block{iB}.")
                    import_failures_ac.add('Mat')
                    break

            path = lambda key, iB, iC: os.path.join(self.M_affine_components_path, f"{key}_Block{iB}_{iC}.npy")
            for key in self.M_parametrizations:
                self.M_Blocks_param_affine[key] = dict()
                for iB in range(9):
                    iC = 0
                    while os.path.isfile(path(key, iB, iC)):
                        if iC == 0:
                            self.M_Blocks_param_affine[key][iB] = [np.load(path(key, iB, iC))]
                        else:
                            self.M_Blocks_param_affine[key][iB].append(np.load(path(key, iB, iC)))
                        iC += 1

        if 'f' in operators:
            logger.debug("Dumping RB RHS vector parameter-independent blocks to file")

            f_blocks = [0] if f_blocks is None else f_blocks
            if self._has_IC():
                for iB in f_blocks:
                    try:
                        cur_path = os.path.join(self.M_affine_components_path, f"F_Block{iB}.npy")
                        self.M_f_Blocks_no_param[iB] = np.load(cur_path)
                    except (IOError, OSError, FileNotFoundError) as e:
                        logger.error(f"Error {e}. Impossible to open the desired file for block F_Block{iB}.")
                        import_failures_ac.add('f')
                        break

                path = lambda key, iB, iC: os.path.join(self.M_affine_components_path, f"{key}_F_Block{iB}_{iC}.npy")
                for key in self.M_parametrizations:
                    self.M_f_Blocks_param_affine[key] = dict()
                    for iB in range(3):
                        iC = 0
                        while os.path.isfile(path(key, iB, iC)):
                            if iC == 0:
                                self.M_f_Blocks_param_affine[key][iB] = [np.load(path(key, iB, iC))]
                            else:
                                self.M_f_Blocks_param_affine[key][iB].append(np.load(path(key, iB, iC)))
                            iC += 1

        self.set_empty_blocks()

        if not import_failures_ac:
            self.set_param_functions()

        return import_failures_ac

    @property
    def parametrizations(self):
        return self.M_parametrizations

    def set_parametrizations(self, parametrizations):
        """
        MODIFY
        """

        self.M_parametrizations = parametrizations

        if 'inflow' not in parametrizations:
            raise ValueError("Inflow parametrization must be included!")

        for param in self.M_parametrizations:
            if param not in self.M_valid_parametrizations:
                logger.warning(f"{param} parametrization is not valid and hence it will not be considered!")

        # TODO: handle fluid physics parametrization
        if 'fluid' in self.M_parametrizations:
            raise NotImplementedError("Fluid physics parametrization is not yet implemented!")

        return

    def get_param_indices(self, consider_extra=True):
        """
        Get indices of parameter vector associated to the different parametrizations
        """

        assert self.M_parametrizations

        params_idxs_map = dict()
        n_params_cum = 0

        for parametrization in self.M_valid_parametrizations:
            if parametrization in self.M_parametrizations:
                if parametrization == 'inflow':
                    n_params = self.M_n_inflow_params + \
                               (self.M_n_weak_inlets if consider_extra else (self.M_n_weak_inlets - 1))
                elif parametrization == 'outflow':
                    n_params = self.M_n_outflow_params + \
                               (self.M_n_weak_outlets if consider_extra else (self.M_n_weak_outlets - 1))
                elif parametrization == 'fluid':
                    n_params = 2
                elif parametrization == 'clot':
                    n_params = None
                else:
                    raise ValueError("Invalid parametrization!")

                params_idxs_map[parametrization] = [n_params_cum,
                                                    n_params_cum + n_params if n_params is not None else None]
                if n_params is not None:  # safe since 'clot' is (and it must!) listed as the last parameterization
                    n_params_cum += n_params

        return params_idxs_map

    def get_flow_rates(self, param):
        param_map = self.differentiate_parameters(param)

        idx1, idx2 = self.M_n_inflow_params, self.M_n_inflow_params + self.M_n_weak_inlets
        flow_rates = (self.M_inflow_rate_function(param_map['inflow'][:idx1])[:, None] *
                      param_map['inflow'][None, idx1:idx2])

        if 'outflow' in self.M_parametrizations:
            _idx1, _idx2 = self.M_n_outflow_params, self.M_n_outflow_params + self.M_n_weak_outlets
            outflow_params = param_map['outflow'][:_idx1] if self.M_n_outflow_params > 0 else \
                             param_map['inflow'][:idx1]
            outflow_rates = (self.M_outflow_rate_function(outflow_params)[:, None] *
                             param_map['outflow'][None, _idx1:_idx2])
            flow_rates = np.hstack([flow_rates, outflow_rates])

        return flow_rates

    def differentiate_parameters(self, _param, consider_extra=True, perform_rescaling=True):
        """
        MODIFY
        """

        params_idxs_map = self.get_param_indices(consider_extra=consider_extra)

        params_map = dict()
        for parametrization in self.M_parametrizations:
            idxs = params_idxs_map[parametrization]
            params_map[parametrization] = _param[idxs[0]:idxs[1]]

        return params_map

    def concatenate_parameters(self, _param_map):
        """
        MODIFY
        """

        assert self.M_parametrizations

        param = np.hstack([_param_map[_param_name] for _param_name in self.M_valid_parametrizations
                           if _param_name in self.M_parametrizations])

        return param

    def set_param_functions(self):
        """
        MODIFY
        """

        n_clots = self.get_clots_number()

        if 'clot' in self.M_parametrizations:
            self.M_Blocks_param_affine_fun['clot'] = dict()
            if 'clot' in self.M_Blocks_param_affine.keys():
                for iB in self.M_Blocks_param_affine['clot'].keys():
                    self.M_Blocks_param_affine_fun['clot'][iB] = [lambda mu, _k=k: mu[_k] for k in range(n_clots)]

            self.M_f_Blocks_param_affine_fun['clot'] = dict()
            if 'clot' in self.M_f_Blocks_param_affine.keys():
                for ifB in self.M_f_Blocks_param_affine['clot'].keys():
                    self.M_f_Blocks_param_affine_fun['clot'][ifB] = [lambda mu, _k=k: mu[_k] for k in range(n_clots)]

        return

    def build_reduced_problem(self, _param=None):
        """
        MODIFY
        """

        if _param is None:
            logger.warning("A parameter value must be given in input to assemble the rhs vector")
            if self.M_N_params > 0:
                logger.debug("Proceeding with a null parameter vector")
                _param = np.zeros(self.M_N_params)
            else:
                logger.debug("Impossible to build the parametrized reduced problem")

        self.build_rb_LHS(_param)
        self.build_rb_RHS(_param)

        return

    def _solve_lu(self, tol=1e-5, tol_abs=1e-8, max_err=1e-4, max_err_abs=1e-7, max_iter=50):
        """
        MODIFY
        """

        my_newton = Newton(tol=tol, tol_abs=tol_abs, max_err=max_err, max_err_abs=max_err_abs,
                           max_iter=max_iter, jac_iter=1, alpha=1.0)

        def residual(u):
            res = self.M_Block.dot(u) - self.M_f_Block
            return res

        def jacobian(u):
            return self.M_Block_LU

        u0 = self.M_un if self.M_un.size > 0 else np.zeros_like(self.M_f_Block)

        # call Newton method and compute solution
        self.M_un, self.M_solver_converged = my_newton(residual, jacobian, u0,
                                                       use_lu=self.M_use_LU)

        status = 0 if self.M_solver_converged else 1

        return status

    def _solve(self, _param=None):
        """
        MODIFY
        """

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                status = 1
                if self.M_use_LU:
                    status = self._solve_lu()
                if status == 1:
                    self.M_un = scipy.linalg.solve(self.M_Block, self.M_f_Block)
                    status = 0
            except scipy.linalg.LinAlgWarning:
                logger.warning(f"The linear system is ill-conditioned and the solution is inaccurate! -- "
                               f"Condition number: {np.linalg.cond(self.M_Block):.2e}")
                status = 1

        return status

    def solve_reduced_problem(self, _param):
        """
        MODIFY
        """

        logger.debug(f"Building RB problem for parameter: {_param}")
        if self.M_use_LU and len(self.M_Block_LU) == 0:
            self.build_mu_independent_LHS()
        self.build_reduced_problem(_param)

        logger.debug(f"Solving RB problem for parameter: {_param}")
        status = self._solve(_param=_param)

        return status

    def solve_reduced_problem_iteratively(self, _param, n_cycles=1, update_T=False, update_params=False):
        """
        MODIFY
        """

        self.M_save_offline_structures = False  # to prevent from saving structures with "wrong" value of Delta_t

        solutions = [np.zeros(0)] * n_cycles
        solutions_fem = [dict() for _ in range(n_cycles)]
        dts = [0] * n_cycles
        _, Nt = self.M_fom_problem.time_specifics

        if self.M_use_LU:
            self.build_mu_independent_LHS()

        param_trace = self.get_parameters_trace(_param, n_cycles=n_cycles,
                                                update_T=update_T, update_params=update_params)

        logger.debug(f"Solving RB problem for initial parameter {_param} and {n_cycles} cycles")

        fom_rec_time = 0
        for n in range(n_cycles):
            if update_T:
                self.M_fom_problem.M_fom_specifics['final_time'] = param_trace['T'][n]
                self.build_rb_affine_decompositions()  # rebuild reduced system to account for new dt (inefficient...)

            if update_params:
                _param = param_trace['param'][n]

            if update_T or update_params or n == 0:
                self.set_param_functions()
                self.build_reduced_problem(_param)  # rebuild parameter-dependent terms
            else:
                self.build_rb_RHS()  # rebuild only the RHS vector, potentially changed after IC update

            logger.debug(f"Cycle {n} - Value of T: {self.dt*Nt:.2e}")
            status = self._solve(_param=_param)

            if status:
                logger.critical("System solver failed!")
                return None, None, None

            solutions[n] = np.copy(self.M_un)
            dts[n] = self.dt

            start1 = time.perf_counter()

            self.reconstruct_fem_solution(self.M_un)

            for field in self.M_utildeh:
                solutions_fem[n][field] = self.M_utildeh[field]

            fom_rec_time += time.perf_counter() - start1

            if n < n_cycles - 1:
                self._update_IC()  # update the initial condition

        return solutions, solutions_fem, dts, fom_rec_time

    @staticmethod
    def _sample_from_gpr(kernel, M, m, n_samples=1, n_cycles=1, seed=0, K=10):
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=seed, copy_X_train=False)
        x = np.linspace(0, n_cycles - 1, K*n_cycles)

        x_rep = np.zeros((100,1))
        y_rep = np.zeros((100,1))
        gpr.fit(x_rep, y_rep)

        X = x.reshape(-1, 1)
        y = gpr.sample_y(X, n_samples, random_state=seed)
        y_samples = 0.5 * (m + M + (M - m) * np.tanh(3 / 5 * y))
        y_samples = y_samples[::K]

        return y_samples

    def __update_flow_parameters(self, _param, _n_flow_params, _n_boundaries, kernel, n_cycles=1, m=-0.25, M=0.25):
        """
        MODIFY
        """

        new_params, n_params = [], 0

        # here I handle the parameters related to the flow waveform
        if _n_flow_params:
            new_params.append(self._sample_from_gpr(kernel, m, M, n_samples=_n_flow_params, n_cycles=n_cycles,
                                                    seed=0, K=2))
            n_params += _n_flow_params

        # here I handle the parameters related to flow splitting
        if _n_boundaries == 1:
            new_params.append(np.zeros((n_cycles, _n_boundaries)))
        else:
            tmp_params = self._sample_from_gpr(kernel, m, M, n_samples=_n_boundaries, n_cycles=n_cycles,
                                               seed=1, K=2)
            cur_params = _param[n_params:n_params + _n_boundaries]
            tmp_params = (cur_params * tmp_params - np.mean(cur_params * tmp_params, axis=1)[:, None]) / cur_params

            if 'bif' in self.M_mesh_name:
                tmp_params = np.insert(tmp_params, 0, 0.0, axis=1)

            new_params.append(tmp_params)
            n_params += _n_boundaries

        return new_params, n_params

    def _sample_parameters_over_time(self, _param, n_cycles=1, m=-0.25, M=0.25):
        """
        MODIFY
        """

        new_params, cur_n_params = [], 0

        kernel = RBF(length_scale=1.0, length_scale_bounds='fixed')

        if 'inflow' in self.M_parametrizations:
            new_params, cur_n_params = self.__update_flow_parameters(_param,
                                                                     self.M_n_inflow_params, self.M_n_weak_inlets,
                                                                     kernel, n_cycles=n_cycles, m=m, M=M)

        if 'outflow' in self.M_parametrizations:
            tmp_params, tmp_n_params = self.__update_flow_parameters(_param,
                                                                     self.M_n_outflow_params, self.M_n_weak_outlets,
                                                                     kernel, n_cycles=n_cycles, m=m, M=M)

            if len(new_params):
                [new_params.append(_params) for _params in tmp_params]
                cur_n_params += tmp_n_params
            else:
                new_params, cur_n_params = tmp_params, tmp_n_params

        if 'fluid' in self.M_parametrizations:
            raise NotImplementedError

        if 'clot' in self.M_parametrizations:
            n_active_params = _param.shape[0] - cur_n_params
            new_densities = (10**(self._sample_from_gpr(kernel, m, M, n_samples=n_active_params, n_cycles=n_cycles,
                                                        seed=2, K=2)) *
                             (_param[-n_active_params:] > 0.0))

            if len(new_params):
                [new_params.append(_params) for _params in new_densities]
            else:
                new_params = new_densities

        new_params = np.hstack([param for param in new_params])

        return new_params

    def get_parameters_trace(self, _param=None, n_cycles=1, update_T=True, update_params=True):
        """
        MODIFY
        """

        ret_dict = {'T': np.array([self.M_fom_problem.M_fom_specifics['final_time']]),
                    'param': _param[None]}

        if n_cycles == 1:
            return ret_dict

        if update_T:
            T = self.M_fom_problem.M_fom_specifics['final_time']
            kernel = RBF(length_scale=1.0, length_scale_bounds='fixed')
            Ts = (np.repeat(np.array([T])[np.newaxis, :], n_cycles, axis=0) *
                  (1.0 + self._sample_from_gpr(kernel, -0.25, 0.25, n_samples=1, n_cycles=n_cycles,
                                               seed=0, K=2)))[:, 0]
            ret_dict['T'] = Ts

        if update_params:
            assert _param is not None
            params = (np.repeat(_param[np.newaxis, :], n_cycles, axis=0) *
                      (1.0 + self._sample_parameters_over_time(_param, n_cycles=n_cycles)))
            ret_dict['param'] = params

        return ret_dict

    def get_solution_field(self, field, n=0):
        """
        MODIFY
        """
        if field == "velocity":
            sol = np.reshape(self.M_un[:self.M_N['velocity']],
                             (self.M_N_space['velocity'], self.M_N_time['velocity']))
        elif field == "pressure":
            sol = np.reshape(self.M_un[self.M_N['velocity']:self.M_N['velocity']+self.M_N['pressure']],
                             (self.M_N_space['pressure'], self.M_N_time['pressure']))
        elif field == "lambda":
            assert n < self.M_n_coupling, f"Invalid coupling index {n}"
            sol = np.reshape(self.M_un[self.M_N['velocity']+self.M_N['pressure']+self.M_N_lambda_cumulative[n]:
                                       self.M_N['velocity']+self.M_N['pressure']+self.M_N_lambda_cumulative[n+1]],
                             (self.M_N_space['lambda'][n], self.M_N_time['lambda'][n]))
        else:
            raise ValueError(f"Unrecognized field {field}")

        return sol

    @property
    def n_coupling(self):
        return self.M_n_coupling

    @property
    def n_weak_inlets(self):
        return self.M_n_weak_inlets

    @property
    def n_weak_outlets(self):
        return self.M_n_weak_outlets

    @property
    def inflow_rate_function(self):
        return self.M_inflow_rate_function

    @inflow_rate_function.setter
    def inflow_rate_function(self, _inflow_rate_function):
        self.M_inflow_rate_function = _inflow_rate_function
        return

    @property
    def outflow_rate_function(self):
        return self.M_outflow_rate_function

    @outflow_rate_function.setter
    def outflow_rate_function(self, _outflow_rate_function):
        self.M_outflow_rate_function = _outflow_rate_function
        return

    @property
    def n_inflow_params(self):
        return self.M_n_inflow_params

    @n_inflow_params.setter
    def n_inflow_params(self, _n_inflow_params):
        self.M_n_inflow_params = _n_inflow_params
        return

    @property
    def n_outflow_params(self):
        return self.M_n_outflow_params

    @n_outflow_params.setter
    def n_outflow_params(self, _n_outflow_params):
        self.M_n_outflow_params = _n_outflow_params
        return

    def weak_outflow(self):
        return self.M_outflow_rate_function is not None

    def set_flow_rate(self, inflow_rate=(None,0), outflow_rate=(None,0)):
        """
        Set inflow/outflow rate function and number of inflow rate parameters
        """
        self.inflow_rate_function = inflow_rate[0]
        self.n_inflow_params = inflow_rate[1]
        self.outflow_rate_function = outflow_rate[0]
        self.n_outflow_params = outflow_rate[1]
        return

    def update_IC_terms(self, update_IC=False):
        raise NotImplementedError("This method is not implemented by this class")

    def _has_IC(self, field='velocity', n=0):
        if field != 'lambda':
            return field in self.M_u0 and np.linalg.norm(self.M_u0[field]) > 0
        else:
            return field in self.M_u0 and np.linalg.norm(self.M_u0[field][n]) > 0

    def build_rb_nonparametric_RHS(self):
        """
        MODIFY
        """

        logger.debug(f'Building non-parametric RHS for '
                     f'method: {self.M_reduction_method}, mesh: {self.M_mesh_name}')

        if self._has_IC():
            self.update_IC_terms(update_IC=False)

        return

    def build_rb_parametric_RHS(self, param):
        """
        MODIFY
        """

        logger.debug(f'Building parametric RHS for param: {param}, '
                     f'method: {self.M_reduction_method}, mesh: {self.M_mesh_name}')

        self.M_f_Blocks_param = dict()

        param_map = self.differentiate_parameters(param)
        affine_params = list(self.M_f_Blocks_param_affine.keys())

        for param_type in affine_params:
            for iB in self.M_f_Blocks_param_affine[param_type]:
                vec = np.sum(_fun(param_map[param_type]) * _vector
                             for _vector, _fun in zip(self.M_f_Blocks_param_affine[param_type][iB],
                                                      self.M_f_Blocks_param_affine_fun[param_type][iB]))
                if iB not in self.M_f_Blocks_param.keys():
                    self.M_f_Blocks_param[iB] = vec
                else:
                    self.M_f_Blocks_param[iB] += vec

        return

    def _get_f_block(self, idx, coeff=1.0, param_coeff=1.0):
        """
        MODIFY
        """

        return (coeff * self.M_f_Blocks_no_param[idx] + param_coeff * self.M_f_Blocks_param[idx]
                if idx in self.M_f_Blocks_param.keys()
                else coeff * self.M_f_Blocks_no_param[idx])

    def build_rb_RHS(self, param=None):
        """
        MODIFY
        """

        self.build_rb_nonparametric_RHS()
        if param is not None:
            self.build_rb_parametric_RHS(param)

        self.M_f_Block = np.hstack([self._get_f_block(0), self._get_f_block(1), self._get_f_block(2)])

        return

    def build_rb_nonparametric_LHS(self):
        """
        MODIFY
        """
        raise NotImplementedError("This method is not implemented by this class")

    def build_rb_parametric_LHS(self, param):
        """
        MODIFY
        """

        logger.debug(f'Building parametric LHS blocks for param: {param}, '
                     f'method: {self.M_reduction_method}, mesh: {self.M_mesh_name}')

        self.M_Blocks_param = dict()

        param_map = self.differentiate_parameters(param)
        affine_params = list(self.M_Blocks_param_affine.keys())

        for param_type in affine_params:
            for iB in self.M_Blocks_param_affine[param_type]:
                mat = sum([_fun(param_map[param_type]) * _matrix
                           for _matrix, _fun in zip(self.M_Blocks_param_affine[param_type][iB],
                                                    self.M_Blocks_param_affine_fun[param_type][iB])])
                if iB not in self.M_Blocks_param.keys():
                    self.M_Blocks_param[iB] = mat
                else:
                    self.M_Blocks_param[iB] += mat

        return

    def _get_block(self, idx, coeff=1.0, param_coeff=1.0):
        """
        MODIFY
        """

        return (coeff * self.M_Blocks[idx] + param_coeff * self.M_Blocks_param[idx] if idx in self.M_Blocks_param.keys()
                else coeff * self.M_Blocks[idx])

    def build_rb_LHS(self, param, get_parametric=True):
        """
        MODIFY
        """

        if get_parametric:
            self.build_rb_parametric_LHS(param)

        __get_block = lambda i: self._get_block(i, coeff=1.0, param_coeff=float(get_parametric))

        self.M_Block = np.block([[__get_block(0), __get_block(1), __get_block(2)],
                                 [__get_block(3), __get_block(4), __get_block(5)],
                                 [__get_block(6), __get_block(7), __get_block(8)]])

        return

    def build_mu_independent_LHS(self):
        """
        Build a parameter-independent LHS and perform LU decomposition to speedup solve
        """

        self.build_rb_LHS(None, get_parametric=False)
        jac = np.copy(self.M_Block)

        self.M_Block_LU = scipy.linalg.lu_factor(jac)

        return

    def set_paths(self, _snapshot_matrix=None, _basis_matrix=None,
                  _affine_components=None,
                  _fom_structures=None, _reduced_structures=None,
                  _generalized_coords=None, _results=None):
        """
        Set data paths
        """

        self.M_snapshots_path = _snapshot_matrix
        self.M_basis_path = _basis_matrix
        self.M_affine_components_path = _affine_components
        self.M_fom_structures_path = _fom_structures
        self.M_reduced_structures_path = _reduced_structures
        self.M_generalized_coords_path = _generalized_coords

        self.M_results_path = _results

        return

    def set_test_paths(self, _snapshot_matrix=None):
        """
        Set test data paths
        """

        self.M_test_snapshots_path = _snapshot_matrix

        return

    def time_marching(self, *args, **kwargs):
        raise NotImplementedError("Time marching method available only with SRB-TFO method")

    def test_time_marching(self, *args, **kwargs):
        raise NotImplementedError("Time marching method available only with SRB-TFO method")

    def _reset_solution(self):
        """
        MODIFY
        """

        self.M_un = np.zeros(0)
        self.M_u_hat = dict()
        self.M_utildeh = dict()
        self.M_u0 = dict()

        return

    def _reset_errors(self):
        """
        MODIFY
        """

        for field in set(self.M_valid_fields) - {'lambda'}:
            self.M_relative_error[field] = np.zeros(self.M_Nt * self.M_N_periods)
            self.M_relative_error[field + '-l2'] = 0.0

        return

    def _update_errors(self, N=None):
        """
        MODIFY
        """

        if N is None:
            N = 1

        for field in set(self.M_valid_fields) - {'lambda'}:
            self.M_relative_error[field] += self.M_cur_errors[field] / N
            self.M_relative_error[field + '-l2'] += self.M_cur_errors[field + '-l2'] / N

        return

    def load_IC(self, param_nb, is_test=False, ss_ratio=1):
        """
        Load initial conditions from file, considering the last two timesteps of the first period
        """

        if not self.check_norm_matrices():
            self.get_norm_matrices()

        snapshots_path = self.M_test_snapshots_path if is_test else self.M_snapshots_path

        for field in set(self.M_valid_fields) - {'lambda'}:
            u0 = self._load_snapshot_file(os.path.join(snapshots_path, f'param{param_nb}'), field,
                                          save=False, remove=False)[1]
            if u0 is not None:
                norm_mat = self.M_norm_matrices[field] if field != "displacement" else self.M_norm_matrices['velocity']
                self.M_u0[field] = self.project_vector(u0.T, self.M_basis_space[field],
                                                       norm_matrix=norm_mat).T

        self.M_u0['lambda'] = [np.zeros(0)] * self.M_n_coupling
        for n in range(self.M_n_coupling):
            u0 = self._load_snapshot_file(os.path.join(snapshots_path, f'param{param_nb}'), f"lambda{n}",
                                          save=False, remove=False)[1]
            if u0 is not None:
                self.M_u0['lambda'][n] = self.project_vector(u0.T, self.M_basis_space['lambda'][n], norm_matrix=None).T

        return

    def _reconstruct_IC(self, field, n=0):
        """
        Reconstruct the initial condition for a given field from the available solution
        """
        raise NotImplementedError

    def _update_IC(self):
        """
        Update initial conditions
        """

        # sorted is crucial, so that displacement is updated before velocity in membrane model !
        for field in sorted(set(self.M_valid_fields) - {'lambda'}):
            # tmp = np.copy(self.M_u0[field])
            self.M_u0[field] = self._reconstruct_IC(field)
            # diff = np.linalg.norm(self.M_u0[field][0] - tmp[0]) / np.linalg.norm(tmp[0])
            # logger.warning(f"{field} initial condition relative change: {diff:.2e}")

        for n in range(self.M_n_coupling):
            self.M_u0['lambda'][n] = self._reconstruct_IC('lambda', n=n)

        return

    def compute_flow_rates(self, param_nb, solutions_fem, dts, is_test=False, **kwargs):
        """Compute flow rates at inlets and outlets based on the obtained velocity field."""

        _, Nt = self.M_fom_problem.time_specifics
        windows = [np.sum(dts[:i]) * Nt for i in range(kwargs['n_cycles'])]
        times = np.concatenate([np.linspace(np.sum(dts[:i]) * Nt + dts[i], np.sum(dts[:i + 1]) * Nt, Nt)
                                for i in range(kwargs['n_cycles'])])

        FEM_matrices = self.import_FEM_structures(structures={'q'})
        inflow_rates = [np.hstack([FEM_matrices['q_in'][k].dot(solutions_fem[i]['velocity'])
                                   for i in range(kwargs['n_cycles'])])
                        for k in range(self.M_n_inlets)]
        outflow_rates = [np.hstack([FEM_matrices['q_out'][k].dot(solutions_fem[i]['velocity'])
                                    for i in range(kwargs['n_cycles'])])
                         for k in range(self.M_n_outlets)]

        self._save_flow_rates(inflow_rates, outflow_rates, times, windows, param_nb, is_test=is_test)

        return

    def solve_pipeline(self, param_nb, is_test=False, ss_ratio=1, **kwargs):
        """
        MODIFY
        """

        start = time.perf_counter()

        print('\n')
        logger.info(f"Considering snapshot number {param_nb}" + (" (test)" if is_test else ""))

        snapshots_path = self.M_test_snapshots_path if is_test else self.M_snapshots_path

        fname = os.path.join(snapshots_path, f'param{param_nb}', 'coeffile.txt')
        param = np.genfromtxt(fname, delimiter=',')

        status = self.solve_reduced_problem(param)

        if status:
            return 1, 0.0

        elapsed_time = time.perf_counter() - start
        logger.info(f"Online execution wall time: {elapsed_time:.4f} s")

        logger.debug("Computing the errors")
        self.reconstruct_fem_solution(self.M_un)
        self.M_cur_errors = self.compute_online_errors(param_nb, is_test=is_test, ss_ratio=ss_ratio)
        self._update_errors()
        self._save_results_snapshot(param_nb, self.M_cur_errors, is_test=is_test)

        self.compute_flow_rates(param_nb, [self.M_utildeh], [self.dt], is_test=is_test, n_cycles=1)

        return status, elapsed_time

    def compute_WSS(self, velocity):
        """
        Compute the WSS given the velocity
        """

        logger.debug("Computing the WSS")

        try:
            WSS_left = self._load_FEM_matrix_file(os.path.join(self.M_fom_structures_path, 'WSS_left'),
                                                  matrixtype=csc_matrix, save=True, remove=True)

            WSS_right = self._load_FEM_matrix_file(os.path.join(self.M_fom_structures_path, 'WSS_right'),
                                                   matrixtype=csc_matrix, save=True, remove=True)

        except FileNotFoundError:
            logger.warning("WSS matrices not found! Impossible to compute the WSS.")
            WSS = None

        else:

            WSS_left.eliminate_zeros()
            nonzero_rows = np.unique(WSS_left.indices)

            row_mask = np.zeros(WSS_left.shape[0], dtype=bool)
            row_mask[nonzero_rows] = True

            WSS_left_bd = WSS_left[row_mask][:, row_mask]
            WSS_right_bd = WSS_right.dot(velocity)[row_mask]

            WSS_bd = arr_utils.solve_sparse_system(WSS_left_bd, WSS_right_bd)

            WSS = np.zeros_like(velocity)
            WSS[row_mask] = WSS_bd

        return WSS

    def solve_pipeline_cycles(self, param_nb, is_test=False, **kwargs):
        """
        MODIFY
        """

        start = time.perf_counter()

        print('\n')
        logger.info(f"Considering snapshot number {param_nb}" + (" (test)" if is_test else ""))

        snapshots_path = self.M_test_snapshots_path if is_test else self.M_snapshots_path

        fname = os.path.join(snapshots_path, f'param{param_nb}', 'coeffile.txt')
        param = np.genfromtxt(fname, delimiter=',')

        solutions, solutions_fem, dts, fom_rec_time = self.solve_reduced_problem_iteratively(param, **kwargs)
        status = solutions is None

        if status:
            return 1, 0.0

        elapsed_time = time.perf_counter() - start - fom_rec_time
        logger.info(f"Online execution wall time (with eventual FEM exporting): {elapsed_time:.4f} s")

        results_dir = os.path.join(self.M_results_path, f'param{param_nb}' + ('_test' if is_test else ''))
        gen_utils.create_dir(results_dir)

        if self.M_N_periods == kwargs['n_cycles']:
            for field in solutions_fem[0]:
                if field not in {'lambda'}:
                    self.M_utildeh[field] = np.hstack([solutions_fem[n][field]
                                                       for n in range(kwargs['n_cycles'])])
                else:
                    self.M_utildeh['lambda'] = [np.zeros(0)] * self.M_n_coupling
                    for _n in range(self.M_n_coupling):
                        self.M_utildeh['lambda'][_n] = np.hstack([solutions_fem[n]['lambda'][_n]
                                                                  for n in range(kwargs['n_cycles'])])

            self.M_cur_errors = self.compute_online_errors(param_nb, is_test=is_test)
            self._update_errors()
            self._save_errors(self.M_cur_errors, results_dir)

        self.compute_flow_rates(param_nb, solutions_fem, dts, is_test=is_test, **kwargs)

        if self.M_save_results:

            if not self.M_utildeh:
                for field in solutions_fem[0]:
                    if field not in {'lambda'}:
                        self.M_utildeh[field] = np.hstack([solutions_fem[n][field] for n in range(kwargs['n_cycles'])])

            self._save_solution(param_nb, results_dir, is_test=is_test, n_cycles=kwargs['n_cycles'],
                                save_reduced=False, save_full=True, save_FOM=False, save_lambda=False)

        return 0, elapsed_time

    def test_rb_solver(self, param_nbs, is_test=False, ss_ratio=1, cycles_specifics=None, **kwargs):
        """
        Solve the ST-reduced problem for param numbers in param_nbs
        """

        if cycles_specifics is None:
            cycles_specifics = {'n_cycles': 1,
                                'update_T': False,
                                'update_params': False}

        self._reset_errors()
        self.M_online_mean_time = 0.0

        solves_cnt = 0
        for param_nb in param_nbs:

            self.load_IC(param_nb, is_test=is_test, ss_ratio=ss_ratio)

            if cycles_specifics['n_cycles'] == 1:
                status, elapsed_time = self.solve_pipeline(param_nb, is_test=is_test, ss_ratio=ss_ratio, **kwargs)
            else:
                status, elapsed_time = self.solve_pipeline_cycles(param_nb, is_test=is_test, **cycles_specifics)

            if status == 0:
                solves_cnt += 1
                self.M_online_mean_time += elapsed_time

            self._reset_solution()
            self.update_IC_terms(update_IC=False)  # to zero-out the IC-related terms

        if solves_cnt > 0:
            self.M_relative_error = {key: self.M_relative_error[key] / solves_cnt for key in self.M_relative_error}

            self.M_online_mean_time /= solves_cnt
            print('\n')
            logger.info(f"Average online execution wall time: {self.M_online_mean_time:.4f} s")

            if cycles_specifics['n_cycles'] == 1:
                self._save_results_general()

        return

    def _save_solution(self, param_nb, save_path, is_test=False, n_cycles=None, compute_WSS=True,
                       save_reduced=True, save_full=True, save_FOM=True, save_lambda=True):
        """
        MODIFY
        """

        if n_cycles is not None:
            assert not save_FOM, "Full order solution exporting not available with more than 1 cycle"

        snapshots_path = self.M_test_snapshots_path if is_test else self.M_snapshots_path

        if save_FOM:
            snap_velocity = self._load_snapshot_file(os.path.join(snapshots_path, f'param{param_nb}'), 'velocity',
                                                     save=False, remove=False)[0].T
            snap_pressure = self._load_snapshot_file(os.path.join(snapshots_path, f'param{param_nb}'), 'pressure',
                                                     save=False, remove=False)[0].T
            snap_lambda = []
            for n in range(self.M_n_coupling):
                snap_lambda.append(self._load_snapshot_file(os.path.join(snapshots_path, f'param{param_nb}'), f'lambda{n}',
                                                            save=False, remove=False)[0].T)

            FEM_folder_name = os.path.join(save_path, 'FEM', 'Block0')
            gen_utils.create_dir(FEM_folder_name)

        STRB_folder_name = os.path.join(save_path, 'RB' + ('' if n_cycles is None else f'_T{n_cycles}'), 'Block0')
        gen_utils.create_dir(STRB_folder_name)

        if save_reduced:

            np.save(os.path.join(STRB_folder_name, 'velocity_strb'),
                    self.M_un[:self.M_N['velocity']])
            np.save(os.path.join(STRB_folder_name, 'pressure_strb'),
                    self.M_un[self.M_N['velocity']:self.M_N['velocity'] + self.M_N['pressure']])
            np.save(os.path.join(STRB_folder_name, f'lambda_strb'),
                    self.M_un[self.M_N['velocity'] + self.M_N['pressure']:])

            if save_FOM:
                np.save(os.path.join(FEM_folder_name, 'velocity_strb'),
                        self.compute_generalized_coordinates(snap_velocity, field="velocity"))
                np.save(os.path.join(FEM_folder_name, 'pressure_strb'),
                        self.compute_generalized_coordinates(snap_pressure, field="pressure"))
                np.save(os.path.join(FEM_folder_name, 'lambda_strb'),
                        self.compute_generalized_coordinates([snap for snap in snap_lambda], field="lambda"))

        if save_full:

            if os.path.isdir(os.path.join(self.M_snapshots_path, 'Template')):
                logger.debug('Dumping FOM solutions to .h5 files')

                shutil.copytree(os.path.join(self.M_snapshots_path, 'Template'),
                                os.path.join(STRB_folder_name, 'Solution'), dirs_exist_ok=True)

                gen_utils.write_field_to_h5(os.path.join(STRB_folder_name, 'Solution', 'block0.h5'),
                                            'velocity', self.M_utildeh['velocity'], field_dim=3)
                gen_utils.write_field_to_h5(os.path.join(STRB_folder_name, 'Solution', 'block0.h5'),
                                            'pressure', self.M_utildeh['pressure'])

                if self.M_utildeh['velocity'].shape[1] > self.M_Nt:
                    hdf5_file = os.path.join(STRB_folder_name, 'Solution', 'block0.h5')
                    xmf_file = hdf5_file.replace(".h5", ".xmf")
                    gen_utils.update_xmf_file(xmf_file, xmf_file,
                                              self.M_Nt, self.M_utildeh['velocity'].shape[1], self.dt)

                if compute_WSS:
                    WSS = self.compute_WSS(self.M_utildeh['velocity'])
                    if WSS is not None:
                        gen_utils.write_field_to_h5(os.path.join(STRB_folder_name, 'Solution', 'block0.h5'),
                                                    'WSS', WSS, field_dim=3)
                        gen_utils.add_field_to_xmf(os.path.join(STRB_folder_name, 'Solution', 'block0.xmf'), 'WSS')

                if save_FOM:
                    shutil.copytree(os.path.join(self.M_snapshots_path, 'Template'),
                                    os.path.join(FEM_folder_name, 'Solution'), dirs_exist_ok=True)

                    gen_utils.write_field_to_h5(os.path.join(FEM_folder_name, 'Solution', 'block0.h5'),
                                                'velocity', snap_velocity, field_dim=3)
                    gen_utils.write_field_to_h5(os.path.join(FEM_folder_name, 'Solution', 'block0.h5'),
                                                'pressure', snap_pressure)

                    if compute_WSS:
                        WSS = self.compute_WSS(snap_velocity)
                        if WSS is not None:
                            gen_utils.write_field_to_h5(os.path.join(FEM_folder_name, 'Solution', 'block0.h5'),
                                                        'WSS', WSS, field_dim=3)
                            gen_utils.add_field_to_xmf(os.path.join(FEM_folder_name, 'Solution', 'block0.xmf'), 'WSS')

            else:
                logger.debug('Dumping FOM solutions to .txt files')

                arr_utils.save_array(self.M_utildeh['velocity'].T, os.path.join(STRB_folder_name, 'velocity.txt'))
                arr_utils.save_array(self.M_utildeh['pressure'].T, os.path.join(STRB_folder_name, 'pressure.txt'))

                if save_FOM:
                    arr_utils.save_array(snap_velocity.T, os.path.join(FEM_folder_name, 'velocity.txt'))
                    arr_utils.save_array(snap_pressure.T, os.path.join(FEM_folder_name, 'pressure.txt'))

            # storing Lagrange multipliers in .npy files
            if save_lambda:
                np.save(os.path.join(STRB_folder_name, 'lambda.npy'),
                        np.hstack([snap.T for snap in self.M_utildeh['lambda']]))

                if save_FOM:
                    np.save(os.path.join(FEM_folder_name, 'lambda.npy'),
                            np.hstack([snap.T for snap in snap_lambda]))

        return

    def _fill_errors(self, errors, data=None):
        """
        MODIFY
        """

        if data is None:
            data = dict()

        for field in set(self.M_valid_fields) - {'lambda'}:
            data[field] = errors[field + '-l2']

        for n in range(self.M_n_coupling):
            data[f"Lagrange multipliers {n}"] = errors['lambda-l2'][n]

        return data

    def _save_errors(self, errors, save_path):
        """
        MODIFY
        """

        errors_folder_name = os.path.join(save_path, 'errors')
        gen_utils.create_dir(errors_folder_name)

        data = self._fill_errors(errors)

        with open(os.path.join(errors_folder_name, 'errors.json'), "w") as file:
            json.dump(data, file)

        for field in set(self.M_valid_fields) - {'lambda'}:
            np.save(os.path.join(errors_folder_name, f'{field}.npy'), errors[field])
            np.save(os.path.join(errors_folder_name, f'{field}-ref.npy'), errors[field + '-ref'])

        return

    def _save_results_snapshot(self, param_nb, errors, is_test=False):
        """
        MODIFY
        """

        if self.M_save_results:
            logger.debug(f"Dumping the errors and the FEM and the RB solution for parameter {param_nb} to file\n")
        else:
            logger.debug(f"Dumping the errors for parameter {param_nb} to file\n")

        folder_name = os.path.join(self.M_results_path, f'param{param_nb}' + ('_test' if is_test else ''))
        gen_utils.create_dir(folder_name)

        if self.M_save_results:
            self._save_solution(param_nb, folder_name, is_test=is_test)

        self._save_errors(errors, folder_name)

        return

    def _save_results_general(self):
        """
        MODIFY
        """

        ERRORS_folder_name = os.path.join(self.M_results_path, 'avg_errors')
        gen_utils.create_dir(ERRORS_folder_name)

        for field in set(self.M_valid_fields) - {'lambda'}:
            np.save(os.path.join(ERRORS_folder_name, f'{field}.npy'), self.M_relative_error[field])

        np.savetxt(os.path.join(self.M_results_path, 'online_time.txt'), np.array([self.M_online_mean_time]))

        return

    def _save_flow_rates(self, inflow_rates, outflow_rates, times, windows, param_nb,
                         is_test=False, save_fig=True):
        """
        MODIFY
        """

        folder_flows = os.path.join(self.M_results_path, f'param{param_nb}' + ('_test' if is_test else ''), 'flows')
        gen_utils.create_dir(folder_flows)

        for idx, inflow in enumerate(inflow_rates):
            np.save(os.path.join(folder_flows, f'inflow{idx}.npy'), inflow)
        for idx, outflow in enumerate(outflow_rates):
            np.save(os.path.join(folder_flows, f'outflow{idx}.npy'), outflow)

        if not save_fig:
            return

        import matplotlib.pyplot as plt
        plt.set_loglevel("warning")

        def __setup_axs(_axs, _title=None):
            _axs.grid(which='major', linestyle='-', linewidth=1)
            _axs.grid(which='minor', linestyle='--', linewidth=0.5)
            if _title is not None and type(_title) is str:
                _axs.set_title(_title, fontsize=20, fontweight='bold')
            _axs.set_xlabel('Time [$s$]')
            _axs.set_ylabel(r'Flow [$cm^3/s$]')
            return

        fig, axs = plt.subplots(self.M_n_inlets, 1)
        for i in range(self.M_n_inlets):
            cur_axs = axs if self.M_n_inlets == 1 else axs[i]
            cur_axs.plot(times, -inflow_rates[i])
            if len(windows) > 1:
                cur_axs.vlines(windows, np.min(-inflow_rates[i]), np.max(-inflow_rates[i]), 'r', linestyles='dashdot')
            __setup_axs(cur_axs, _title=f"Inflow {i}")
        plt.savefig(os.path.join(folder_flows, f'inflows.eps'), format='eps', dpi=300)
        plt.close()

        fig, axs = plt.subplots(self.M_n_outlets, 1)
        for i in range(self.M_n_outlets):
            cur_axs = axs if self.M_n_outlets == 1 else axs[i]
            cur_axs.plot(times, outflow_rates[i])
            if len(windows) > 1:
                cur_axs.vlines(windows, np.min(outflow_rates[i]), np.max(outflow_rates[i]), 'r', linestyles='dashdot')
            __setup_axs(cur_axs, _title=f"Outflow {i}")
        plt.savefig(os.path.join(folder_flows, f'outflows.eps'), format='eps', dpi=300)
        plt.close()

        delta_Q = np.sum(np.array(inflow_rates), axis=0) + np.sum(np.array(outflow_rates), axis=0)
        fig, axs = plt.subplots(1, 1)
        axs.plot(times, np.sum(-np.array(inflow_rates), axis=0), label="Inflow")
        axs.plot(times, np.sum(np.array(outflow_rates), axis=0), label='Outflow')
        axs.plot(times, delta_Q, label='Delta')
        if len(windows) > 1:
            axs.vlines(windows, np.min(outflow_rates[i]), np.max(outflow_rates[i]), 'k', linestyles='dashdot')
        __setup_axs(axs, _title=f"Flow difference")
        axs.legend()
        plt.savefig(os.path.join(folder_flows, f'flow_diff.eps'), format='eps', dpi=300)
        plt.close()

        return

    def _compute_error_field(self, sol, sol_true, field):
        """
        Compute the error field for a given field
        """

        error = sol - sol_true
        error = self.compute_norm(error, field=field)
        denom = self.compute_norm(sol_true, field=field)
        error_l2 = np.linalg.norm(error) / np.linalg.norm(denom)

        logger.info(f"{field} absolute error: {np.linalg.norm(error):.2e}")
        logger.info(f"{field} relative error: {error_l2:.2e}")

        return error, denom, error_l2

    def compute_online_errors(self, param_nb, sol=None, is_test=False, ss_ratio=1):
        """
        Compute the errors (H1-norm for velocity and L2-norm for pressure, in both cases l2 in time) after online
        phase for param 'param_nb'.

        Remark: if the FOM data feature multiple periods (e.g. heartbeats), the errors are computed only
                with respect to the first one.
        """

        if sol is not None:
            assert type(sol) is dict and set(self.M_valid_fields).issubset(sol.keys()), "Invalid solution format"
        else:
            sol = self.M_utildeh

        if not self.check_norm_matrices():
            self.get_norm_matrices()

        errors = dict()

        snapshots_path = self.M_test_snapshots_path if is_test else self.M_snapshots_path
        is_h5 = os.path.isfile(os.path.join(snapshots_path, f'param{param_nb}', 'block0.h5'))

        for k,field in enumerate(set(self.M_valid_fields) - {"lambda"}):
            remove = not is_h5 or k == len(self.M_valid_fields) - 2
            snap = self._load_snapshot_file(os.path.join(snapshots_path, f'param{param_nb}'), field,
                                            save=True, remove=remove)[0][ss_ratio-1::ss_ratio].T
            error, denom, error_l2 = self._compute_error_field(sol[field], snap, field)

            errors[field] = np.array(error)
            errors[field + '-ref'] = np.array(denom)
            errors[field + '-l2'] = error_l2

        errors_l, denoms_l, errors_l_l2 = [], [], []
        for n in range(self.M_n_coupling):
            snap_lambda = self._load_snapshot_file(os.path.join(snapshots_path, f'param{param_nb}'), f'lambda{n}',
                                                   save=True, remove=True)[0][ss_ratio-1::ss_ratio].T
            error, denom, error_l2 = self._compute_error_field(sol['lambda'][n], snap_lambda, f'lambda{n}')

            errors_l.append(error)
            denoms_l.append(denom)
            errors_l_l2.append(error_l2)

        errors['lambda'] = np.array(errors_l)
        errors['lambda-ref'] = np.array(denoms_l)
        errors['lambda-l2'] = np.array(errors_l_l2)

        return errors

    def postprocess_GS_errors(self, params, errors, param_names, folder):
        """ Draw plots and save files from GS-like analysis
        """

        dt = self.dt
        times = np.linspace(dt, dt * self.M_Nt, self.M_Nt)

        import matplotlib.pyplot as plt
        plt.set_loglevel("warning")

        plt.figure()
        for (param, error) in zip(params, errors):
            labels = [f"{_param_name} = {_param}" for (_param_name, _param) in zip(param_names, param)]
            plt.semilogy(times, error['velocity'], label=' '.join(labels))
        plt.title(r'Velocity')
        plt.xlabel('Time (s)')
        plt.ylabel(r'$H^1$-norm error')
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(folder, 'velocity_error.eps'))

        plt.figure()
        for (param, error) in zip(params, errors):
            labels = [f"{_param_name} = {_param}" for (_param_name, _param) in zip(param_names, param)]
            plt.semilogy(times, error['pressure'], label=' '.join(labels))
        plt.title(r'Pressure')
        plt.xlabel('Time (s)')
        plt.ylabel(r'$L^2$-norm error')
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(folder, 'pressure_error.eps'))

        result = dict(zip([repr(param) for param in params], errors))
        import pickle
        with open(os.path.join(folder, "result_dict.pkl"), 'wb') as f:
            pickle.dump(result, f)

        return

    def check_dataset(self, _nsnap):

        if arr_utils.is_empty(self.M_snapshots_matrix['velocity']):
            self.import_snapshots_matrix(_nsnap)

        if not self.check_norm_matrices():
            self.get_norm_matrices()

        t = np.random.randint(2, self.M_Nt)
        dt = self.dt
        t_n = dt * (t + 1)
        T = dt * self.M_Nt
        FEM_matrices = self.import_FEM_structures()

        mom_err = np.zeros(self.M_Nh['velocity'])
        cont_err = np.zeros(self.M_Nh['pressure'])
        coupl_err = np.zeros(self.M_N_lambda_cumulative[-1])

        for ns in range(_nsnap):
            u_n = self.M_snapshots_matrix['velocity'][:, ns * self.M_Nt + t]
            u_n_1 = self.M_snapshots_matrix['velocity'][:, ns * self.M_Nt + t - 1]
            u_n_2 = self.M_snapshots_matrix['velocity'][:, ns * self.M_Nt + t - 2]
            p_n = self.M_snapshots_matrix['pressure'][:, ns * self.M_Nt + t]

            fname = os.path.join(self.M_snapshots_path,
                                 os.path.normpath('param' + str(ns) + '/coeffile.txt'))
            param = np.genfromtxt(fname, delimiter=',')

            l_n = [np.zeros(0)] * self.M_n_coupling
            RHS_param = [np.zeros(0)] * self.M_n_coupling
            base_flow = 1 - np.cos(2 * np.pi * t_n / T) + \
                        param[1] * np.sin(2 * np.pi * param[0] * t_n / T)
            flow_rate_n = 0
            for n in range(self.M_n_coupling):
                l_n[n] = self.M_snapshots_matrix['lambda'][n][:, ns * self.M_Nt + t]
                if n == 0:
                    flow_rate_n = param[2] * base_flow
                elif n == 1:
                    if 'tube' in self.M_mesh_name:
                        raise ValueError("Test cases in tubes only admit a single coupling!")
                    elif 'bif' in self.M_mesh_name or 'bypass' in self.M_mesh_name:
                        flow_rate_n = param[3] * base_flow

                RHS_param[n] = FEM_matrices['RHS'][n] * flow_rate_n

            l_n = np.vstack([l_n[n] for n in range(self.M_n_coupling)])
            B = np.vstack([FEM_matrices['B'][n] for n in range(self.M_n_coupling)])
            BT = np.hstack([FEM_matrices['BT'][n] for n in range(self.M_n_coupling)])
            RHS_param = np.vstack([RHS_param[n] for n in range(self.M_n_coupling)])

            mom_err += ((FEM_matrices['M'] + self.M_bdf_rhs * dt * FEM_matrices['A']).dot(u_n)
                        + self.M_bdf[0] * FEM_matrices['M'].dot(u_n_1) + self.M_bdf[1] * FEM_matrices['M'].dot(u_n_2)
                        + self.M_bdf_rhs * dt * FEM_matrices['BdivT'].dot(p_n) + self.M_bdf_rhs * dt * BT.dot(l_n)) / _nsnap
            cont_err += (FEM_matrices['Bdiv'].dot(u_n)) / _nsnap
            coupl_err += (B.dot(u_n) - RHS_param) / _nsnap

        mom_err_H1 = self.compute_norm(mom_err, field='velocity')
        cont_err_L2 = self.compute_norm(cont_err, field='pressure')
        coupl_err_l2 = np.sqrt(coupl_err.T.dot(coupl_err))

        logger.info(f"Momentum equation error: {mom_err_H1}")
        logger.info(f"Continuity equation error: {cont_err_L2}")
        logger.info(f"Coupling equation error: {coupl_err_l2}")

        return mom_err_H1, cont_err_L2, coupl_err_l2

    def check_offline_phase(self, _nsnap):

        if arr_utils.is_empty(self.M_snapshots_matrix['velocity']):
            self.import_snapshots_matrix(_nsnap)

        if len(self.M_N) == 0:
            tolerances = dict()
            tolerances['velocity-space'] = 1e-4
            tolerances['velocity-time'] = 1e-4
            tolerances['pressure-space'] = 1e-4
            tolerances['pressure-time'] = 1e-4
            tolerances['lambda-space'] = None
            tolerances['lambda-time'] = 1e-4
            self.build_ST_basis(tolerances)

        self.get_generalized_coordinates(_nsnap)

        offline_err_H1_rel = 0
        offline_err_L2_rel = 0
        offline_err_l2_rel = [0] * self.M_n_coupling

        for i in range(_nsnap):
            fname = os.path.join(self.M_snapshots_path,
                                 os.path.normpath('param' + str(i) + '/coeffile.txt'))
            param = np.genfromtxt(fname, delimiter=',')

            logger.debug(f"\nConsidering snapshot number {i}\nParameters: {param}")

            offline_err_H1 = np.zeros(self.M_Nt)
            offline_err_L2 = np.zeros(self.M_Nt)
            denom_H1 = np.zeros(self.M_Nt)
            denom_L2 = np.zeros(self.M_Nt)

            lambda_hat = np.hstack([self.M_u_hat['lambda'][n][:, i] for n in range(self.M_n_coupling)])
            self.reconstruct_fem_solution(np.hstack((self.M_u_hat['velocity'][:, i].flatten(),
                                                     self.M_u_hat['pressure'][:, i].flatten(),
                                                     lambda_hat.flatten())))

            err_vel = self.M_utildeh['velocity'] - self.M_snapshots_matrix['velocity'][:, self.M_Nt * i: self.M_Nt * (i + 1)]
            err_press = self.M_utildeh['pressure'] - self.M_snapshots_matrix['pressure'][:, self.M_Nt * i: self.M_Nt * (i + 1)]

            for iT in range(self.M_Nt):
                offline_err_H1[iT] += self.compute_norm(err_vel[:, iT], field='velocity')
                denom_H1[iT] += self.compute_norm(self.M_snapshots_matrix['velocity'][:, self.M_Nt * i + iT],
                                                  field='velocity')

                offline_err_L2[iT] += self.compute_norm(err_press[:, iT], field='pressure')
                denom_L2[iT] += self.compute_norm(self.M_snapshots_matrix['pressure'][:, self.M_Nt * i + iT],
                                                  field='pressure')

            logger.debug(f"Snapshot {i} - H1-l2 velocity relative error: {np.linalg.norm(offline_err_H1 / denom_H1)}")
            logger.debug(f"Snapshot {i} - L2-l2 pressure relative error: {np.linalg.norm(offline_err_L2 / denom_L2)}")

            offline_err_H1_rel += np.linalg.norm(offline_err_H1 / denom_H1) / _nsnap
            offline_err_L2_rel += np.linalg.norm(offline_err_L2 / denom_L2) / _nsnap

            for n in range(self.M_n_coupling):
                err_lambda = self.M_utildeh['lambda'][n] - \
                             self.M_snapshots_matrix['lambda'][n][:, self.M_Nt * i: self.M_Nt * (i + 1)]
                offline_err_l2 = np.sqrt(np.sum(np.square(err_lambda), axis=0))
                denom_l2 = np.sqrt(np.sum(np.square(self.M_snapshots_matrix['lambda'][n][:, self.M_Nt * i:
                                                                                            self.M_Nt * (i + 1)]),
                                          axis=0))

                logger.debug(f"Snapshot {i} - l2-l2 Lagrange multiplier {n} relative error:"
                             f" {np.linalg.norm(offline_err_l2 / denom_l2)}")

                offline_err_l2_rel[n] += np.linalg.norm(offline_err_l2 / denom_l2) / _nsnap

            print('\n')

        # import matplotlib.pyplot as plt
        # dt = self.dt
        # times = np.linspace(dt, dt * self.M_Nt, self.M_Nt)
        # plt.semilogy(times, offline_err_H1_rel, label='H1 vel')
        # plt.semilogy(times, offline_err_L2_rel, label='L2 press')
        # plt.semilogy(times, offline_err_l2_rel, label='l2 mult')
        # plt.legend()
        # plt.show()

        print('\n\n')
        logger.info(f"Average H1-l2 velocity relative error: {offline_err_H1_rel}")
        logger.info(f"Average L2-l2 pressure relative error: {offline_err_L2_rel}")
        for n in range(self.M_n_coupling):
            logger.info(f"Average l2-l2 Lagrange multiplier {n} relative error: {offline_err_l2_rel[n]}")

        return offline_err_H1_rel, offline_err_L2_rel, offline_err_l2_rel


__all__ = [
    "RbManagerSpaceTimeStokes"
]
