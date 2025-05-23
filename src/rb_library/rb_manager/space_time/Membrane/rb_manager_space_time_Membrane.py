#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 3 16:10:21 2023
@author: Riccardo Tenderini
@mail: riccardo.tenderini@epfl.ch
"""

import numpy as np
import scipy.sparse
from sklearn.gaussian_process.kernels import RBF
import os
import shutil
import time

import src.rb_library.rb_manager.space_time.Navier_Stokes.rb_manager_space_time_Navier_Stokes as rbmstNS
import src.utils.array_utils as arr_utils
import src.utils.general_utils as gen_utils

from scipy.sparse import csc_matrix

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../../../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class RbManagerSpaceTimeMembrane(rbmstNS.RbManagerSpaceTimeNavierStokes):
    """
    MODIFY
    """
    def __init__(self, _fom_problem, _affine_decomposition=None):
        """ MODIFY
        """

        super().__init__(_fom_problem, _affine_decomposition=_affine_decomposition)

        self.M_valid_fields = ['velocity', 'pressure', 'displacement', 'lambda']

        self.M_snapshots_matrix['displacement'] = np.zeros(0)
        self.M_test_snapshots_matrix['displacement'] = np.zeros(0)

        self.M_Abd_matrices = [np.zeros(0)] * 3
        self.M_Mbd_matrix = np.zeros(0)
        self.M_MbdWall_matrix = np.zeros(0)

        self.M_Nh['displacement'] = 0
        self.M_N_space['displacement'] = 0
        self.M_N_time['displacement'] = 0
        self.M_N['displacement'] = 0

        self.M_basis_time_IC_int = np.zeros(0)

        self.M_param_ref = np.array([1.2, 0.1, 4e6, 0.5])  # density, thickness, Young, Poisson
        self.M_wall_elasticity = 0
        self.M_default_coefficients = np.zeros(0)
        self._compute_default_coefficients()

        self.M_bd_indexes = np.zeros(0)

        self.M_valid_parametrizations = ['inflow', 'outflow', 'fluid', 'structure', 'clot']

        return

    @property
    def wall_elasticity(self):
        return self.M_wall_elasticity

    @wall_elasticity.setter
    def wall_elasticity(self, value):
        self.M_wall_elasticity = value
        return

    def _has_wall_elasticity(self):
        return self.M_wall_elasticity > 0 and not arr_utils.is_empty(self.M_MbdWall_matrix)

    def get_boundary_dofs(self):
        """
        MODIFY
        """

        if self.M_bd_indexes.size > 0 and self.M_bd_indexes.shape[0] < self.M_norm_matrices['displacement'].shape[0]:
            return

        norms = scipy.sparse.linalg.norm(self.M_norm_matrices['displacement'], axis=0)
        self.M_bd_indexes = np.arange(norms.shape[0])  # otherwise large errors on displacement !!
        # self.M_bd_indexes = np.where(norms > 1e-10)[0]

        return

    def get_norm_matrices(self, matrixtype=csc_matrix, check_spd=False):
        """
        MODIFY
        """

        super().get_norm_matrices(matrixtype=matrixtype, check_spd=check_spd)

        self.M_norm_matrices['displacement'] = self._load_FEM_matrix_file(os.path.join(self.M_fom_structures_path, 'norm2'),
                                                                          matrixtype=matrixtype,
                                                                          save=True, remove=True)

        self.get_boundary_dofs()
        self.M_norm_matrices['displacement'] = self.M_norm_matrices['displacement'][self.M_bd_indexes, :][:, self.M_bd_indexes]

        if check_spd:
            if not (arr_utils.is_symmetric(self.M_norm_matrices['displacement']) and
                    arr_utils.is_positive_definite(self.M_norm_matrices['displacement'])):
                logger.warning("The displacement norm matrix is not SPD!")

        return

    def check_norm_matrices(self):
        """
        MODIFY
        """

        return (super().check_norm_matrices() and
                ('displacement' in self.M_norm_matrices.keys()) and
                (self.M_norm_matrices['displacement'].shape[0] > 0))

    def _import_snapshots(self, _ns=None, is_test=False, ss_ratio=1):
        """
        MODIFY
        """

        assert self.M_import_snapshots, "Snapshots import disabled"

        start = time.time()

        if not self.check_norm_matrices():
            self.get_norm_matrices()

        if not is_test:
            old_snap_u = self.M_snapshots_matrix['velocity']
            old_snap_p = self.M_snapshots_matrix['pressure']
            old_snap_d = self.M_snapshots_matrix['displacement']
            old_snap_lambda = self.M_snapshots_matrix['lambda']
            old_snap_params = self.M_offline_ns_parameters
            path = self.M_snapshots_path
        else:
            old_snap_u = self.M_test_snapshots_matrix['velocity']
            old_snap_p = self.M_test_snapshots_matrix['pressure']
            old_snap_d = self.M_test_snapshots_matrix['displacement']
            old_snap_lambda = self.M_test_snapshots_matrix['lambda']
            old_snap_params = self.M_test_offline_ns_parameters
            path = self.M_test_snapshots_path

        if not arr_utils.is_empty(old_snap_u) and not arr_utils.is_empty(old_snap_p) and not arr_utils.is_empty(old_snap_d) and \
           all([not arr_utils.is_empty(old_snap_lambda[n]) for n in range(self.M_n_coupling)]):
            assert self.M_Nt > 0
            _ns_cur = int(old_snap_u.shape[1]/self.M_Nt)
            logger.warning(f"{_ns_cur} snapshots already available. Importing {_ns - _ns_cur} new snapshots.")
        else:
            _ns_cur = 0
            logger.info(f"Reading {_ns} snapshots ...")

        count = 0
        if _ns is None:
            _ns = 0
            while os.path.isdir(os.path.join(path, f'param{_ns}')):
                _ns += 1

        snap_u, snap_p, snap_d, snap_lambda, snap_params = [], [], [], {n: [] for n in range(self.M_n_coupling)}, []

        for i in range(_ns_cur, _ns):

            cur_path = os.path.join(path, f'param{i}')
            is_h5 = os.path.isfile(os.path.join(cur_path, 'block0.h5'))

            if is_h5 and not os.path.isdir(os.path.join(path, 'Template')):
                gen_utils.create_dir(os.path.join(path, 'Template'))
                shutil.copy(os.path.join(cur_path, 'block0.h5'), os.path.join(path, 'Template', 'block0.h5'))
                shutil.copy(os.path.join(cur_path, 'block0.xmf'), os.path.join(path, 'Template', 'block0.xmf'))

            try:
                snap_u.append(self._load_snapshot_file(cur_path, 'velocity',
                                                       save=True, remove=not is_h5)[ss_ratio-1::ss_ratio].T)
                snap_p.append(self._load_snapshot_file(cur_path, 'pressure',
                                                       save=True, remove=not is_h5)[ss_ratio-1::ss_ratio].T)
                snap_d.append(self._load_snapshot_file(cur_path, 'displacement',
                                                       save=True, remove=True)[:, self.M_bd_indexes][ss_ratio-1::ss_ratio].T)
                snap_params.append(np.genfromtxt(os.path.join(cur_path, 'coeffile.txt'), delimiter=',')[None])

                for n in range(self.M_n_coupling):
                    snap_lambda[n].append(self._load_snapshot_file(cur_path, f'lambda{n}',
                                                                   save=True, remove=True)[ss_ratio-1::ss_ratio].T)

                count += 1

            except (OSError, IOError, FileNotFoundError):
                logger.warning(f"Impossible to load files for snapshot number {i}")

        if count == 0:
            logger.warning("No snapshots available!")
            return False
        else:
            assert snap_u[-1].shape[1] % self.M_N_periods == 0, \
                (f"The total number of FOM time instants ({snap_u[-1].shape[1]}) is not a multiple of selected "
                 f"the number of periods {self.M_N_periods}")
            self.M_Nt = (snap_u[-1].shape[1]) // self.M_N_periods

        logger.info(f"Number of read snapshots : {count}")

        if _ns_cur == 0:
            snap_u, snap_p, snap_d = np.hstack(snap_u), np.hstack(snap_p), np.hstack(snap_d)
            snap_lambda = [np.hstack(_snap_lambda) for _,_snap_lambda in snap_lambda.items()]
            snap_params = np.vstack(snap_params)
        else:
            snap_u = np.concatenate((old_snap_u, np.hstack(snap_u)), axis=1)
            snap_p = np.concatenate((old_snap_p, np.hstack(snap_p)), axis=1)
            snap_d = np.concatenate((old_snap_d, np.hstack(snap_d)), axis=1)
            snap_lambda = [np.concatenate((_old_snap_lambda, np.hstack(_snap_lambda)), axis=1)
                           for (_old_snap_lambda, (_,_snap_lambda)) in zip(old_snap_lambda, snap_lambda.items())]
            snap_params = np.concatenate((old_snap_params, np.vstack(snap_params)), axis=0)

        logger.debug(f"Size of velocity snapshots matrix : {snap_u.shape}")
        logger.debug(f"Size of pressure snapshots matrix : {snap_p.shape}")
        logger.debug(f"Size of displacement snapshots matrix : {snap_d.shape}")
        for n in range(self.M_n_coupling):
            logger.debug(f"Size of multipliers {n} snapshots matrix : {snap_lambda[n].shape}")
        logger.debug(f"Size of parameters matrix: {snap_params.shape}")

        if not is_test:
            self.M_snapshots_matrix['velocity'] = snap_u
            self.M_snapshots_matrix['pressure'] = snap_p
            self.M_snapshots_matrix['displacement'] = snap_d
            self.M_snapshots_matrix['lambda'] = snap_lambda
            self.M_offline_ns_parameters = snap_params
        else:
            self.M_test_snapshots_matrix['velocity'] = snap_u
            self.M_test_snapshots_matrix['pressure'] = snap_p
            self.M_test_snapshots_matrix['displacement'] = snap_d
            self.M_test_snapshots_matrix['lambda'] = snap_lambda
            self.M_test_offline_ns_parameters = snap_params

        self._compute_parameter_bounds(is_test=is_test)

        logger.debug(f"Snapshots importing duration: {(time.time() - start):.4f} s")

        return True

    def import_snapshots_matrix(self, _ns=None, ss_ratio=1):
        """
        MODIFY
        """

        import_success = super().import_snapshots_matrix(_ns=_ns, ss_ratio=ss_ratio)

        self.M_Nh['displacement'] = self.M_snapshots_matrix['displacement'].shape[0]

        return import_success

    def import_test_snapshots_matrix(self, _ns=None, ss_ratio=1):
        """
        MODIFY
        """

        super().import_test_snapshots_matrix(_ns=_ns, ss_ratio=ss_ratio)

        self.M_Nh['displacement'] = self.M_test_snapshots_matrix['displacement'].shape[0]

        return

    def get_snapshot(self, _snapshot_number, _fom_coordinates=np.array([]), timesteps=None,
                     field='velocity', n=0):
        """
        MODIFY
        """

        if field == 'displacement':
            return self._get_snapshot(self.M_snapshots_matrix['displacement'], _snapshot_number,
                                      _fom_coordinates=_fom_coordinates, timesteps=timesteps)
        else:
            return super().get_snapshot(_snapshot_number,
                                        _fom_coordinates=_fom_coordinates, timesteps=timesteps, field=field, n=n)

    def get_test_snapshot(self, _snapshot_number, _fom_coordinates=np.array([]), timesteps=None,
                          field='velocity', n=0):
        """
        MODIFY
        """

        if field == 'displacement':
            return self._get_snapshot(self.M_test_snapshots_matrix['displacement'], _snapshot_number,
                                      _fom_coordinates=_fom_coordinates, timesteps=timesteps)
        else:
            return super().get_test_snapshot(_snapshot_number,
                                             _fom_coordinates=_fom_coordinates, timesteps=timesteps, field=field,
                                             n=n)

    def perform_pod_space(self, _tol=1e-3, field="velocity"):
        """
        MODIFY
        """

        if field == "displacement":
            self.get_boundary_dofs()
            self.M_basis_space['displacement'] = self.M_basis_space['velocity'][self.M_bd_indexes]
            self.M_sv_space['displacement'] = self.M_sv_space['velocity']
            self.M_N_space['displacement'] = self.M_N_space['velocity']
        else:
            super().perform_pod_space(_tol=_tol, field=field)

        return

    def _bdf2_integration(self, vec):
        """
        MODIFY
        """

        Nt = len(vec)

        lhs_mat = (np.diagflat(np.ones(Nt)) +
                   self.M_bdf[0] * np.diagflat(np.ones(Nt-1), -1) +
                   self.M_bdf[1] * np.diagflat(np.ones(Nt-2), -2))
        rhs_vec = self.M_bdf_rhs * self.dt * vec

        new_vec = scipy.linalg.solve(lhs_mat, rhs_vec)

        return new_vec

    def perform_pod_time(self, _tol=1e-3, method="reduced", field="velocity"):
        """
        MODIFY
        """

        if field == "displacement":
            logger.info(f"Building reduced basis in time for {field}, by integration of velocity temporal reduced "
                        f"basis elements.")

            assert self.M_basis_time['velocity'].size > 0 and self.M_basis_time['pressure'].size > 0

            self.M_basis_time['displacement'] = np.array([self._bdf2_integration(basis)
                                                          for basis in self.M_basis_time['velocity'].T]).T
            logger.info(f"Dimension of displacement temporal reduced basis: {self.M_basis_time['displacement'].shape[1]}")

            self.M_sv_time['displacement'] = self.M_sv_time['velocity']
            self.M_N_time['displacement'] = self.M_basis_time['displacement'].shape[1]

        else:
            super().perform_pod_time(_tol=_tol, method=method, field=field)

        return

    def build_IC_basis_elements(self):
        """
        Build the temporal basis needed to impose the ICs
        """

        rbmstNS.RbManagerSpaceTimeNavierStokes.build_IC_basis_elements(self)

        self.M_basis_time_IC_int = np.array([self._bdf2_integration(basis)
                                             for basis in self.M_basis_time_IC])

        return

    def check_build_ST_basis(self):
        """
        MODIFY
        """

        return super().check_build_ST_basis() and \
               self.M_basis_space['displacement'].shape[0] > 0 and self.M_basis_time['displacement'].shape[0] > 0

    def save_ST_basis(self, which=None):
        """
        MODIFY
        """

        if which is None:
            which = {'velocity-space', 'velocity-time',
                     'pressure-space', 'pressure-time',
                     'displacement-space', 'displacement-time',
                     'lambda-space', 'lambda-time'}

        if self.M_save_offline_structures and which:
            logger.debug("Dumping ST bases to file ...")

            super().save_ST_basis(which=which)

            gen_utils.create_dir(os.path.join(self.M_basis_path, 'displacement'))

            if 'displacement-space' in which:
                self._save_bases('displacement', 'space')
            if 'displacement-time' in which:
                self._save_bases('displacement', 'time')

        return

    def import_basis_space_matrix(self, field="velocity"):
        """MODIFY
        """

        assert self.M_import_offline_structures, "Offline structures import is disabled"

        if field == "displacement":
            logger.info(f"Importing reduced basis in space for {field}")

            try:
                path = os.path.join(self.M_basis_path, field)
                self.M_basis_space['displacement'] = np.load(os.path.join(path, 'space_basis.npy'))
                self.M_sv_space['displacement'] = np.load(os.path.join(path, 'space_sv.npy'))
                if len(self.M_basis_space['displacement'].shape) == 1:
                    self.M_basis_space['displacement'] = np.expand_dims(self.M_basis_space['displacement'], axis=1)
                self.M_Nh['displacement'] = self.M_basis_space['displacement'].shape[0]
                self.M_N_space['displacement'] = self.M_basis_space['displacement'].shape[1]
                import_success = True
            except (IOError, OSError, FileNotFoundError) as e:
                logger.error(f"Error {e}: impossible to load the space basis matrix for {field}")
                import_success = False

        else:
            import_success = super().import_basis_space_matrix(field=field)

        return import_success

    def import_basis_time_matrix(self, field="velocity"):
        """MODIFY
        """

        assert self.M_import_offline_structures, "Offline structures import is disabled"

        if field == "displacement":
            logger.info(f"Importing reduced basis in time for {field}")

            try:
                path = os.path.join(self.M_basis_path, field)
                self.M_basis_time['displacement'] = np.load(os.path.join(path, 'time_basis.npy'))
                self.M_sv_time['displacement'] = np.load(os.path.join(path, 'time_sv.npy'))
                if len(self.M_basis_time['displacement'].shape) == 1:
                    self.M_basis_time['displacement'] = np.expand_dims(self.M_basis_time['displacement'], axis=1)
                self.M_Nt = self.M_basis_time['displacement'].shape[0]
                self.M_N_time['displacement'] = self.M_basis_time['displacement'].shape[1]
                import_success = True
            except (IOError, OSError, FileNotFoundError) as e:
                logger.error(f"Error {e}: impossible to load the space basis matrix for {field}")
                import_success = False

        else:
            import_success = super().import_basis_time_matrix(field=field)

        return import_success

    def import_snapshots_basis(self):
        """
        MODIFY
        """

        assert self.M_import_offline_structures, "Offline structures import is disabled"

        import_failures_basis = super().import_snapshots_basis()

        import_failures_basis.add('displacement-space' if not self.import_basis_space_matrix(field="displacement")
                                  else None)
        import_failures_basis.add('displacement-time' if not self.import_basis_time_matrix(field="displacement")
                                  else None)

        import_failures_basis = set(filter(None, import_failures_basis))

        if not ({'displacement-space', 'displacement-time'} & import_failures_basis):
            self.M_N['displacement'] = self.M_N_space['displacement'] * self.M_N_time['displacement']

        return import_failures_basis

    def import_FEM_structures(self, structures=None, matrixtype=csc_matrix):
        """
        MODIFY
        """

        if structures is None:
            structures = {'A', 'M', 'Bdiv', 'B', 'RHS', 'R', 'Abd', 'Mbd'}

        matrices = super().import_FEM_structures(structures=structures, matrixtype=matrixtype)

        try:
            if 'Abd' in structures:
                logger.debug("Importing boundary stiffness matrices...")
                matrices['Abd'] = []
                k = 0
                path = lambda cnt: os.path.join(self.M_fom_structures_path, f"A_bd_{cnt}")

                while os.path.isfile(path(k) + '.m') or os.path.isfile(path(k) + '.npz'):
                    matrices['Abd'].append(self._load_FEM_matrix_file(path(k), matrixtype=matrixtype,
                                                                      save=True, remove=True))
                    k += 1

                if k == 0:
                    raise FileNotFoundError("No membrane boundary matrices found!")

            if 'Mbd' in structures:
                logger.debug("Importing boundary mass matrix...")
                matrices['Mbd'] = self._load_FEM_matrix_file(os.path.join(self.M_fom_structures_path, "M_bd"),
                                                             matrixtype=matrixtype, save=True, remove=True)

                path = os.path.join(self.M_fom_structures_path, "M_bd_W")
                if os.path.isfile(path + '.m') or os.path.isfile(path + '.npz'):
                    matrices['MbdWall'] = self._load_FEM_matrix_file(path, matrixtype=matrixtype,
                                                                     save=True, remove=True)

            if 'd0' in structures:
                logger.debug("Importing the displacement initial condition (inferred from the first snapshot)...")
                fname = os.path.join(self.M_snapshots_path, 'param0', 'field2_IC.snap')
                matrices['d0'] = np.genfromtxt(fname, delimiter=',')

        except (IOError, OSError, FileNotFoundError) as e:
            raise ValueError(f"Error {e}: impossible to load the FEM matrices and FEM RHS")

        return matrices

    def reset_reduced_structures(self):
        """
        MODIFY
        """

        super().reset_reduced_structures()

        self.M_Abd_matrices = [np.zeros(0)] * 3
        self.M_Mbd_matrix = np.zeros(0)

        return

    def reset_rb_approximation(self):
        """
        MODIFY
        """

        super().reset_rb_approximation()

        self.M_N['displacement'] = 0
        self.M_N_space['displacement'] = 0
        self.M_N_time['displacement'] = 0

        self.M_basis_space['displacement'] = np.zeros(0)
        self.M_basis_time['displacement'] = np.zeros(0)

        return

    def assemble_reduced_structures(self, _space_projection='standard',
                                    _tolerances=None, N_components=None):
        """
        MODIFY
        """

        super().assemble_reduced_structures(_space_projection=_space_projection,
                                            _tolerances=_tolerances,
                                            N_components=N_components)

        FEM_matrices = self.import_FEM_structures(structures={'Abd', 'Mbd'})

        if _space_projection == 'natural':
            norms = [self.M_norm_matrices['velocity'], self.M_norm_matrices['pressure'],
                     self.M_norm_matrices['displacement']]
        elif _space_projection == 'standard':
            norms = [None, None, None]
        else:
            raise ValueError(f"Unrecognized space projection {_space_projection}")

        logger.debug("Projecting the boundary stiffness matrix")
        self.M_Abd_matrices = [np.zeros(0)] * len(FEM_matrices['Abd'])
        for (Abd_cnt, Abd_mat) in enumerate(FEM_matrices['Abd']):
            self.M_Abd_matrices[Abd_cnt] = self.project_matrix(Abd_mat[:, self.M_bd_indexes],  # only boundary DOFs!
                                                                [self.M_basis_space['velocity'],
                                                               self.M_basis_space['displacement']],
                                                               norm_matrix=norms[0])

        logger.debug("Projecting the boundary mass matrix")
        self.M_Mbd_matrix = self.project_matrix(FEM_matrices['Mbd'],
                                                [self.M_basis_space['velocity'],
                                                 self.M_basis_space['velocity']],
                                                norm_matrix=norms[0])
        if 'MbdWall' in FEM_matrices:
            self.M_MbdWall_matrix = self.project_matrix(FEM_matrices['MbdWall'][:, self.M_bd_indexes],
                                                        [self.M_basis_space['velocity'],
                                                         self.M_basis_space['displacement']],
                                                        norm_matrix=norms[0])

        logger.debug("Projecting the displacement initial condition")
        try:
            initial_condition = self.import_FEM_structures(structures={'d0'})
            self.M_u0['displacement'] = np.zeros((2, self.M_N_space['displacement']))
            for k in range(2):
                self.M_u0['displacement'][k] = self.project_vector(initial_condition['d0'][k][self.M_bd_indexes],  # only boundary DOFs!
                                                   self.M_basis_space['displacement'],
                                                   norm_matrix=None)
        except ValueError:
            logger.warning("Impossible to load the displacement initial condition. "
                           "Proceeding with homogeneous initial condition.")
            self.M_u0['displacement'] = np.zeros((2, self.M_N_space['displacement']))

        if self.M_save_offline_structures:
            gen_utils.create_dir(self.M_reduced_structures_path)

            for (Abd_cnt, Abd_mat) in enumerate(self.M_Abd_matrices):
                np.save(os.path.join(self.M_reduced_structures_path, f'Abd_rb_uu_{Abd_cnt}.npy'), Abd_mat)
            np.save(os.path.join(self.M_reduced_structures_path, 'Mbd_rb.npy'), self.M_Mbd_matrix)
            if 'MbdWall' in FEM_matrices:
                np.save(os.path.join(self.M_reduced_structures_path, 'MbdWall_rb.npy'), self.M_MbdWall_matrix)
            np.save(os.path.join(self.M_reduced_structures_path, 'd0_rb.npy'), self.M_u0['displacement'])

        return

    def import_reduced_structures(self, _tolerances=None, N_components=None,
                                  _space_projection='standard'):
        """
        MODIFY
        """

        import_success = super().import_reduced_structures(_tolerances=_tolerances,
                                                           N_components=N_components,
                                                           _space_projection=_space_projection)
        if not import_success:
            return False

        try:
            self.M_Abd_matrices = []

            k1 = 0
            path = os.path.join(self.M_reduced_structures_path, f'Abd_rb_uu_{k1}.npy')
            while os.path.isfile(path):
                self.M_Abd_matrices.append(np.load(path))
                k1 += 1
                path = os.path.join(self.M_reduced_structures_path, f'Abd_rb_uu_{k1}.npy')

            import_success = (k1 >= 1)

            self.M_Mbd_matrix = np.load(os.path.join(self.M_reduced_structures_path, 'Mbd_rb.npy'))
            if os.path.isfile(os.path.join(self.M_reduced_structures_path, 'MbdWall_rb.npy')):
                self.M_MbdWall_matrix = np.load(os.path.join(self.M_reduced_structures_path, 'MbdWall_rb.npy'))

            self.M_u0['displacement'] = np.load(os.path.join(self.M_reduced_structures_path, "d0_rb.npy"))

        except (OSError, FileNotFoundError) as e:
            logger.error(f"Error {e}: failed to import the reduced structures!")
            import_success = False

        if import_success:
            self.M_N_space['displacement'] = self.M_u0['displacement'].shape[1]

        return import_success

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
                elif parametrization == 'structure':
                    n_params = 4
                elif parametrization == 'clot':
                    n_params = None
                else:
                    raise ValueError("Invalid parametrization!")

                params_idxs_map[parametrization] = [n_params_cum,
                                                    n_params_cum + n_params if n_params is not None else None]
                n_params_cum += n_params

        return params_idxs_map

    def differentiate_parameters(self, _param, consider_extra=True, perform_rescaling=True):
        """
        MODIFY
        """

        params_map = super().differentiate_parameters(_param, consider_extra=consider_extra,
                                                      perform_rescaling=perform_rescaling)

        if 'structure' in self.M_parametrizations and perform_rescaling:
            param_structure = self._rescale_parameters_structure(params_map['structure'])
            cur_coeffs = self._compute_coefficients(param_structure)
            params_map['structure'] = cur_coeffs / self.M_default_coefficients

        return params_map

    def _rescale_parameters_structure(self, param):
        """
        MODIFY
        """
        return (1.0 + param) * self.M_param_ref

    def _compute_default_coefficients(self):
        """
        MODIFY
        """
        self.M_default_coefficients = self._compute_coefficients(self.M_param_ref)
        return

    @staticmethod
    def _compute_coefficients(param):
        """
        MODIFY  -> param = [rho_s, h_s, E, nu]
        """

        c1 = param[1] * param[0]  # h_s * rho_s
        c2 = param[1] * (param[2] / (2.0 * (1.0 + param[3])))  # h_s * lame_2
        c3 = param[1] * (param[2] * param[3]) / ((1.0 - param[3]) * (1.0 + param[3]))  # h_s * lame_1
        c4 = -1/6 * c1

        return np.array([c1, c2, c3, c4])

    def set_param_functions(self):
        """
        MODIFY
        """

        super().set_param_functions()

        if 'structure' in self.M_parametrizations:

            self.M_Blocks_param_affine_fun['structure'] = dict()
            if 'structure' in self.M_Blocks_param_affine.keys():
                for iB in self.M_Blocks_param_affine['structure']:
                    self.M_Blocks_param_affine_fun['structure'][iB] = [lambda mu, _k=k: mu[_k] for k in range(4)]

            self.M_f_Blocks_param_affine_fun['structure'] = dict()
            if 'structure' in self.M_f_Blocks_param_affine.keys():
                for ifB in self.M_f_Blocks_param_affine['structure']:
                    self.M_f_Blocks_param_affine_fun['structure'][ifB] = [lambda mu, _k=k: mu[_k] for k in range(4)]

        return

    def build_mu_independent_LHS(self):
        """
        Build a parameter-independent LHS and perform LU decomposition to speedup solve
        """

        self.build_rb_LHS(None, get_parametric=False)
        jac = np.copy(self.M_Block)

        if 'structure' in self.M_parametrizations:
            mat = sum([_fun(np.ones(4)) * _matrix
                       for _matrix, _fun in zip(self.M_Blocks_param_affine['structure'][0],
                                                self.M_Blocks_param_affine_fun['structure'][0])])
            jac[:self.M_N['velocity'], :self.M_N['velocity']] += mat

        self.M_Block_LU = scipy.linalg.lu_factor(jac)

        return

    def _sample_parameters_over_time(self, _param, n_cycles=1, m=-0.25, M=0.25):
        """
        MODIFY
        """

        kernel = RBF(length_scale=1.0, length_scale_bounds='fixed')

        _param_map = self.differentiate_parameters(_param, consider_extra=True, perform_rescaling=False)
        _param_NS = np.vstack([_param_map[k] for k in ['inflow', 'outflow', 'fluid', 'clot']
                               if k in self.M_parametrizations])
        new_params = super()._sample_parameters_over_time(_param_NS, n_cycles=n_cycles, m=m, M=M)

        if 'structure' in self.M_parametrizations:
            n_active_params = 0
            if 'inflow' in self.M_parametrizations:
                n_active_params += self.M_n_inflow_params + self.M_n_weak_inlets
            if 'outflow' in self.M_parametrizations:
                n_active_params += self.M_n_outflow_params + self.M_n_weak_outlets
            _new_params = self._sample_from_gpr(kernel, m, M, n_samples=4, n_cycles=n_cycles, seed=3, K=2)
            new_params = np.array([np.hstack([new_param[:n_active_params], _new_param, new_param[n_active_params:]])
                                   for (new_param, _new_param) in zip(new_params, _new_params)])

        return new_params

    def _reset_errors(self):
        """
        MODIFY
        """

        super()._reset_errors()

        self.M_relative_error['displacement'], self.M_relative_error['displacement-l2'] = np.zeros(self.M_Nt), 0.0
        self.M_relative_error['IG-displacement'], self.M_relative_error['IG-displacement-l2'] = np.zeros(self.M_Nt), 0.0

        return

    def _update_errors(self, N=None, is_IG=False):
        """
        MODIFY
        """

        if N is None:
            N = 1

        super()._update_errors(N=N, is_IG=is_IG)

        if not is_IG:
            self.M_relative_error['displacement'] += self.M_cur_errors['displacement'] / N
            self.M_relative_error['displacement-l2'] += self.M_cur_errors['displacement-l2'] / N
        else:
            self.M_relative_error['IG-displacement'] += self.M_cur_errors['displacement'] / N
            self.M_relative_error['IG-displacement-l2'] += self.M_cur_errors['displacement-l2'] / N

        return

    def get_field(self, _wn, field, n=None, reshape=False):
        """
        Get target field from solution
        """

        if field in {'velocity', 'pressure', 'lambda'}:
            vec = super().get_field(_wn, field, n=n, reshape=reshape)
        elif field == 'displacement':
            vec = _wn[:self.M_N['velocity']]
            if reshape:
                vec = np.reshape(vec, (self.M_N_space['velocity'], self.M_N_time['velocity']))
        else:
            raise ValueError(f"Unrecognized field {field}")

        return vec

    def reconstruct_fem_solution(self, _w, fields=None, indices_space=None, indices_time=None):
        """
        MODIFY
        """

        if fields is None:
            fields = ["velocity", "displacement", "pressure", "lambda"]

        super().reconstruct_fem_solution(_w, fields=fields,
                                         indices_space=indices_space, indices_time=indices_time)

        return

    def _save_solution(self, param_nb, save_path, is_test=False, n_cycles=None,
                       save_reduced=True, save_full=True, save_FOM=True, save_lambda=True):
        """
        MODIFY
        """

        super()._save_solution(param_nb, save_path, is_test=is_test, n_cycles=n_cycles,
                               save_reduced=save_reduced, save_full=save_full,
                               save_FOM=save_FOM, save_lambda=save_lambda)

        snapshots_path = self.M_test_snapshots_path if is_test else self.M_snapshots_path

        if save_FOM:
            snap_displacement = self._load_snapshot_file(os.path.join(snapshots_path, f'param{param_nb}'), 'displacement',
                                                         save=False, remove=False).T

            FEM_folder_name = os.path.join(save_path, 'FEM', 'Block0')

        STRB_folder_name = os.path.join(save_path, 'RB' + ('' if n_cycles is None else f'_T{n_cycles}'), 'Block0')

        if save_reduced:
            logger.warning("Computation of the reduced displacement from the FOM simulation not yet implemented.")
            np.save(os.path.join(STRB_folder_name, 'displacement_strb'), self.M_un[:self.M_N['velocity']])

        if save_full:

            if os.path.isdir(os.path.join(self.M_snapshots_path, 'Template')):
                gen_utils.write_field_to_h5(os.path.join(STRB_folder_name, 'Solution', 'block0.h5'),
                                            'displacement', self.M_utildeh['displacement'], field_dim=3)
                gen_utils.write_field_to_h5(os.path.join(STRB_folder_name, 'Solution', 'block0.h5'),
                                            'thickness', np.zeros_like(self.M_utildeh['pressure']))

                if save_FOM:
                    gen_utils.write_field_to_h5(os.path.join(FEM_folder_name, 'Solution', 'block0.h5'),
                                                'displacement', snap_displacement, field_dim=3)

            else:
                # reconstructed_displacement = np.zeros((self.M_Nh['velocity'], self.M_Nt))
                # reconstructed_displacement[self.M_bd_indexes] = self.M_utildeh['displacement']
                arr_utils.save_array(self.M_utildeh['displacement'].T, os.path.join(STRB_folder_name, 'displacement.txt'))

                if save_FOM:
                    arr_utils.save_array(snap_displacement.T, os.path.join(FEM_folder_name, 'displacement.txt'))

        return

    def _fill_errors(self, errors, data=None, is_IG=False):
        """
        MODIFY
        """

        data = super()._fill_errors(errors, data=data, is_IG=is_IG)

        data[("IG-" if is_IG else "") + "displacement"] = errors['displacement-l2']

        return data

    def _save_results_general(self):
        """
        MODIFY
        """

        super()._save_results_general()

        ERRORS_folder_name = os.path.join(self.M_results_path, 'mean_errors')
        np.savetxt(os.path.join(ERRORS_folder_name, 'displacement.txt'), self.M_relative_error['displacement'])

        return

    def compute_online_errors(self, param_nb, sol=None, is_test=False, ss_ratio=1):
        """
        Compute the errors (H1-norm for velocity, L2-norm for pressure, displacement, in both cases l2 in time) after
        the online phase for param 'param_nb'
        """

        if sol is not None:
            assert type(sol) is dict and {'displacement'}.issubset(sol.keys()), "Invalid solution format"
        else:
            sol = self.M_utildeh

        errors = super().compute_online_errors(param_nb, sol=sol, is_test=is_test, ss_ratio=ss_ratio)

        snapshots_path = self.M_test_snapshots_path if is_test else self.M_snapshots_path

        if sol['displacement'].shape[0] > self.M_bd_indexes.shape[0]:
            sol['displacement'] = sol['displacement'][self.M_bd_indexes]

        snap = self._load_snapshot_file(os.path.join(snapshots_path, f'param{param_nb}'), 'displacement',
                                        save=False, remove=False)[:self.M_Nt][ss_ratio-1::ss_ratio].T[self.M_bd_indexes]
        error_L2bd, denom_L2bd, error_L2bd_l2 = self._compute_error_field(sol['displacement'], snap, 'displacement',
                                                                          error_norm='L2bd')

        errors['displacement'] = np.array(error_L2bd)
        errors['displacement-ref'] = np.array(denom_L2bd)
        errors['displacement-l2'] = error_L2bd_l2

        return errors

    def postprocess_GS_errors(self, params, errors, param_names, folder):
        """ Draw plots and save files from GS-like analysis
        """

        super().postprocess_GS_errors(params, errors, param_names, folder)

        dt = self.dt
        times = np.linspace(dt, dt * self.M_Nt, self.M_Nt)

        import matplotlib.pyplot as plt
        plt.set_loglevel("warning")

        plt.figure()
        for (param, error) in zip(params, errors):
            labels = [f"{_param_name} = {_param}" for (_param_name, _param) in zip(param_names, param)]
            plt.semilogy(times, error['displacement'], label=' '.join(labels))
        plt.title(r'Displacement')
        plt.xlabel('Time (s)')
        plt.ylabel(r'$L^2$-norm error')
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(folder, 'displacement_error.eps'))

        return

    def check_dataset(self, _nsnap):
        raise NotImplementedError

    def check_offline_phase(self, _nsnap):
        raise NotImplementedError
