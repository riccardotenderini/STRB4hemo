#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 10:33:16 2019
@author: Riccardo Tenderini
@mail: riccardo.tenderini@epfl.ch
"""

import sys
import os
sys.path.insert(0, '../../../..')
sys.path.insert(0, os.path.normpath('../../../'))
sys.path.insert(0, os.path.normpath('../../'))

import numpy as np
import os
import random
import time
try:
    from scipy.integrate import simpson
except ImportError:
    from scipy.integrate import simps as simpson

import src.utils.array_utils as arr_utils
import src.utils.general_utils as gen_utils
import src.rb_library.rb_manager.rb_manager as rbm

import logging.config
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class RbManagerUnsteady(rbm.RbManager):
    """Class which handles some of the routines employed during the offline phase of the RB approach for unsteady
    (parabolic) parametrized PDE problems, performing dimensionality reduction via the construction of a suitable
    Reduced Basis. Since more than one method to effectively solve parametrized unsteady problems in a dimensionality
    reduced framework is implemented in this project, this class configures as an abstract class, that defines the
    methods which are common to all the unsteady RB managers. In any case, the Reduced Basis is built by performing
    a POD over different FOM solutions, computed for different values of the characteristic parameters. Such solutions
    are furthermore computed interfacing to another library, which could be either the 'feamat' Matlab library
    (https://github.com/lucapegolotti/feamat/tree/pyorb_wrappers_parabolic) or the 'LifeV' C++ library
    (https://bitbucket.org/lifev-dev/lifev-release/wiki/Home) in the context of this project.
    It inherits from :class:`~rb_manager.RbManager`
    """

    def __init__(self, _fom_problem, _affine_decomposition=None):
        """Initialization of the RbManagerUnsteady class

        :param _fom_problem: unsteady FOM problem at hand
        :type _fom_problem: FomProblemUnsteady
        :param _affine_decomposition: AffineDecompositionHandlerUnsteady object, used to handle the affine decomposition
          of the unsteady parametric FOM problem at hand with respect to the characteristic parameters.
          Defaults to None.
        :type _affine_decomposition: AffineDecompositionHandlerUnsteady or NoneType
        """
        super().__init__(_fom_problem, _affine_decomposition)

        self.M_Nt = _fom_problem.time_dimension
        self.M_Mn = np.zeros((0, 0))
        self.M_step_number = None
        self.M_used_Qm = 0

        return

    @property
    def Nt(self):
        """Getter method, which allows to get the number of dofs of the FOM problem at hand

        :return: number of dofs of the FOM problem at hand
        :rtype: int
        """
        return self.M_Nt

    @property
    def step_number(self):
        """Getter method, which allows to get the number of steps of the multistep method

        :return: number of steps of the multistep method
        :rtype: int
        """
        return self.M_step_number

    @property
    def dt(self):
        """Getter method, which allows to get the timestep size

        :return: timestep size
        :rtype: float
        """
        return self.M_fom_problem.dt

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
            self.M_Nh = self.M_snapshots_matrix.shape[0]
            if self.M_snapshots_matrix.shape[1] == 0:
                self.M_ns = 0
            else:
                self.M_ns = int(self.M_snapshots_matrix.shape[1] / self.M_Nt)
            if _ns is not None and _ns < self.M_ns:
                self.M_snapshots_matrix = self.M_snapshots_matrix[:, :self.M_Nt * _ns]
                self.M_ns = _ns
            import_success = True
        except (IOError, OSError, FileNotFoundError, TypeError) as e:
            logger.error(f"Error {e}: impossible to load the snapshots matrix")
            import_success = False

        return import_success

    def import_test_snapshots_matrix(self, _ns=None):
        """Method which allows to import the FOM snapshots (i.e. FOM solutions to the problem at hand, corresponding
        to different parameter values) that are later used as test dataset in Deep Learning applications. The snapshots
        are imported from an _input_file, provided that it represents a valid path.

        :return: True if the importing has been successful, False otherwise
        :rtype: bool
        """

        assert self.M_import_snapshots, "Snapshots import disabled"

        try:
            if _ns is not None:
                self.M_test_snapshots_matrix = np.load(self.M_test_snapshots_path)[:, :self.M_Nt * _ns]
            else:
                self.M_test_snapshots_matrix = np.load(self.M_test_snapshots_path)

            if self.M_test_snapshots_matrix.shape[1] == 0:
                self.M_ns_test = 0
            else:
                self.M_ns_test = int(self.M_test_snapshots_matrix.shape[1] / self.M_Nt)
            import_success = True

        except (IOError, OSError, FileNotFoundError, TypeError) as e:
            logger.error(f"Error {e}: impossible to load the test snapshots matrix")
            import_success = False

        return import_success

    def get_snapshots_matrix(self, _fom_coordinates=np.array([]), timesteps=None):
        """Getter method, which returns the whole set of snapshots. If the input argument '_fom_coordinates' is not
        an empty array, the evaluation of the snapshots is restricted to the FOM dofs whose index is present in
        '_fom_coordinates'. If the input argument 'timesteps' is not None, the evaluation of the snapshots is performed
        just at the specified time instants. If no snapshot is stored, ir raises a ValueError.

        :param _fom_coordinates: index of the FOM dofs at which the evaluation of the snapshots is desired. If empty,
          the evaluation occurs at all the FOM dofs. Defaults to an empty numpy array
        :type _fom_coordinates: numpy.ndarray
        :param timesteps: indexes of the time instants at which the evaluation of the snapshots is desired. If None, the
           evaluation occurs at all time instants. Defaults to None
        :type timesteps: int or list or NoneType
        :return: matrix of the snapshots, eventually evaluated in a subset of the FOM dofs and of the time instants
        :rtype: numpy.ndarray
        """

        if timesteps is not None and type(timesteps) is int:
            timesteps = [timesteps]

        if not _fom_coordinates.shape[0] and timesteps is None:
            return self.M_snapshots_matrix
        elif _fom_coordinates.shape[0] and timesteps is None:
            return self.M_snapshots_matrix[_fom_coordinates.astype(int), :]
        elif not _fom_coordinates.shape[0] and timesteps is not None:
            indexes = [index for index in range(self.M_snapshots_matrix.shape[1]) if (index % self.M_Nt) in timesteps]
            return self.M_snapshots_matrix[:, indexes]
        else:
            indexes = [index for index in range(self.M_snapshots_matrix.shape[1]) if (index % self.M_Nt) in timesteps]
            return self.M_snapshots_matrix[_fom_coordinates.astype(int), indexes]

    def _get_snapshot(self, snapshots_matrix, _snapshot_number, _fom_coordinates=np.array([]), timesteps=None):
        """
        MODIFY
        """

        try:
            assert snapshots_matrix.shape[0] > 0
        except AssertionError:
            logger.critical("Impossible to get the snapshots. You need to construct or import them in advance")
            raise ValueError

        try:
            assert self.M_Nt is not None and self.M_Nt > 0
        except AssertionError:
            logger.critical("The number of time instances has not been defined!")
            raise ValueError

        if timesteps is not None and type(timesteps) is int:
            timesteps = [timesteps]

        if not _fom_coordinates.shape[0] and timesteps is None:
            return snapshots_matrix[:,
                                    _snapshot_number*self.M_Nt:(_snapshot_number+1)*self.M_Nt]
        elif _fom_coordinates.shape[0] and timesteps is None:
            return snapshots_matrix[_fom_coordinates.astype(int),
                                    _snapshot_number*self.M_Nt:(_snapshot_number+1)*self.M_Nt]
        elif not _fom_coordinates.shape[0] and timesteps is not None:
            indexes = [index for index in range(snapshots_matrix.shape[1])
                       if (index % self.M_Nt) in timesteps and (index // self.M_Nt) == _snapshot_number]
            return snapshots_matrix[:, indexes]
        else:
            indexes = [index for index in range(snapshots_matrix.shape[1])
                       if (index % self.M_Nt) in timesteps and (index // self.M_Nt) == _snapshot_number]
            return snapshots_matrix[np.ix_(_fom_coordinates.astype(int), indexes)]

    def get_snapshot(self, _snapshot_number, _fom_coordinates=np.array([]), timesteps=None, field=None):
        """Getter method, which returns the snapshot with index '_snapshot_number'. If the input argument
        '_fom_coordinates' is not an empty array, the evaluation of the snapshot is restricted to the FOM dofs
        whose index is present in '_fom_coordinates'. If the input argument 'timesteps' is not None, the evaluation of
        the snapshots is performed just at the specified time instants. If no snapshot is stored or if the given index
        is not valid, it raises a ValueError.

        :param _snapshot_number: index of the desired snapshot
        :type _snapshot_number: int
        :param _fom_coordinates: index of the FOM dofs at which the evaluation of the snapshot is desired. If
          empty, the evaluation occurs at all the FOM dofs. Defaults to an empty numpy array
        :type _fom_coordinates: numpy.ndarray
        :param timesteps: indexes of the time instants at which the evaluation of the snapshots is desired. If None, the
          evaluation occurs at all time instants. Defaults to None
        :type timesteps: int or list or NoneType
        :return: desired snapshot, eventually evaluated in a subset of the FOM dofs
        :rtype: numpy.ndarray
        """

        return self._get_snapshot(self.M_snapshots_matrix, _snapshot_number,
                                  _fom_coordinates=_fom_coordinates, timesteps=timesteps)

    def get_test_snapshot(self, _snapshot_number, _fom_coordinates=np.array([]), timesteps=None, field=None):
        """Getter method, which returns the test snapshot with index '_snapshot_number'. If the input argument
        '_fom_coordinates' is not an empty array, the evaluation of the test snapshot is restricted to the FOM dofs
        whose index is present in '_fom_coordinates'. If the input argument 'timesteps' is not None, the evaluation of
        the snapshots is performed just at the specified time instants. If no snapshot is stored or if the given index
        is not valid, it raises a ValueError.

        :param _snapshot_number: index of the desired test snapshot
        :type _snapshot_number: int
        :param _fom_coordinates: index of the FOM dofs at which the evaluation of the test snapshot is desired. If
          empty, the evaluation occurs at all the FOM dofs. Defaults to an empty numpy array
        :type _fom_coordinates: numpy.ndarray
        :param timesteps: indexes of the time instants at which the evaluation of the snapshots is desired. If None, the
          evaluation occurs at all time instants. Defaults to None
        :type timesteps: int or list or NoneType
        :return: desired test snapshot, eventually evaluated in a subset of the FOM dofs
        :rtype: numpy.ndarray
        """

        return self._get_snapshot(self.M_test_snapshots_matrix, _snapshot_number,
                                  _fom_coordinates=_fom_coordinates, timesteps=timesteps)

    def get_snapshot_function(self, _snapshot_number, _fom_coordinates=np.array([]), timesteps=None, field=None):
        """Getter method, which allows to get either the test or the train snapshot corresponding to the index
        '_snapshot_number', depending on the value of the class attribute flag self.M_get_test. It is basically a
        wrapper around :func:`~rb_manager.RbManager.get_snapshot` and :func:`~rb_manager.RbManager.get_test_snapshot`.
        Additionally, the snapshots can be evaluated in a subset of the dofs, if the input argument '_fom_coordinates'
        is not an empty numpy array, and in a subset of the time instances, if the input argument 'timesteps' is not
        None

        :param _snapshot_number: index of the desired (test) snapshot
        :type _snapshot_number: int
        :param _fom_coordinates: index of the FOM dofs at which the evaluation of the (test) snapshot is desired. If
          empty, the evaluation occurs at all the FOM dofs. Defaults to an empty numpy array
        :type _fom_coordinates: numpy.ndarray
        :param timesteps: indexes of the time instants at which the evaluation of the snapshots is desired. If None, the
          evaluation occurs at all time instants. Defaults to None
        :type timesteps: int or list or NoneType
        :return: desired (test) snapshot, eventually evaluated in a subset of the FOM dofs
        :rtype: numpy.ndarray
        """
        if self.M_get_test:
            return self.get_test_snapshot(_snapshot_number,
                                          _fom_coordinates=_fom_coordinates, timesteps=timesteps, field=field)
        else:
            return self.get_snapshot(_snapshot_number,
                                     _fom_coordinates=_fom_coordinates, timesteps=timesteps, field=field)

    def get_initial_condition(self, _param):
        """Method to get the initial condition of the problem at hand, for a given parameter value. If the FOM
        problem has not been correctly initialized as an unsteady one it raises an error. If the external engine is not
        the Matlab one, it raises a TypeError. It is just a wrapper around
        :func:`~fom_problem_unsteady.FomProblemUnsteady.get_initial_condition`

        :param _param: value of the parameter
        :type _param: numpy.ndarray
        :return: initial condition of the problem at hand if correctly initialized, else None
        :rtype: numpy.ndarray or NoneType
        """
        return self.M_fom_problem.get_initial_condition(_param)

    def get_rb_affine_matrix_A(self, _qA):
        """Getter method, which returns the affine component of index _qA among the FOM affine components of the
        stiffness matrix, projected over the RB-space. If _qA exceeds the number of affine components for the stiffness
        matrix an IndexError is raised.

        :param _qA: index of the desired affine component for the stiffness matrix
        :type _qA: int
        :return: desired RB affine component of the stiffness matrix or empty matrix if it is not available
        :rtype: numpy.ndarray
        """
        if self.M_N > 0:
            return self.M_affine_decomposition.get_rb_affine_matrix_A(_qA)
        else:
            logger.error("The reduced basis has not been computed. Impossible to return the RB affine matrix A")
            return np.zeros(())

    def get_rb_affine_matrix_M(self, _qM=0):
        """Getter method, which returns the affine component of index _qM among the FOM affine components of the
        mass matrix, projected over the RB-space. If _qM exceeds the number of affine components for the mass
        matrix an IndexError is raised.

        :param _qM: index of the desired affine component for the mass matrix. Defaults to 0
        :type _qM: int
        :return: desired RB affine component of the mass matrix or empty matrix if it is not available
        :rtype: numpy.ndarray
        """
        if self.M_N > 0:
            return self.M_affine_decomposition.get_rb_affine_matrix_M(_qM)
        else:
            logger.error("The reduced basis has not been computed. Impossible to return the RB affine matrix M")
            return np.zeros(())

    def get_rb_affine_vector(self, _qF, timestep=None):
        """Getter method, which returns the affine component of index _qF among the FOM affine components of the
        right-hand side, projected over the RB-space and evaluated taking the small timestep, used to solve the FOM
        problem, as reference timestep. If _qF exceeds the number of affine components for the right-hand side an
        IndexError is raised. Additionally, being the right-hand side time-dependent, the evaluation of such component
        can be restricted to a subset of time instances, if the input argument 'timestep' is not None.

        :param _qF: index of the desired affine component for the right-hand side vector
        :type _qF: int
        :param timestep: indexes of the time instants at which the evaluation of the RB components of the right-hand
          side is desired. If None, the evaluation occurs at all time instants. Defaults to None
        :type timestep: int or list or NoneType
        :return: desired FOM affine component of the right-hand side vector or empty vector if it is not available
        :rtype: numpy.ndarray
        """
        if self.M_N > 0:
            return self.M_affine_decomposition.get_rb_affine_vector(_qF, timestep=timestep)
        else:
            logger.error("The reduced basis has not been computed. Impossible to return the reduced affine vector f")
            return np.zeros(())

    def get_rb_affine_vector_reduced(self, _qF, timestep=None):
        """Getter method, which returns the affine component of index _qF among the FOM affine components of the
        right-hand side, projected over the RB-space and evaluated taking the dilated timestep (where the dilation
        factor has to be specified at problem configuration stage) as reference timestep. If _qF exceeds the number of
        affine components for the right-hand side an IndexError is raised. Additionally, being the right-hand side
        time-dependent, the evaluation of such component can be restricted to a subset of time instances, if the input
        argument 'timestep' is not None.

        :param _qF: index of the desired affine component for the right-hand side vector
        :type _qF: int
        :param timestep: indexes of the time instants at which the evaluation of the RB components of the right-hand
          side is desired. If None, the evaluation occurs at all time instants. Defaults to None
        :type timestep: int or list or NoneType
        :return: desired FOM affine component of the right-hand side vector or empty vector if it is not available
        :rtype: numpy.ndarray
        """
        if self.M_N > 0:
            return self.M_affine_decomposition.get_rb_affine_vector_reduced(_qF, timestep=timestep)
        else:
            logger.error("The reduced basis has not been computed. Impossible to return the RB affine vector f_reduced")
            return np.zeros(())

    def get_rb_affine_vector_space(self, _qF):
        """Getter method, which returns the spatial part of the affine component of index _qF among the FOM affine
        components of the right-hand side, projected over the RB-space; notice that this is possible under the
        assumption that the forcing term can be written as f(x,t) = f1(x)*f2(t)). If _qF exceeds the number of
        affine components for the right-hand side an IndexError is raised.

        :param _qF: index of the desired affine component for the right-hand side vector
        :type _qF: int
        :return: desired FOM affine component of the spatial part of the right-hand side vector or empty vector if it
           is not available
        :rtype: numpy.ndarray
        """
        if self.M_N > 0:
            return self.M_affine_decomposition.get_rb_affine_vector_space(_qF)
        else:
            logger.error("The reduced basis has not been computed. Impossible to return the RB affine vector f_space")
            return np.zeros(())

    def get_rb_initial_condition(self):
        """Getter method, which returns the reduced initial condition

        :return: reduced initial condition
        :rtype: numpy.ndarray
        """

        if self.M_N > 0:
            return self.M_affine_decomposition.get_rb_initial_condition()
        else:
            logger.error("The reduced basis has not been computed. Impossible to return the RB initial condition")
            return np.zeros(())

    def get_dt(self, reduced=False):
        """Getter method, which allows to get the value of the timestep used to solve the RB problem. In particular, if
        'reduced' is passed as False (default) the small timestep, used to solve the FOM problem, is returned;
        conversely, if 'reduced' is passed as True, the dilated timestep, effectively used to solve the RB problem, is
        returned.

        :param reduced: if True, the dilated timestep, effectively used to solve the RB problem, is returned. Otherwise,
          the small timestep used to solve the FOM problem, is returned. Defaults to False
        :type: bool
        :return: value of the desired timestep (either dilated or not)
        :rtype: float
        """
        timesteps_dilation = int((self.M_Nt - 1) / (self.M_affine_decomposition.M_Nt_Reduced - 1)) if reduced else 1
        return self.dt * timesteps_dilation

    def build_snapshots(self, _new_snapshots, seed=42, extend_existing=True, prob=None, is_test=False):
        """Method which allows to construct new FOM snapshots interfacing, via the FomProblemUnsteady class, to the
        external engine that has been chosen to perform the resolution of the  unsteady FOM problems (either the Matlab
        'feamat' library or the C++ 'LifeV' library). The number of snapshots is passed in input; if some snapshots are
        already stored, it is both possible to extend the existing snapshots with new ones, if the input
        'extend_existing' is passed as True, or otherwise recompute all the snapshots from scratch. Additionally, the
        parameters sampling for the snapshots computation follows a  discrete uniform distribution in the default case,
        but other discrete distributions can be passed as well, via the input argument 'prob'. Finally, if the class
        attribute flag self.M_save_offline_structures is True, the computed snapshots and parameters are saved as .txt
        files; the saving path is known to the class if the class method
        :func:`~rb_manager.RbManager.save_offline_structures` has been previously called, , with the input argument
        '_snapshot_matrix' defining such path.

        :param _new_snapshots: number of new desired snapshots
        :type: int
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
                cond_mat = (self.M_snapshots_matrix.size and len(self.M_snapshots_matrix.shape) == 2
                            if not is_test else
                            self.M_test_snapshots_matrix.size and len(self.M_test_snapshots_matrix.shape) == 2)
                cond_param = (self.M_offline_ns_parameters.size and len(self.M_offline_ns_parameters.shape) == 2
                              if not is_test else
                              self.M_test_offline_ns_parameters.size and len(self.M_test_offline_ns_parameters.shape)) == 2
                assert cond_mat and cond_param
            except AssertionError:
                extend_existing = False

        if extend_existing:
            current_snapshots_number = int(self.M_snapshots_matrix.shape[1] / self.M_Nt) if not is_test else \
                                       int(self.M_test_snapshots_matrix.shape[1] / self.M_Nt)
        else:
            current_snapshots_number = 0

        num_parameters = self.M_fom_problem.num_parameters
        params = np.zeros((_new_snapshots, num_parameters))
        snap_mat = np.zeros((self.M_Nh, _new_snapshots * self.M_Nt))

        if not self.M_fom_problem.is_linear():
            nl_term_mat = np.zeros_like(snap_mat)  # (Ns, ns x Nt)
            nl_term_jac_mat = []  # (ns x Nt, Nel, 3)

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
                nl_term_mat[:, iS * self.M_Nt:(iS + 1) * self.M_Nt] = nl_term.T
                nl_term_jac_mat.extend(nl_term_jac.tolist())

            norm_of_snapshot = np.linalg.norm(u, 'fro')
            logger.debug(f'Fröbenius norm of the snapshot is {norm_of_snapshot}')

            snap_mat[:, iS * self.M_Nt:(iS + 1) * self.M_Nt] = u

        if not is_test:
            self.M_snapshots_matrix = (snap_mat if not extend_existing else
                                       np.append(self.M_snapshots_matrix, snap_mat, axis=1))
            self.M_offline_ns_parameters = (params if not extend_existing else
                                            np.append(self.M_offline_ns_parameters, params, axis=0))
            self.M_ns = current_snapshots_number + _new_snapshots
            logger.info(f"Current snapshots number: {self.M_ns}")
            if not self.M_fom_problem.is_linear():
                self.M_nl_term_snapshot_matrix = nl_term_mat if not extend_existing else \
                                                 np.append(self.M_nl_term_snapshot_matrix, nl_term_mat, axis=1)
                self.M_nl_term_jac_snapshot_matrix = np.array(nl_term_jac_mat) if not extend_existing else \
                                                     np.append(self.M_nl_term_jac_snapshot_matrix,
                                                               np.swapaxes(np.array(nl_term_jac_mat), 0, 1), axis=1)
        else:
            self.M_test_snapshots_matrix = snap_mat if not extend_existing else \
                                           np.append(self.M_test_snapshots_matrix, snap_mat, axis=1)
            self.M_test_offline_ns_parameters = params if not extend_existing else \
                                                np.append(self.M_test_offline_ns_parameters, params, axis=0)
            self.M_ns_test = current_snapshots_number + _new_snapshots
            logger.info(f"Current test snapshots number: {self.M_ns_test}")

        self.M_Nh = snap_mat.shape[0]

        if not self.M_fom_problem.is_linear() and not extend_existing:
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

    def solve_reduced_problem(self, _param, _used_Qa=0, _used_Qf=0, _used_Qm=0):
        """Virtual method, lately used by all the implemented multistep methods to solve the unsteady reduced problem,
         i.e. the problem whose solution is a vector in the RB space, using the current multistep method.

        :param _param: current parameter value
        :type _param: numpy.ndarray
        :param _used_Qa: number of affine components that are used to assemble the stiffness matrix. If equal to 0, all
          the components are used; values different than 0 make sense only if the stiffness matrix is not affine
          decomposable, so that it is necessary to resort to an approximate affine decomposition via the MDEIM
          algorithm. Defaults to 0
        :type _used_Qa: int
        :param _used_Qf: number of affine components that are used to assemble the right-hand side vector. If equal to
          0, all the components are used; values different than 0 make sense only if the right-hand side vector is not
          affine decomposable, so that it is necessary to resort to an approximate affine decomposition via the DEIM
          algorithm. Defaults to 0
        :type _used_Qf: int
        :param _used_Qm: number of affine components that are used to assemble the mass matrix. If equal to
          0, all the components are used; values different than 0 make sense only if the mass matrix is not
          affine decomposable, so that it is necessary to resort to an approximate affine decomposition via the MDEIM
          algorithm. Defaults to 0
        :type _used_Qm: int
        """

        logger.critical("_solve_reduced_problem method cannot be called from the abstract RbManagerUnsteady class")
        raise SystemError

    def compute_rb_snapshots_error(self, _reduction_method, _snapshot_number=1,
                                   error_norms=None, test_flag=False, show=True):
        """Testing method, which evaluates the behavior of the RB solver onto the same snapshots that have been used to
        construct the Reduced Basis. In particular, it solves both the FOM and the RB problem for '_snapshot_number'
        snapshots with indexes chosen uniformly at random and computes the relative error between the two solutions in
        all the norms specified via the input argument 'norms'. Such procedure can be done either on the train snapshots
        or on the test snapshots, depending on the value of the 'test_flag' input argument. If the snapshot index is not
        valid it raises an IndexError.

        :param _reduction_method: model order reduction method, to be chosen among {'S-RB', 'ST-RB', 'ST-LSPG'}
        :type _reduction_method: str
        :param _snapshot_number: number of snapshots to be tested. Defaults to 1
        :type _snapshot_number: int
        :param error_norms: set containing the identificative strings of the vector norms that are wished to be used to
          evaluate the committed error. Admissible values are "l2", "L2", "H1" and "H10". If None, the L2-norm is
          used. Defaults to None
        :type error_norms: set or NoneType
        :param test_flag: if True, the test is performed over the test snapshots, otherwise it is performed on the
          train snapshots. Defaults to False.
        :param show: if True, the results are printed. Defaults to True.
        :type show: bool
        :return: dictionary whose keys are the available error norms and whose values are either the computed errors, if
          the corresponding norm has been selected, or numpy.nan, if such norm has not been selected
        :rtype: dict
        """
        goOn = True
        try:
            if not test_flag:
                assert len(self.M_snapshots_matrix.shape) and len(self.M_offline_ns_parameters.shape)
            else:
                assert len(self.M_test_snapshots_matrix.shape) and len(self.M_test_offline_ns_parameters.shape)
        except AssertionError:
            logger.warning("Impossible to compute the error over the pre-computed snapshots, "
                           "since no snapshot has been loaded")
            goOn = False

        error_l2 = 0.0
        error_L2 = 0.0
        error_H1 = 0.0
        error_H10 = 0.0
        num_converged_tests = 0

        random.seed(10)

        if goOn:

            if error_norms is None:
                error_norms = {'L2'}
            if _snapshot_number >= (self.M_ns if not test_flag else self.M_ns_test):
                snapshot_numbers = np.arange(self.M_ns if not test_flag else self.M_ns_test)
            elif _snapshot_number <= 0:
                logger.error(f"Invalid snapshot number {_snapshot_number} inserted. Returning 0.0 as default value here")
                return error_L2, error_H1, error_H10
            else:
                snapshot_numbers = random.sample(range(0, self.M_ns if not test_flag else self.M_ns_test),
                                                 _snapshot_number)

            for snapshot_index in snapshot_numbers:

                param = self.M_offline_ns_parameters[snapshot_index, :] if not test_flag \
                        else self.M_test_offline_ns_parameters[snapshot_index, :]
                
                logger.debug(f"SOLVING THE REDUCED PROBLEM FOR SNAPSHOT NUMBER {snapshot_index} - PARAMETER: {param}")
                self.solve_reduced_problem(param)
                if self.M_solver_converged:
                    num_converged_tests += 1
                    self.reconstruct_fem_solution(self.M_un)

                    logger.debug(f"RETRIEVING FOM SOLUTION FOR SNAPSHOT NUMBER {snapshot_index} - PARAMETER: {param}")
                    fom_sol = (self.M_snapshots_matrix[:, snapshot_index*self.M_Nt:(snapshot_index+1)*self.M_Nt] if not test_flag
                               else self.M_test_snapshots_matrix[:, snapshot_index*self.M_Nt:(snapshot_index+1)*self.M_Nt])
                    error = self.M_utildeh - fom_sol

                    errors, fom_sol_norm = self.compute_online_errors(error, fom_sol, _reduction_method,
                                                                      error_norms=error_norms, show=show)

                    if 'l2-l2' in error_norms:
                        error_l2 += (errors['l2-l2'] / fom_sol_norm['l2-l2'])
                    if 'L2-l2' in error_norms:
                        error_L2 += (errors['L2-l2'] / fom_sol_norm['L2-l2'])
                    if 'H1-l2' in error_norms:
                        error_H1 += (errors['H1-l2'] / fom_sol_norm['H1-l2'])
                    if 'H10-l2' in error_norms:
                        error_H10 += (errors['H10-l2'] / fom_sol_norm['H10-l2'])

            if num_converged_tests:
                error_l2 /= num_converged_tests
                error_L2 /= num_converged_tests
                error_H1 /= num_converged_tests
                error_H10 /= num_converged_tests

                if show:
                    print('\n')
                    logger.info("RB SOLVER CONVERGENCE")
                    logger.info(f"Number of tests where the ST-ROM solver converged: {num_converged_tests}/{_snapshot_number}")
                    logger.info(f"ST-ROM solver convergence rate: {num_converged_tests / _snapshot_number * 100} %")
                    print('\n')

                    logger.info("AVERAGE ERRORS")
                    logger.info("The average relative snapshots error in l2-L2 norm is %e" % error_l2)
                    logger.info("The average relative snapshots error in L2-L2 norm is %e" % error_L2)
                    logger.info("The average relative snapshots error in H1-L2 norm is %e" % error_H1)
                    logger.info("The average relative snapshots error in H10-L2 norm is %e" % error_H10)
                    print('\n\n')

        errors = dict()
        errors["l2"] = error_l2 if error_l2 > 0 else np.nan
        errors["L2"] = error_L2 if error_L2 > 0 else np.nan
        errors["H1"] = error_H1 if error_H1 > 0 else np.nan
        errors["H10"] = error_H10 if error_H10 > 0 else np.nan

        return errors

    def test_rb_solver(self, _n_test, _reduction_method,
                       _params=None, _noise=0.0, error_norms=None,
                       prob=None, show=True, make_plot=False):
        """Testing method, which evaluates the performance of the RB solver onto newly generated snapshots.
        Specifically, it either generates '_n_test' new parameter values (if '_params' is None) or takes the parameter
        values to be tested from the input argument '_params', for each of those it computes the solution of both the
        FOM problem and the RB problem, and finally it calculates the relative error in all the norms that are specified
        by the input argument 'error_norms'. As final result, the average errors (for each selected norm) are computed.
        Additionally, it is possible to add noise to the parameter values, by properly setting the input argument
        '_noise'; the added noise is a gaussian noise and the input argument identifies the SNR (Signal to Noise Ratio).
        Also, it is possible to evaluate the error at all time instants or just at the last one (via the input argument
        'all_times').

        :param _n_test: number of newly-generated parameters or of parameters to be considered among the input
          ones (if any)
        :type _n_test: int
        :param _reduction_method: model order reduction method, to be chosen among {'S-RB', 'ST-RB', 'ST-LSPG'}
        :type _reduction_method: str
        :param _params: if None, all parameters are generated from scratch. If True, it contains the parameters values
          to be tested; if it contains more than '_n_test' parameters, only the first '_n_test' are used; if it
          contains less than '_n_test' parameters, all parameters are tested, but no additional new parameter is
          computed. Defaults to None.
        :type: list[numpy.ndarray] or NoneType
        :param _noise: SNR of the noise which is added to the parameters. Defaults to 0.0
        :type _noise: float
        :param error_norms: set containing the identificative strings of the vector norms that are wished to be used to
          evaluate the committed error. Admissible values are "l2", "L2", "H1" and "H10". If None, the L2-norm is
          used. Defaults to None
        :type error_norms: set or NoneType
        :param prob: discrete probability distribution used for the random parameter sampling
        :type prob: list or tuple or numpy.ndarray
        :param show: if True, the results are printed. Defaults to True.
        :type show: bool
        :param make_plot: if True, the solutions are plotted and saved. Defaults to False.
        :type make_plot: bool
        :return: tuple of two dictionaries.

            * errors: dictionary whose keys are the available error norms and whose values are either the computed
              errors, if the corresponding norm has been selected, or numpy.nan, if such norm has not been selected
            * times: dictionary containing the average execution times of the FEM and RB problems and the corresponding
              speedup. Its keys are "execution_time_fem", "execution_time_rb" and "speedup"

        :rtype: tuple(dict, dict)
        """

        if error_norms is None:
            error_norms = {'L2'}
        execution_time_rb = 0
        execution_time_fem = 0
        speedup = 0
        num_converged_tests = 0

        timesteps_dilation = int(self.M_Nt / self.M_affine_decomposition.M_Nt_Reduced)

        error_l2 = 0.0
        error_L2 = 0.0
        error_H1 = 0.0
        error_H10 = 0.0

        if _params is None:
            _params = []
            for iP in range(_n_test):
                seed = 2019 + iP
                self.M_fom_problem.generate_parameter(prob=prob, seed=seed)
                _params.append(np.copy(self.M_fom_problem.param))

        for iP in range(_n_test):
            logger.info(f"Solving for parameter {iP}: {_params[iP]}")

            if _noise > 0:
                new_noised_param = _params[iP] * (1 + _noise * np.random.normal(0, 1, size=_params[iP].shape))
                logger.debug(f"New NOISED parameter {iP}: {new_noised_param}")
            else:
                new_noised_param = _params[iP]

            logger.debug("SOLVING THE REDUCED PROBLEM ...")
            start_rb = time.time()
            if self.M_step_number is not None:
                _, elapsed_time = self.solve_reduced_problem(new_noised_param)
            else:
                self.solve_reduced_problem(new_noised_param)
            end_rb = time.time()

            if self.M_solver_converged:
                if self.M_step_number is not None:
                    execution_time_rb += (end_rb - start_rb - elapsed_time)
                else:
                    execution_time_rb += (end_rb - start_rb)
                num_converged_tests += 1
                self.reconstruct_fem_solution(self.M_un)

                logger.debug("SOLVING THE FOM PROBLEM ...")
                uh, time_fem = self.M_fom_problem.solve_fom_problem(new_noised_param)
                if _reduction_method == "S-RB" and timesteps_dilation > 1:
                    uh = uh[:, timesteps_dilation-1::timesteps_dilation]
                execution_time_fem += time_fem

                speedup += (execution_time_fem / execution_time_rb)

                diff_shape = self.M_utildeh.shape[1] - uh.shape[1]
                error = self.M_utildeh[:, diff_shape:] - uh

                if self.M_save_results and self.M_results_path is not None:
                    logger.debug("SAVING THE RESULTS ...")
                    comm_path = os.path.join(self.M_results_path, f'test_param{iP}')
                    gen_utils.create_dir(comm_path)

                    RB_path = os.path.join(comm_path, 'RB.npy')
                    arr_utils.save_array(self.M_utildeh, RB_path)

                    FOM_path = os.path.join(comm_path, 'FOM.npy')
                    arr_utils.save_array(uh, FOM_path)

                    param_path = os.path.join(comm_path, 'param.txt')
                    arr_utils.save_array(new_noised_param, param_path)

                    if make_plot:
                        self.plot_fe_solution(self.M_utildeh, uh, folder=comm_path, name="Solution")

                errors, fom_sol_norm = self.compute_online_errors(error, uh, _reduction_method,
                                                                  error_norms=error_norms, show=show)

                if 'l2' in error_norms:
                    error_l2 += (errors['l2-l2'] / fom_sol_norm['l2-l2'])
                if 'L2' in error_norms:
                    error_L2 += (errors['L2-l2'] / fom_sol_norm['L2-l2'])
                if 'H1' in error_norms:
                    error_H1 += (errors['H1-l2'] / fom_sol_norm['H1-l2'])
                if 'H10' in error_norms:
                    error_H10 += (errors['H10-l2'] / fom_sol_norm['H10-l2'])

        if num_converged_tests:
            error_l2 /= num_converged_tests
            error_L2 /= num_converged_tests
            error_H1 /= num_converged_tests
            error_H10 /= num_converged_tests

            execution_time_fem /= num_converged_tests
            execution_time_rb /= num_converged_tests
            speedup /= num_converged_tests

            if show:
                print('\n')
                logger.info("RB SOLVER CONVERGENCE")
                logger.info(f"Number of tests where the ST-ROM solver converged: {num_converged_tests}/{_n_test}")
                logger.info(f"ST-ROM solver convergence rate: {num_converged_tests / _n_test * 100} %")
                print('\n')

                logger.info("AVERAGE ERRORS")
                logger.info("The average error in l2-L2 norm is %e" % error_l2)
                logger.info("The average error in L2-L2 norm is %e" % error_L2)
                logger.info("The average error in H1-L2 norm is %e" % error_H1)
                logger.info("The average error in H10-L2 norm is %e" % error_H10)
                print('\n')

                logger.info("AVERAGE EXECUTION TIMES AND SPEEDUP")
                logger.info("The average FEM execution time is %f s" % execution_time_fem)
                logger.info("The average RB  execution time is %f s" % execution_time_rb)
                logger.info("The average speedup FEM/RB is %f " % speedup)
                print('\n\n')

        errors = dict([])
        errors['error_l2'] = error_l2 if error_l2 > 0 else np.nan
        errors['error_L2'] = error_L2 if error_L2 > 0 else np.nan
        errors['error_H1'] = error_H1 if error_H1 > 0 else np.nan
        errors['error_H10'] = error_H10 if error_H10 > 0 else np.nan

        times = dict([])
        times['execution_time_fem'] = execution_time_fem
        times['execution_time_rb'] = execution_time_rb
        times['speedup'] = speedup

        return errors, times

    def compute_online_errors(self, error, uh, _reduction_method,
                              error_norms=None, show=False):
        """Method to compute the errors in the online phase of the RB methods

        :param error:
        :type error:
        :param uh:
        :type uh:
        :param _reduction_method:
        :type _reduction_method:
        :param error_norms:
        :type error_norms:
        :param show:
        :type show:
        """

        if error_norms is None:
            error_norms = {'L2'}

        dt = self.get_dt(reduced=True)
        T = self.M_fom_problem.M_fom_specifics['final_time']
        times = np.arange(dt, T + dt, dt)

        if _reduction_method == "S-RB":
            error_l2 = np.zeros(int(self.M_affine_decomposition.M_Nt_Reduced))
            error_L2 = np.zeros(int(self.M_affine_decomposition.M_Nt_Reduced))
            error_H1 = np.zeros(int(self.M_affine_decomposition.M_Nt_Reduced))
            error_H10 = np.zeros(int(self.M_affine_decomposition.M_Nt_Reduced))
            fom_sol_l2 = np.zeros(int(self.M_affine_decomposition.M_Nt_Reduced))
            fom_sol_L2 = np.zeros(int(self.M_affine_decomposition.M_Nt_Reduced))
            fom_sol_H1 = np.zeros(int(self.M_affine_decomposition.M_Nt_Reduced))
            fom_sol_H10 = np.zeros(int(self.M_affine_decomposition.M_Nt_Reduced))
        elif _reduction_method in {"ST-RB", "ST-LSPG"}:
            error_l2 = np.zeros(int(self.M_affine_decomposition.M_Nt))
            error_L2 = np.zeros(int(self.M_affine_decomposition.M_Nt))
            error_H1 = np.zeros(int(self.M_affine_decomposition.M_Nt))
            error_H10 = np.zeros(int(self.M_affine_decomposition.M_Nt))
            fom_sol_l2 = np.zeros(int(self.M_affine_decomposition.M_Nt))
            fom_sol_L2 = np.zeros(int(self.M_affine_decomposition.M_Nt))
            fom_sol_H1 = np.zeros(int(self.M_affine_decomposition.M_Nt))
            fom_sol_H10 = np.zeros(int(self.M_affine_decomposition.M_Nt))

        errors = dict([])
        fom_sol = dict([])

        for i in range(error.shape[1]):

            norms_of_error = self.compute_norm(error[:, i], norm_types=error_norms)
            norms_of_fom_sol = self.compute_norm(uh[:, i], norm_types=error_norms)

            if 'l2' in error_norms:
                error_l2[i] = norms_of_error["l2"]
                fom_sol_l2[i] = norms_of_fom_sol["l2"]

            if 'L2' in error_norms:
                error_L2[i] = norms_of_error["L2"]
                fom_sol_L2[i] = norms_of_fom_sol["L2"]

            if 'H10' in error_norms:
                error_H10[i] = norms_of_error["H10"]
                fom_sol_H10[i] = norms_of_fom_sol["H10"]

            if 'H1' in error_norms:
                error_H1[i] = norms_of_error["H1"]
                fom_sol_H1[i] = norms_of_fom_sol["H1"]

        if 'l2' in error_norms and show:
            error_l2_L2 = np.sqrt(simpson(error_l2 ** 2, times))
            fom_sol_l2_L2 = np.sqrt(simpson(fom_sol_l2 ** 2, times))
            logger.debug("The relative error in l2-L2 norm is %e \n" % (error_l2_L2 / fom_sol_l2_L2))
            errors['l2'], errors['l2-l2'] = error_l2, error_l2_L2
            fom_sol['l2'], fom_sol['l2-l2'] = fom_sol_l2, fom_sol_l2_L2

        if 'L2' in error_norms and show:
            error_L2_L2 = np.sqrt(simpson(error_L2 ** 2, times))
            fom_sol_L2_L2 = np.sqrt(simpson(fom_sol_L2 ** 2, times))
            logger.debug("The relative error in L2-L2 norm is %e \n" % (error_L2_L2 / fom_sol_L2_L2))
            errors['L2'], errors['L2-l2'] = error_l2, error_L2_L2
            fom_sol['L2'], fom_sol['L2-l2'] = fom_sol_l2, fom_sol_L2_L2

        if 'H1' in error_norms and show:
            error_H1_L2 = np.sqrt(simpson(error_H1 ** 2, times))
            fom_sol_H1_L2 = np.sqrt(simpson(fom_sol_H1 ** 2, times))
            logger.debug("The relative error in H1-L2 norm is %e \n" % (error_H1_L2 / fom_sol_H1_L2))
            errors['H1'], errors['H1-l2'] = error_l2, error_H1_L2
            fom_sol['H1'], fom_sol['H1-l2'] = fom_sol_l2, fom_sol_H1_L2

        if 'H10' in error_norms and show:
            error_H10_L2 = np.sqrt(simpson(error_H10 ** 2, times))
            fom_sol_H10_L2 = np.sqrt(simpson(fom_sol_H10 ** 2, times))
            logger.debug("The relative error in H10-L2 norm is %e \n" % (error_H10_L2 / fom_sol_H10_L2))
            errors['H10'], errors['H10-l2'] = error_l2, error_H10_L2
            fom_sol['H10'], fom_sol['H10-l2'] = fom_sol_l2, fom_sol_H10_L2

        return errors, fom_sol


__all__ = [
    "RbManagerUnsteady"
]
