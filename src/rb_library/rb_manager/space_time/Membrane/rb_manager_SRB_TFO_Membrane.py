#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 17:59:31 2022
@author: Riccardo Tenderini
@mail: riccardo.tenderini@epfl.ch
"""

import os
import numpy as np

import src.utils.general_utils as gen_utils
from src.utils.array_utils import is_empty

import src.rb_library.rb_manager.space_time.Navier_Stokes.rb_manager_SRB_TFO_Navier_Stokes as rbmsrbtfoNS
import src.rb_library.rb_manager.space_time.Membrane.rb_manager_space_time_Membrane as rbmstM

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../../../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class RbManagerSRB_TFO_Membrane(rbmsrbtfoNS.RbManagerSRB_TFO_NavierStokes,
                                rbmstM.RbManagerSpaceTimeMembrane):
    """MODIFY
    """

    def __init__(self, _fom_problem, _affine_decomposition=None):
        """
        MODIFY
        """

        super().__init__(_fom_problem, _affine_decomposition=_affine_decomposition)

        self.M_As_matrix = np.zeros(0)
        self.M_Ms_matrix = np.zeros(0)

        return

    def import_snapshots_basis(self):
        """
        MODIFY
        """

        import_failures_basis = rbmsrbtfoNS.RbManagerSRB_TFO_NavierStokes.import_snapshots_basis(self)
        import_failures_basis.add('displacement-space' if not self.import_basis_space_matrix(field="displacement")
                                  else None)
        import_failures_basis = set(filter(None, import_failures_basis))

        self.M_basis_time['displacement'] = None
        self.M_N_time['displacement'] = self.M_Nt

        if 'displacement-space' not in import_failures_basis:
            self.M_N['displacement'] = self.M_N_space['displacement'] * self.M_N_time['displacement']

        return import_failures_basis

    def build_ST_basis(self, _tolerances, which=None):
        """
        MODIFY
        """

        if which is None:
            which = {'velocity-space', 'pressure-space',
                     'displacement-space', 'lambda-space'}

        which_d = set()
        if 'displacement-space' in which:
            which_d.add('displacement-space')
            which.remove('displacement-space')

        rbmsrbtfoNS.RbManagerSRB_TFO_NavierStokes.build_ST_basis(self, _tolerances, which=which)

        if 'displacement-space' in which_d:
            self.perform_pod_space(field="displacement")
            self.M_basis_time['displacement'] = None
            self.M_sv_time['displacement'] = None
            self.M_N_time['displacement'] = self.M_Nt
            self.M_N['displacement'] = self.M_N_space['displacement'] * self.M_N_time['displacement']
            logger.info('Finished displacement snapshots PODs \n')
            self.save_ST_basis(which={'displacement-space'})

        return

    def check_build_ST_basis(self):
        """
        MODIFY
        """

        NS_basis = rbmsrbtfoNS.RbManagerSRB_TFO_NavierStokes.check_build_ST_basis(self)

        return NS_basis and self.M_basis_space['displacement'].shape[0] > 0

    def save_ST_basis(self, which=None):
        """
        MODIFY
        """

        if which is None:
            which = {'velocity-space', 'pressure-space', 'lambda-space', 'displacement-space'}

        rbmsrbtfoNS.RbManagerSRB_TFO_NavierStokes.save_ST_basis(self, which=which)

        if self.M_save_offline_structures:
            gen_utils.create_dir(os.path.join(self.M_basis_path, 'displacement'))

            if 'displacement-space' in which:
                self._save_bases('displacement', 'space')

        return

    def get_solution_field(self, field, n=0):
        """
        MODIFY
        """
        if field == "velocity":
            sol = np.reshape(self.M_un[:self.M_N['velocity']],
                             (self.M_N_time['velocity'], self.M_N_space['velocity'])).T
        elif field == "displacement":
            sol = np.reshape(self.M_un[self.M_N['velocity']:self.M_N['velocity']+self.M_N['displacement']],
                             (self.M_N_time['displacement'], self.M_N_space['displacement'])).T
        elif field == "pressure":
            sol = np.reshape(self.M_un[self.M_N['velocity'] + self.M_N['displacement']:self.M_N['velocity'] + self.M_N['displacement'] + self.M_N['pressure']],
                             (self.M_N_time['pressure'], self.M_N_space['pressure'])).T
        elif field == "lambda":
            assert n < self.M_n_coupling, f"Invalid coupling index {n}"
            sol = np.reshape(self.M_un[self.M_N['velocity'] + self.M_N['displacement'] + self.M_N['pressure'] + self.M_N_lambda_cumulative[n]:
                                       self.M_N['velocity'] + self.M_N['displacement'] + self.M_N['pressure'] + self.M_N_lambda_cumulative[n+1]],
                             (self.M_N_time['lambda'][n], self.M_N_space['lambda'][n])).T
        else:
            raise ValueError(f"Unrecognized field {field}")

        return sol

    def reconstruct_fem_solution(self, _w, fields=None, indices_space=None, indices_time=None):
        """
        MODIFY
        """

        if self.M_utildeh:
            logger.warning("FEM solution is already available!")
            return

        if fields is None:
            fields = {"velocity", "displacement", "pressure", "lambda"}

        do_reshape = (len(_w.shape)) < 2

        _w_ns = np.delete(_w, np.arange(self.M_N['velocity'], self.M_N['velocity'] + self.M_N['displacement']), 0) if do_reshape \
                else np.delete(_w, np.arange(self.M_N_space['velocity'], self.M_N_space['velocity'] + self.M_N_space['displacement']), 1)
        rbmsrbtfoNS.RbManagerSRB_TFO_NavierStokes.reconstruct_fem_solution(self, _w_ns,
                                                                           fields=fields,
                                                                           indices_space=indices_space,
                                                                           indices_time=indices_time)

        if "displacement" in fields:
            times = slice(None) if (indices_time is None or 'displacement' not in indices_time.keys()) \
                else indices_time['displacement']
            spaces = np.arange(self.M_Nh['displacement']) if (indices_space is None or 'displacement' not in indices_space.keys()) \
                else indices_space['displacement']
            _dn = np.reshape(_w[self.M_N['velocity']:self.M_N['velocity'] + self.M_N['displacement']], (self.M_Nt, self.M_N_space['displacement'])).T if do_reshape \
                  else _w[:, self.M_N_space['velocity']: self.M_N_space['velocity'] + self.M_N_space['displacement']].T
            self.M_utildeh['displacement'] = self.M_basis_space['displacement'][spaces].dot(_dn)[:, times]

        return

    def _update_LHS(self, lhs_blocks, param_map, force=False, update_T=False, update_params=False, get_blocks=False):

        lhs_blocks = super()._update_LHS(lhs_blocks, param_map, force=force, update_T=update_T,
                                         update_params=update_params, get_blocks=True)

        if update_params or force:
            self.M_As_matrix = sum([param_map['structure'][i + 1] * self.M_Abd_matrices[i] for i in range(3)])
            self.M_Ms_matrix = param_map['structure'][0] * self.M_Mbd_matrix

        if update_params or update_T or force:
            lhs_blocks[0] += (self.M_Ms_matrix + (self.M_bdf_rhs * self.dt) ** 2 * self.M_As_matrix)

        if update_T or force:
            if self.M_wall_elasticity > 0 and not is_empty(self.M_MbdWall_matrix):
                lhs_blocks[0] += (self.M_bdf_rhs * self.dt) ** 2 * self.M_wall_elasticity * self.M_MbdWall_matrix

        if get_blocks:
            return lhs_blocks

        lhs_block = np.block([[lhs_blocks[0], lhs_blocks[1], lhs_blocks[2]],
                              [lhs_blocks[3], lhs_blocks[4], lhs_blocks[5]],
                              [lhs_blocks[6], lhs_blocks[7], lhs_blocks[8]]])

        return lhs_block

    def _update_RHS(self, rhs_blocks, sol, flow_rates, ind_t, get_blocks=False, sol_d=None):

        assert sol_d is not None, "The past displacement solutions must be provided!"

        rhs_blocks = super()._update_RHS(rhs_blocks, sol, flow_rates, ind_t, get_blocks=True)

        u_old = self._combine_old_solutions(sol)[:self.M_N_space['velocity']]
        rhs_blocks[0] += self.M_Ms_matrix.dot(u_old)

        d_old = self._combine_old_solutions(sol_d)[:self.M_N_space['velocity']]
        rhs_blocks[0] -= self.M_bdf_rhs * self.dt * self.M_As_matrix.dot(d_old)

        if self.M_wall_elasticity > 0 and not is_empty(self.M_MbdWall_matrix):
            rhs_blocks[0] -= self.M_bdf_rhs * self.dt * self.M_wall_elasticity * self.M_MbdWall_matrix.dot(d_old)

        if get_blocks:
            return rhs_blocks

        rhs_block = np.hstack([rhs_blocks[k] for k in range(3)])

        return rhs_block

    def _do_time_step(self, lhs_block, rhs_blocks, flow_rates, times, ind_t, sol, sol_d=None):
        """
        MODIFY
        """

        assert sol_d is not None, "The past displacement solutions must be provided!"

        _, Nt = self.M_fom_problem.time_specifics

        times[ind_t] = (times[ind_t - 1] if ind_t > 0 else 0.0) + self.dt
        # logger.debug(f"Solving timestep {ind_t} - Time: {times[ind_t]:.4f} s")

        rhs_block = self._update_RHS(rhs_blocks, sol[-2:], flow_rates, ind_t % Nt, sol_d=sol_d[-2:])

        cur_sol, converged = self._solve(lhs_block, rhs_block, sol[-2:])
        if converged:
            sol.append(cur_sol)

            u_bd = sol[-1][:self.M_N_space['velocity']]
            sol_d.append(self.M_bdf_rhs * self.dt * u_bd + self._combine_old_solutions(sol_d))

        return converged

    def time_marching(self, param, reconstruct_fem=True, n_cycles=1, update_T=False, update_params=False):
        """
        MODIFY
        """

        _, Nt = self.M_fom_problem.time_specifics

        param_trace = self.get_parameters_trace(param, n_cycles=n_cycles,
                                                update_T=update_T, update_params=update_params)

        lhs_blocks, rhs_blocks = self._initialize_blocks()

        sol = [np.pad(self.M_u0['velocity'][0], (0, self.M_N_space['pressure'] + np.sum(self.M_N_space['lambda']))),
               np.pad(self.M_u0['velocity'][1], (0, self.M_N_space['pressure'] + np.sum(self.M_N_space['lambda'])))]
        sol_d = [self.M_u0['displacement'][0], self.M_u0['displacement'][1]]

        times = [0] * (n_cycles * Nt)
        for ind_t in range(n_cycles * Nt):

            if ind_t % Nt == 0:
                lhs_block, flow_rates = self._do_cycle_update(lhs_blocks, ind_t, param_trace,
                                                              update_T=update_T, update_params=update_params)

            converged = self._do_time_step(lhs_block, rhs_blocks, flow_rates, times, ind_t, sol, sol_d=sol_d)
            if not converged:
                logger.critical(f"Failed to compute the solution at timestep {ind_t}!")
                break

        if not converged:
            return None, times

        sol_full = np.hstack([np.array(sol)[:, :self.M_N_space['velocity']],
                              np.array(sol_d),
                              np.array(sol)[:, self.M_N_space['velocity']:]])
        self.M_un = sol_full.flatten()

        if reconstruct_fem:
            self.reconstruct_fem_solution(np.array(sol_full)[2:])

        return sol, times

    def test_time_marching(self, param_nbs, is_test=False, ss_ratio=1, cycles_specifics=None, compute_errors=True):
        """
        Solve the S-reduced problem for param numbers in param_nbs
        """

        return rbmsrbtfoNS.RbManagerSRB_TFO_NavierStokes.test_time_marching(self, param_nbs,
                                                                            is_test=is_test,
                                                                            ss_ratio=ss_ratio,
                                                                            cycles_specifics=cycles_specifics,
                                                                            compute_errors=compute_errors)
