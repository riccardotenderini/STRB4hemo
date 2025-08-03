#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 09:48:05 2021
@author: Riccardo Tenderini
@mail: riccardo.tenderini@epfl.ch
"""

import numpy as np
import scipy
import os

import src.rb_library.rb_manager.space_time.Stokes.rb_manager_space_time_Stokes as rbmstS
import src.utils.general_utils as gen_utils
import time

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../../../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class RbManagerSRB_TFO_Stokes(rbmstS.RbManagerSpaceTimeStokes):
    """MODIFY
    """

    def __init__(self, _fom_problem, _affine_decomposition=None):
        """ MODIFY
        """

        super().__init__(_fom_problem, _affine_decomposition=_affine_decomposition)

        self.M_reduction_method = "SRB-TFO"

        return

    def import_snapshots_basis(self):
        """
        MODIFY
        """

        assert self.M_import_offline_structures, "Offline structures import is disabled"

        import_failures_basis = set()

        import_failures_basis.add('velocity-space' if not self.import_basis_space_matrix(field="velocity") else None)
        import_failures_basis.add('pressure-space' if not self.import_basis_space_matrix(field="pressure") else None)
        import_failures_basis.add('lambda-space' if not self.import_basis_space_matrix(field="lambda") else None)

        import_failures_basis = set(filter(None, import_failures_basis))

        if self.M_Nt == 0:
            _, self.M_Nt = self.M_fom_problem.time_specifics
        self.M_basis_time['velocity'] = None
        self.M_basis_time['pressure'] = None
        self.M_basis_time['lambda'] = [None] * self.M_n_coupling
        self.M_N_time['velocity'] = self.M_Nt
        self.M_N_time['pressure'] = self.M_Nt
        self.M_N_time['lambda'] = self.M_Nt * np.ones(self.M_n_coupling, dtype=int)

        if 'velocity-space' not in import_failures_basis:
            self.M_N['velocity'] = self.M_N_space['velocity'] * self.M_N_time['velocity']
        if 'pressure-space' not in import_failures_basis:
            self.M_N['pressure'] = self.M_N_space['pressure'] * self.M_N_time['pressure']
        if 'lambda-space' not in import_failures_basis:
            self.M_N['lambda'] = [self.M_N_space['lambda'][i] * self.M_N_time['lambda'][i] for i in range(self.M_n_coupling)]
            self.M_N_lambda_cumulative = np.cumsum(np.vstack([self.M_N['lambda'][n] for n in range(self.M_n_coupling)]))
            self.M_N_lambda_cumulative = np.insert(self.M_N_lambda_cumulative, 0, 0)

        return import_failures_basis

    def build_ST_basis(self, _tolerances, which=None):
        """
        MODIFY
        """

        if which is None:
            which = {'velocity-space', 'pressure-space', 'lambda-space'}

        if 'pressure-space' in which:
            self.perform_pod_space(_tolerances['pressure-space'], field="pressure")
            self.M_basis_time['pressure'] = None
            self.M_sv_time['pressure'] = None
            self.M_N_time['pressure'] = self.M_Nt
            self.M_N['pressure'] = self.M_N_space['pressure'] * self.M_N_time['pressure']
            logger.info('Finished pressure snapshots PODs \n')

        if 'lambda-space' in which:
            if _tolerances['lambda-space'] is not None:
                self.perform_pod_space(_tol=_tolerances['lambda-space'], field="lambda")
            else:
                self.M_N_space['lambda'] = self.M_Nh['lambda']
                self.M_basis_space['lambda'] = [np.eye(self.M_Nh['lambda'][n])
                                                for n in range(self.M_n_coupling)]
                self.M_sv_space['lambda'] = [np.ones(self.M_Nh['lambda'][n]) / self.M_Nh['lambda'][n]
                                             for n in range(self.M_n_coupling)]

            self.M_N_time['lambda'] = self.M_Nt * np.ones(self.M_n_coupling, dtype=int)
            self.M_basis_time['lambda'] = [None] * self.M_n_coupling

            self.M_N['lambda'] = self.M_N_space['lambda'] * self.M_Nt
            self.M_N_lambda_cumulative = np.cumsum(np.vstack([self.M_N['lambda'][n] for n in range(self.M_n_coupling)]))
            self.M_N_lambda_cumulative = np.insert(self.M_N_lambda_cumulative, 0, 0)
            logger.info('Finished Lagrange multipliers snapshots PODs \n')

        if 'velocity-space' in which:
            self.perform_pod_space(_tol=_tolerances['velocity-space'], field="velocity")
            self.primal_supremizers(stabilize=False)
            self.dual_supremizers(stabilize=False)
            self.M_basis_space['velocity'] = np.hstack((self.M_basis_space['velocity'], self.supr_primal, self.supr_dual))
            self.M_N_space['velocity'] += self.supr_primal.shape[1] + self.supr_dual.shape[1]
            self.M_basis_time['velocity'] = None
            self.M_sv_time['velocity'] = None
            self.M_N_time['velocity'] = self.M_Nt
            self.M_N['velocity'] = self.M_N_space['velocity'] * self.M_N_time['velocity']
            logger.info('Finished velocity snapshots PODs \n')

        if which:
            self.save_ST_basis(which=which)

        return

    def check_build_ST_basis(self):
        """
        MODIFY
        """

        return (self.M_basis_space['velocity'].shape[0] != 0 and
                self.M_basis_space['pressure'].shape[0] != 0 and
                all([self.M_basis_space['lambda'][n].shape[0] != 0 for n in range(self.M_n_coupling)]))

    def save_ST_basis(self, which=None):
        """
        MODIFY
        """

        if which is None:
            which = {'velocity-space', 'pressure-space', 'lambda-space'}

        if self.M_save_offline_structures:
            logger.debug("Dumping ST bases to file ...")

            gen_utils.create_dir(os.path.join(self.M_basis_path, 'velocity'))
            gen_utils.create_dir(os.path.join(self.M_basis_path, 'pressure'))

            if 'velocity-space' in which:
                self._save_bases('velocity', 'space')
            if 'pressure-space' in which:
                self._save_bases('pressure', 'space')
            if 'lambda-space' in which:
                for n in range(self.M_n_coupling):
                    cur_path = os.path.join(self.M_basis_path, os.path.normpath('lambda' + str(n) + '/'))
                    gen_utils.create_dir(cur_path)
                    self._save_bases('lambda', 'space', n=n)

        return

    def build_rb_nonparametric_LHS(self):
        raise NotImplementedError

    def build_rb_parametric_RHS(self, param):
        raise NotImplementedError

    def update_IC_terms(self, **kwargs):
        raise NotImplementedError

    def get_solution_field(self, field, n=0):
        """
        MODIFY
        """
        if field == "velocity":
            sol = np.reshape(self.M_un[:self.M_N['velocity']],
                             (self.M_N_time['velocity'], self.M_N_space['velocity'])).T
        elif field == "pressure":
            sol = np.reshape(self.M_un[self.M_N['velocity']:self.M_N['velocity'] + self.M_N['pressure']],
                             (self.M_N_time['pressure'], self.M_N_space['pressure'])).T
        elif field == "lambda":
            assert n < self.M_n_coupling, f"Invalid coupling index {n}"
            sol = np.reshape(self.M_un[self.M_N['velocity'] + self.M_N['pressure'] + self.M_N_lambda_cumulative[n]:
                                       self.M_N['velocity'] + self.M_N['pressure'] + self.M_N_lambda_cumulative[n + 1]],
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
            fields = {"velocity", "pressure", "lambda"}

        do_reshape = (len(_w.shape)) < 2

        if "velocity" in fields:
            times = slice(None) if (indices_time is None or 'velocity' not in indices_time.keys()) \
                else indices_time['velocity']
            spaces = np.arange(self.M_Nh['velocity']) if (indices_space is None or 'velocity' not in indices_space.keys()) \
                else indices_space['velocity']
            _un = np.reshape(_w[:self.M_N['velocity']], (self.M_Nt, self.M_N_space['velocity'])).T if do_reshape \
                  else _w[:, :self.M_N_space['velocity']].T
            self.M_utildeh['velocity'] = self.M_basis_space['velocity'][spaces].dot(_un)[:, times]

        if "pressure" in fields:
            times = slice(None) if (indices_time is None or 'pressure' not in indices_time.keys()) \
                else indices_time['pressure']
            spaces = np.arange(self.M_Nh['pressure']) if (indices_space is None or 'pressure' not in indices_space.keys()) \
                else indices_space['pressure']
            _pn = np.reshape(_w[self.M_N['velocity']:self.M_N['velocity']+ self.M_N['pressure']], (self.M_Nt, self.M_N_space['pressure'])).T if do_reshape \
                  else _w[:, self.M_N_space['velocity']: self.M_N_space['velocity'] + self.M_N_space['pressure']].T
            self.M_utildeh['pressure'] = self.M_basis_space['pressure'][spaces].dot(_pn)[:, times]

        if "lambda" in fields:
            lambdatildeh = [np.zeros(0)] * self.M_n_coupling
            for n in range(self.M_n_coupling):
                _ln = np.reshape(_w[self.M_N['velocity']+ self.M_N['pressure'] + self.M_N_lambda_cumulative[n]:
                                    self.M_N['velocity']+ self.M_N['pressure'] + self.M_N_lambda_cumulative[n+1]],
                                 (self.M_Nt, self.M_N_space['lambda'][n])).T if do_reshape  \
                      else _w[:, self.M_N_space['velocity'] + self.M_N_space['pressure'] + self.M_N_lambda_cumulative[n]//self.M_Nt:
                                 self.M_N_space['velocity'] + self.M_N_space['pressure'] + self.M_N_lambda_cumulative[n+1]//self.M_Nt].T
                lambdatildeh[n] = self.M_basis_space['lambda'][n].dot(_ln)
            self.M_utildeh['lambda'] = lambdatildeh

        return

    def _combine_old_solutions(self, sol):
        return -(self.M_bdf[0] * sol[-1] + self.M_bdf[1] * sol[-2])

    def _extrapolate_solution(self, sol):
        return self.M_extrap_coeffs[0] * sol[-1] + self.M_extrap_coeffs[1] * sol[-2]

    def _initialize_blocks(self):

        lhs_blocks = [np.zeros(0)] * 9

        lhs_blocks[3] = self.M_Bdiv_matrix
        lhs_blocks[4] = np.zeros((self.M_N_space['pressure'], self.M_N_space['pressure']))
        lhs_blocks[5] = np.zeros((self.M_N_space['pressure'], np.sum(self.M_N_space['lambda'])))
        lhs_blocks[6] = np.vstack([self.M_B_matrix[n] for n in range(self.M_n_coupling)])
        lhs_blocks[7] = np.zeros((np.sum(self.M_N_space['lambda']), self.M_N_space['pressure']))
        lhs_blocks[8] = np.zeros((np.sum(self.M_N_space['lambda']), np.sum(self.M_N_space['lambda'])))

        rhs_blocks = [np.zeros(0)] * 3
        rhs_blocks[1] = np.zeros(self.M_N_space['pressure'])

        return lhs_blocks, rhs_blocks

    def _update_LHS(self, lhs_blocks, param_map, force=False, update_T=False, update_params=False, get_blocks=False):

        dt = self.dt

        if update_T or force:
            lhs_blocks[0] = self.M_M_matrix + self.M_bdf_rhs * dt * self.M_A_matrix
            if self.M_has_resistance:
                # lhs_blocks[0] += self.M_bdf_rhs * self.dt * self.M_Radd_matrix   # not needed with implicit version
                lhs_blocks[0] += self.M_bdf_rhs * self.dt * self.M_R_matrix   # implicit version
            lhs_blocks[1] = self.M_bdf_rhs * dt * self.M_BdivT_matrix
            lhs_blocks[2] = self.M_bdf_rhs * dt * np.hstack([self.M_BT_matrix[n] for n in range(self.M_n_coupling)])

        if 'clot' in self.M_parametrizations:
            if update_params or force:
                Mclot_matrix = np.sum(_param * _matrix for _param, _matrix in zip(param_map['clot'],
                                                                                  self.M_Mclot_matrices))
            if update_params or update_T or force:
                lhs_blocks[0] += self.M_bdf_rhs * dt * Mclot_matrix

        if get_blocks:
            return lhs_blocks

        lhs_block = np.block([[lhs_blocks[0], lhs_blocks[1], lhs_blocks[2]],
                              [lhs_blocks[3], lhs_blocks[4], lhs_blocks[5]],
                              [lhs_blocks[6], lhs_blocks[7], lhs_blocks[8]]])

        return lhs_block

    def _update_RHS(self, rhs_blocks, sol, flow_rates, ind_t, get_blocks=False):

        u_old = self._combine_old_solutions(sol)[:self.M_N_space['velocity']]
        rhs_blocks[0] = self.M_M_matrix.dot(u_old)

        rhs_blocks[2] = np.hstack([self.M_RHS_vector[n] * flow_rates[ind_t, n]
                                   for n in range(self.M_n_coupling)])

        if get_blocks:
            return rhs_blocks

        rhs_block = np.hstack([rhs_blocks[k] for k in range(3)])

        return rhs_block

    def _solve(self, lhs_block, rhs_block, *args):
        return scipy.linalg.solve(lhs_block, rhs_block), True

    def _do_cycle_update(self, lhs_blocks, ind_t, param_trace, update_T=False, update_params=False):

        _, Nt = self.M_fom_problem.time_specifics

        self.M_fom_problem.M_fom_specifics['final_time'] = param_trace['T'][ind_t // Nt if update_T else 0]

        param = param_trace['param'][ind_t // Nt if (update_params or (ind_t == 0)) else 0]
        param_map = self.differentiate_parameters(param)

        logger.debug(f"Cycle {ind_t // Nt} - "
                     f"Value of T: {self.dt * Nt:.2e} - "
                     f"Value of parameters: {param}")

        lhs_block = self._update_LHS(lhs_blocks, param_map, force=(ind_t == 0), update_T=update_T,
                                     update_params=update_params)
        flow_rates = self.get_flow_rates(param)

        return lhs_block, flow_rates

    def _do_time_step(self, lhs_block, rhs_blocks, flow_rates, times, ind_t, sol):

        _, Nt = self.M_fom_problem.time_specifics

        times[ind_t] = (times[ind_t - 1] if ind_t > 0 else 0.0) + self.dt
        # logger.debug(f"Solving timestep {ind_t} - Time: {times[ind_t]:.4f} s")

        rhs_block = self._update_RHS(rhs_blocks, sol[-2:], flow_rates, ind_t % Nt)

        cur_sol, converged = self._solve(lhs_block, rhs_block, sol[-2:])
        if converged:
            sol.append(cur_sol)

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

        times = [0] * (n_cycles * Nt)
        for ind_t in range(n_cycles * Nt):

            if ind_t % Nt == 0:
                lhs_block, flow_rates = self._do_cycle_update(lhs_blocks, ind_t, param_trace,
                                                              update_T=update_T, update_params=update_params)

            converged = self._do_time_step(lhs_block, rhs_blocks, flow_rates, times, ind_t, sol)
            if not converged:
                logger.critical(f"Failed to compute the solution at timestep {ind_t}!")
                break

        if not converged:
            return None, times

        self.M_un = np.array(sol).T.flatten()

        if reconstruct_fem:
            self.reconstruct_fem_solution(np.array(sol)[2:])

        return sol, times

    def test_time_marching(self, param_nbs, is_test=False, ss_ratio=1, cycles_specifics=None, compute_errors=True):
        """
        Solve the S-reduced problem for param numbers in param_nbs
        """

        if cycles_specifics is None:
            cycles_specifics = {'n_cycles': 1,
                                'update_T': False,
                                'update_params': False}

        snapshots_path = self.M_test_snapshots_path if is_test else self.M_snapshots_path

        self._reset_errors()
        self.M_online_mean_time = 0.0

        solves_cnt = 0
        for param_nb in param_nbs:

            self.load_IC(param_nb, is_test=is_test, ss_ratio=ss_ratio)

            start = time.perf_counter()

            logger.info(f"Considering snapshot number {param_nb}")

            fname = os.path.join(snapshots_path, f'param{param_nb}', 'coeffile.txt')
            param = np.genfromtxt(fname, delimiter=',')

            logger.debug(f"Considering parameter values {param}")

            solution, times = self.time_marching(param, **cycles_specifics)

            if solution is None: continue  # solver failed

            solves_cnt += 1
            elapsed_time = time.perf_counter() - start
            logger.info(f"Online execution wall time: {elapsed_time :.4f} s")
            self.M_online_mean_time += elapsed_time

            results_dir = os.path.join(self.M_results_path, f'param{param_nb}' + ('_test' if is_test else ''))
            gen_utils.create_dir(results_dir)

            if cycles_specifics['n_cycles'] > 1:
                windows = times[len(times)//cycles_specifics['n_cycles']-1::len(times)//cycles_specifics['n_cycles']]
                _get_velocity = lambda x: np.array(x)[2:, :self.M_N_space['velocity']].T
                inflow_rates = [self.M_q_vectors['in'][k].dot(_get_velocity(solution))
                                for k in range(self.M_n_inlets)]
                outflow_rates = [self.M_q_vectors['out'][k].dot(_get_velocity(solution))
                                 for k in range(self.M_n_outlets)]

                self._save_flow_rates(inflow_rates, outflow_rates, times, windows, param_nb, is_test=is_test)

                if self.M_save_results:
                    self._save_solution(param_nb, results_dir, is_test=is_test, n_cycles=cycles_specifics['n_cycles'],
                                        save_reduced=False, save_full=True, save_FOM=False, save_lambda=False)

            if cycles_specifics['n_cycles'] == self.M_N_periods and compute_errors:
                logger.debug("Computing the errors")
                self.M_cur_errors = self.compute_online_errors(param_nb, is_test=is_test, ss_ratio=ss_ratio)
                self._update_errors()

                if cycles_specifics['n_cycles'] == 1:
                    self._save_results_snapshot(param_nb, self.M_cur_errors, is_test=is_test)
                else:
                    self._save_errors(self.M_cur_errors, results_dir)

        if solves_cnt > 0:
            self.M_relative_error = {key: self.M_relative_error[key] / solves_cnt for key in self.M_relative_error}

            self.M_online_mean_time /= solves_cnt
            print('\n')
            logger.info(f"Average online execution wall time: {self.M_online_mean_time:.4f} s")

            if cycles_specifics['n_cycles'] == 1:
                self._save_results_general()

        return


__all__ = [
    "RbManagerSRB_TFO_Stokes"
]
