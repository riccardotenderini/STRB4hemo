#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 09:48:32 2022
@author: Riccardo Tenderini
@mail: riccardo.tenderini@epfl.ch
"""

import sys
import os

sys.path.insert(0, os.path.normpath('../../../'))
sys.path.insert(0, os.path.normpath('../../'))
sys.path.insert(0, os.path.normpath('../'))

import examples.membrane.membrane_problem as mp

import src.rb_library.affine_decomposition.space_time.affine_decomposition_space_time_Stokes as adstS

import src.rb_library.rb_manager.space_time.Membrane.rb_manager_STRB_Membrane as rmstrbM
import src.rb_library.rb_manager.space_time.Membrane.rb_manager_STPGRB_Membrane as rmstpgrbM
import src.rb_library.rb_manager.space_time.Membrane.rb_manager_SRB_TFO_Membrane as rmsrbtfoM

from src.utils.parametric_flow_rate import InflowFactory

import examples.membrane.SpaceTime.config as config

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


def execute():
    ###################################################################################################################
    ############################################### INITIAL CHECKS ####################################################
    ###################################################################################################################

    # check that the problem is set to unsteady
    assert (config.problem_type == "Unsteady"), "The problem type must be set to 'Unsteady' to run this script"

    # check that the problem name and reduction method are correctly set
    assert config.problem_name == "Navier-Stokes_Membrane" and config.reduction_method in {'SRB-TFO',
                                                                                           'ST-RB', 'ST-PGRB'}, \
        ("The problem name must be set to 'Navier-Stokes_Membrane' and "
         "the reduction method must be selected in ['ST-RB', 'SRB-TFO'] to run this script")

    ###################################################################################################################
    ############################################### INITIALIZATION ####################################################
    ###################################################################################################################

    # define the fem problem
    my_nsmp = mp.NavierStokesMembraneProblem(None)
    my_nsmp.configure_fom(None, config.fom_specifics)

    # defining the affine decomposition structure
    my_affine_decomposition = adstS.AffineDecompositionHandlerSpaceTimeStokes()
    my_affine_decomposition.set_Q(1, 1, 1)  # number of affine terms
    my_affine_decomposition.set_timesteps(int(config.fom_specifics['number_of_time_instances']))

    # building the RB manager
    if config.reduction_method == "ST-RB":
        my_rb_manager = rmstrbM.RbManagerSTRBMembrane(my_nsmp, my_affine_decomposition)
    elif config.reduction_method == "SRB-TFO":
        my_rb_manager = rmsrbtfoM.RbManagerSRB_TFO_Membrane(my_nsmp, my_affine_decomposition)
    elif config.reduction_method == "ST-PGRB":
        my_rb_manager = rmstpgrbM.RbManagerSTPGRBMembrane(my_nsmp, my_affine_decomposition)
    else:
        raise ValueError(f"Unrecognized reduction method: {config.reduction_method}")

    my_rb_manager.set_parametrizations(config.parametrizations)
    my_rb_manager.use_LU = config.use_LU
    my_rb_manager.N_periods = config.N_periods
    my_rb_manager.Nt_IC = config.Nt_IC
    my_rb_manager.wall_elasticity = config.wall_elasticity

    # setting the newton's method specifics
    my_rb_manager.set_newton_specifics(config.newton_specs)

    my_rb_manager.set_import_and_save(config.IMPORT_SNAPSHOTS,
                                      config.IMPORT_OFFLINE_QUANTITIES,
                                      config.SAVE_OFFLINE_QUANTITIES,
                                      config.SAVE_RESULTS)

    paths = {'_snapshot_matrix': config.snapshots_path,
             '_basis_matrix': config.basis_path,
             '_affine_components': config.rb_blocks_path,
             '_fom_structures': config.fom_structures_path,
             '_reduced_structures': config.reduced_structures_path,
             '_generalized_coords': config.gen_coords_path,
             '_results': config.results_directory
             }
    if config.reduction_method == "ST-PGRB":
        paths['_used_norm'] = config.used_norm

    my_rb_manager.set_paths(**paths)

    ###################################################################################################################
    ############################################### CALL RB MANAGER ###################################################
    ###################################################################################################################

    specs = {'_tolerances': config.tolerances,
             '_n_weak_io': [config.n_weak_inlets, config.n_weak_outlets],
             '_mesh_name': config.mesh_name,
             '_N_components': config.N_components_NL_term,
             '_space_projection': config.projection_norm,
             'ss_ratio': config.time_subsample_ratio}
    if config.reduction_method == "ST-PGRB":
        specs['_used_norm'] = config.used_norm

    my_rb_manager.build_rb_approximation(config.n_snapshots, **specs)

    inflow_factory = InflowFactory()
    inflow_factory(my_rb_manager, config.flow_type, **config.flow_specs)

    # my_rb_manager.check_offline_phase(2)

    if config.reduction_method in {"ST-RB", "ST-PGRB"}:
        my_rb_manager.test_rb_solver(config.test_param_nbs,
                                     cycles_specifics=config.cycles_specifics, ss_ratio=config.time_subsample_ratio,
                                     compute_IG_error=config.compute_IG_error)
    elif config.reduction_method == "SRB-TFO":
        my_rb_manager.test_time_marching(config.test_param_nbs,
                                         cycles_specifics=config.cycles_specifics, ss_ratio=config.time_subsample_ratio)
    else:
        raise ValueError(f"Unrecognized reduction method {config.reduction_method}")

    return


if __name__ == "__main__":
    execute()
