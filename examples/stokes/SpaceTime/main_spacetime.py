#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 11:16:23 2020
@author: Riccardo Tenderini
@mail: riccardo.tenderini@epfl.ch
"""

import sys
import os
sys.path.insert(0, os.path.normpath('../../../'))
sys.path.insert(0, os.path.normpath('../../'))
sys.path.insert(0, os.path.normpath('../'))

import examples.stokes.stokes_problem as sp

import src.rb_library.affine_decomposition.space_time.affine_decomposition_space_time_Stokes as adstS

import src.rb_library.rb_manager.space_time.Stokes.rb_manager_STRB_Stokes as rmstrbS
import src.rb_library.rb_manager.space_time.Stokes.rb_manager_SRB_TFO_Stokes as rmsrbtfoS
import src.rb_library.rb_manager.space_time.Stokes.rb_manager_STPGRB_Stokes as rmstpgS

from src.utils.parametric_flow_rate import InflowFactory

import examples.stokes.SpaceTime.config as config

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
    assert config.problem_name == "Stokes" and config.reduction_method in {'ST-RB', 'SRB-TFO', 'ST-PGRB'}, \
        ("The problem name must be set to 'Stokes' and "
         "the reduction method must be selected in ['ST-RB', 'SRB-TFO', 'ST-PGRB'] to run this script")
    
    ###################################################################################################################
    ############################################### INITIALIZATION ####################################################
    ###################################################################################################################

    my_sp = sp.StokesProblem(None)
    my_sp.configure_fom(None, config.fom_specifics)
    
    # defining the affine decomposition structure
    my_affine_decomposition = adstS.AffineDecompositionHandlerSpaceTimeStokes()
    my_affine_decomposition.set_Q(1, 1, 1)  # number of affine terms
    my_affine_decomposition.set_timesteps(int(config.fom_specifics['number_of_time_instances']))

    # building the RB manager        
    if config.reduction_method == "SRB-TFO":
        my_rb_manager = rmsrbtfoS.RbManagerSRB_TFO_Stokes(my_sp, my_affine_decomposition)
    elif config.reduction_method == "ST-RB":
        my_rb_manager = rmstrbS.RbManagerSTRBStokes(my_sp, my_affine_decomposition)
    elif config.reduction_method == "ST-PGRB":
        my_rb_manager = rmstpgS.RbManagerSTPGRBStokes(my_sp, my_affine_decomposition)
    else:
        raise ValueError(f"Unrecognized reduction method: {config.reduction_method}")

    my_rb_manager.set_parametrizations(config.parametrizations)
    my_rb_manager.use_LU = config.use_LU
    my_rb_manager.N_periods = config.N_periods
    my_rb_manager.Nt_IC = config.Nt_IC

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
             '_space_projection': config.projection_norm,
             'ss_ratio': config.time_subsample_ratio}
    if config.reduction_method == "ST-PGRB":
        specs['_used_norm'] = config.used_norm

    my_rb_manager.build_rb_approximation(config.n_snapshots, **specs)

    inflow_factory = InflowFactory()
    inflow_factory(my_rb_manager, config.flow_type, **config.flow_specs)

    # my_rb_manager.check_offline_phase(10)

    if config.reduction_method in {'ST-RB', 'ST-PGRB'}:
        my_rb_manager.test_rb_solver(config.test_param_nbs,
                                     cycles_specifics=config.cycles_specifics, ss_ratio=config.time_subsample_ratio)
    elif config.reduction_method == 'SRB-TFO':
        my_rb_manager.test_time_marching(config.test_param_nbs,
                                         cycles_specifics=config.cycles_specifics, ss_ratio=config.time_subsample_ratio)
    else:
        raise ValueError(f"Unrecognized reduction method {config.reduction_method}")

    return


if __name__ == "__main__":
    execute()
