import os
import json


def create_directory_if_not_existing(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created new directory: {directory}")


#######################################################################################################################
######################################## PROBLEM CONFIGURATION ########################################################
#######################################################################################################################

root = os.path.normpath('/home/riccardo/Desktop/PhD/MyRepos/STRB4hemo')  # TO SET --> root of the project directory
root_tests = os.path.normpath('/home/riccardo/Desktop/PhD/MyRepos/STRB4hemo/TESTS/')  # TO SET root of the tests directory

problem_type = "Unsteady"  # type of the problem --> choose between steady and unsteady
problem_name = "Stokes"  # name of the problem --> choose between Stokes, Navier-Stokes, Navier-Stokes_Membrane

# root to the directory storing the data and where the results are saved
root_test = os.path.join(root_tests, "RB", problem_type)
create_directory_if_not_existing(root_test)

if problem_type == "Steady":
    raise ValueError("No implementation of the steady Stokes problem is provided!")

elif problem_type == "Unsteady":
    
    mesh_name = 'tube_1x2_h0.20'  # name of the mesh

    n_weak_inlets = 1  # number of weak inlet Dirichlet boundaries   --> 1 for tube, 2 for bifurcation, 2 for bypass
    n_weak_outlets = 0  # number of weak outlet Dirichlet boundaries --> 0 for tube, 0 for bifurcation, 1 for bypass

    parametrizations = ['inflow']  # list of parametrizations
    flow_type = 'default'  # type of flow to be imposed
    parametrization_name = ("" + (flow_type if 'inflow' in parametrizations else "") +
                            ("+clot" if 'clot' in parametrizations else ""))

    if flow_type == 'default':
        flow_specs = {'T': 0.80 + 0.00,
                      'Nt': 80,
                      'Nt_ramp': 0,
                      'N_periods': 1}
    elif flow_type == 'systolic':
        flow_specs = {'T': 0.37 + 0.05,
                      'Nt': 43,
                      'Nt_ramp': 5,
                      'N_periods': 1}
    elif flow_type == 'heartbeat':
        flow_specs = {'T': 0.75 + 0.05,
                      'Nt': 160,
                      'Nt_ramp': 10,
                      'N_periods': 1}
    elif flow_type == 'bypass':
        flow_specs = {'T': 0.80 + 0.00,
                      'Nt': 40,
                      'Nt_ramp': 0,
                      'N_periods': 1}
    else:
        raise ValueError(f"Unrecognized flow rate type {flow_type}")

    N_periods = 1  # number of periods (e.g. heartbeats) in the flow rate of FOM data
    cycles_specifics = {'n_cycles': 1,  # number of periods to be simulated
                        'update_T': False,  # change period duration across cycles
                        'update_params': False}  # update parameters across cycles
    time_subsample_ratio = 1  # subsampling ratio to apply over discrete temporal evaluations
                
    # specifics of the fom problem to be solved    
    fom_specifics = {
        'is_linear_problem': True,  # the problem is linear or not
        'mesh_name': mesh_name,  # name of the mesh
        'final_time': flow_specs['T'],  # final time of the simulation
        'number_of_time_instances': flow_specs['Nt'] // time_subsample_ratio,  # number of FOM time instances
    }
    total_fom_coordinates_number = None

    reduction_method = 'ST-RB'  # method to be employed (ST-RB, SRB-TFO, ST-PGRB)
    projection_norm = 'standard'  # how to perform spatial projection ('standard', 'natural')
    used_norm = "P"  # norm to be considered for residual minimization in the ST-PGRB method, within {'X', 'P', 'l2'}
    use_LU = False  # use LU factorization to speed up solve across multiple parameter instances

    n_snapshots = 25  # number of snapshots used to assemble the reduced bases

    tolerances = dict()  # dictionary of POD tolerances
    tolerances['velocity-space'] = 1e-3
    tolerances['velocity-time'] = 1e-3
    tolerances['pressure-space'] = 1e-3
    tolerances['pressure-time'] = 1e-3
    tolerances['lambda-space'] = 1e-5
    tolerances['lambda-time'] = 1e-3

    test_param_nbs = [25]  # list of testing instances

    IMPORT_SNAPSHOTS = True  # import available snapshots
    IMPORT_OFFLINE_QUANTITIES = True  # import available reduced quantities
    SAVE_OFFLINE_QUANTITIES = True  # save reduced quantities, if computed
    SAVE_RESULTS = False  # save solution for visualization

    ######################################## TEST DEFINITION ########################################################

    common_directory = os.path.join(root_test, problem_name)

    test_nb = None  # TO SET --> test to run --> if None, a new test is done, given the specifics specified above

    if test_nb is None:
        test_specs = dict()
        test_specs['tolerances'] = tolerances
        test_specs['n_snapshots'] = n_snapshots
        test_specs['space_projection'] = projection_norm
        test_specs['time_subsample_ratio'] = time_subsample_ratio

        test_cnt = 1
        while os.path.isdir(os.path.join(common_directory, mesh_name, parametrization_name,
                                         reduction_method, f"Test{test_cnt}")):
            test_cnt += 1
        test_nb = test_cnt
        create_directory_if_not_existing(os.path.join(common_directory, mesh_name, parametrization_name,
                                                      reduction_method, f"Test{test_cnt}"))
        with open(os.path.join(common_directory, mesh_name, parametrization_name,
                               reduction_method, f"Test{test_cnt}", 'test_specs.json'), 'w') as fp:
            json.dump(test_specs, fp)

    else:
        assert os.path.isdir(os.path.join(common_directory, mesh_name, parametrization_name,
                                          reduction_method, f"Test{test_nb}"))
        assert os.path.isfile(os.path.join(common_directory, mesh_name, parametrization_name,
                                           reduction_method, f"Test{test_nb}",
                                           "test_specs.json"))

        with open(os.path.join(common_directory, mesh_name, parametrization_name,
                               reduction_method, f"Test{test_nb}", 'test_specs.json'), 'r') as fp:
            test_specs = json.load(fp)

        tolerances = test_specs['tolerances']
        n_snapshots = test_specs['n_snapshots']
        projection_norm = test_specs['space_projection']
        time_subsample_ratio = test_specs['time_subsample_ratio'] if 'time_subsample_ratio' in test_specs else 1
        fom_specifics['number_of_time_instances'] = flow_specs['Nt'] // time_subsample_ratio

    #################################### PATHS CONFIGURATION #######################################################

    FEM_dir = os.path.join(common_directory, mesh_name, parametrization_name, 'FEM_data')
    snapshots_path = os.path.join(FEM_dir, 'snapshots')
    fom_structures_path = os.path.join(FEM_dir, 'matrices')

    test_directory = os.path.join(common_directory, mesh_name, parametrization_name,
                                  reduction_method, f"Test{test_nb}")
    create_directory_if_not_existing(test_directory)

    data_directory = os.path.join(test_directory, 'data')
    create_directory_if_not_existing(data_directory)
    basis_path = os.path.join(data_directory, 'basis')
    rb_blocks_path = os.path.join(data_directory, 'rb_blocks')
    gen_coords_path = os.path.join(data_directory, 'gen_coords')
    reduced_structures_path = os.path.join(data_directory, 'new_matrices')

    results_directory = os.path.join(test_directory, 'results')
    if reduction_method == "ST-PGRB":
        results_directory = os.path.join(test_directory, f'results_{used_norm}')
    create_directory_if_not_existing(results_directory)
    
    ############################################## POST-PROCESS INFO ##################################################
    
    plot_solutions = True
    count_RBvectors = True
    plot_generalizedCoords = False
    
else:
    raise ValueError(f"Unrecognized problem type '{problem_type}'. "
                     f"Admissible values: [Steady, Unsteady]")
