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
root_tests = os.path.normpath('/home/riccardo/Desktop/PhD/MyRepos/STRB4hemo/TESTS/')  # TO SET --> root of the tests directory

problem_type = "Unsteady"  # type of the problem --> choose between steady and unsteady
problem_name = "Navier-Stokes_Membrane"  # name of the problem --> choose between Stokes, Navier-Stokes, Navier-Stokes_Membrane

# root to the directory storing the data and where the results are saved
root_test = os.path.join(root_tests, "RB", problem_type)
create_directory_if_not_existing(root_test)

if problem_type == "Steady":
    raise ValueError("No implementation of the steady Navier-Stokes Membrane problem is provided!")

elif problem_type == "Unsteady":

    mesh_name = 'tube_1x2_h0.20'  # name of the mesh

    n_weak_inlets = 1  # number of weak inlet Dirichlet boundaries   --> 1 for tube, 2 for bifurcation, 2 for bypass
    n_weak_outlets = 0  # number of weak outlet Dirichlet boundaries --> 0 for tube, 0 for bifurcation, 1 for bypass

    parametrizations = ['inflow', 'structure']  # list of parametrizations
    flow_type = 'default'  # type of flow to be imposed
    parametrization_name = ("" + (flow_type if 'inflow' in parametrizations else "") +
                            ("+clot" if 'clot' in parametrizations else "") +
                            ("+structure" if 'structure' in parametrizations else ""))

    if flow_type == 'default':
        flow_specs = {'T': 0.40 + 0.00,
                      'Nt': 40,
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

    wall_elasticity = 0  # elasticity of the external wall

    N_periods = 1  # number of periods (e.g. heartbeats) in the flow rate of FOM data
    cycles_specifics = {'n_cycles': 1,  # number of periods to be simulated
                        'update_T': False,  # change period duration across cycles
                        'update_params': False}  # update parameters across cycles
    time_subsample_ratio = 1  # subsampling ratio to apply over discrete temporal evaluations

    # specifics of the fom problem to be solved
    fom_specifics = {
        'is_linear_problem': False,  # the problem is linear or not
        'mesh_name': mesh_name,  # name of the mesh
        'final_time': flow_specs['T'],  # final time of the simulation
        'number_of_time_instances': flow_specs['Nt'] // time_subsample_ratio,  # number of FOM time instances
        }
    total_fom_coordinates_number = None

    reduction_method = 'ST-RB'  # method to be employed (ST-RB, ST-PGRB, SRB-TFO)
    projection_norm = 'standard'  # how to perform spatial projection ('standard', 'natural')
    used_norm = "l2"  # norm to be considered for residual minimization in the ST-PGRB method, within {'X', 'P', 'l2'}
    use_LU = False  # use LU factorization to speed up solve across multiple parameter instances

    n_snapshots = 50  # number of snapshots used to assemble the reduced bases
    N_components_NL_term = 10  # number of affine components to consider for the non-linear convective term

    newton_specs = dict()  # specifics of the Newton's method
    newton_specs['tolerance'] = 5e-5  # tolerance on relative residual
    newton_specs['absolute tolerance'] = 1e-3  # tolerance on absolute residual
    newton_specs['max error'] = 1e-3  # maximum relative residual
    newton_specs['absolute max error'] = 1e-1  # maximum absolute residual
    newton_specs['max iterations'] = 10  # maximal number of iterations
    newton_specs['IG mode'] = "average"  # mode to choose the initial guess ('NN', 'PODI', 'average', 'zero')
    newton_specs['neighbors number'] = 3  # number of neighbors for NN interpolation
    newton_specs['use convective jacobian'] = False  # use the convective jacobian in the Newton's method

    tolerances = dict()  # dictionary of POD tolerances
    tolerances['velocity-space'] = 1e-3
    tolerances['velocity-time'] = 1e-3
    tolerances['pressure-space'] = 1e-3
    tolerances['pressure-time'] = 1e-3
    tolerances['lambda-space'] = 1e-5
    tolerances['lambda-time'] = 1e-3

    test_param_nbs = [50]  # list of testing instances
    compute_IG_error = False  # compute error on the initial guess

    IMPORT_SNAPSHOTS = False  # import available snapshots
    IMPORT_OFFLINE_QUANTITIES = True  # import available reduced quantities
    SAVE_OFFLINE_QUANTITIES = True  # save reduced quantities, if computed
    SAVE_RESULTS = False  # save solution for visualization
    ######################################## TEST DEFINITION ########################################################

    common_directory = os.path.join(root_test, problem_name)

    test_nb = 1  # TO SET --> choose the test to run --> if None, a new test is done, given the specifics above

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
        # N_components_NL_term = test_specs['n_components_nl_term']

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

    core_results_directory = os.path.join(test_directory, 'results')
    create_directory_if_not_existing(core_results_directory)

    results_directory = os.path.join(core_results_directory, f"NLcomp_{N_components_NL_term}")
    if reduction_method == "ST-PGRB":
        core_results_directory = os.path.join(test_directory, f'results_{used_norm}')
        results_directory = os.path.join(core_results_directory, f"NLcomp_{N_components_NL_term}")
    create_directory_if_not_existing(results_directory)

    ############################################## POST-PROCESS INFO ##################################################

    plot_solutions = True
    count_RBvectors = True
    plot_generalizedCoords = False

else:
    raise ValueError(f"Unrecognized problem type '{problem_type}'."
                     f" Admissible values: [Steady, Unsteady]")
