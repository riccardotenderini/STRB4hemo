import os
import numpy as np

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def flow_rate_sinusoidal(param, **kwargs):
    """Define the parametrized sinusoidal parametric flow rate

    :param param: vector of parameters
    :type param: list or numpy.ndarray
    :return: flow rate vector
    :rtype: numpy.ndarray
    """

    Nt = kwargs['Nt']
    dt = kwargs['T'] / Nt
    T = kwargs['T']
    N_periods = kwargs['N_periods']
    T_period = T / N_periods
    times = np.linspace(dt, T, Nt)

    flow = (1 - np.cos(2 * np.pi * (times % T_period) / T_period) +
            param[1] * np.sin(2 * np.pi * param[0] * (times % T_period) / T_period))

    return flow


def flow_rate_periodic(param, **kwargs):
    """Define a periodic sinusoidal flow rate

    :param param: vector of parameters
    :type param: list or numpy.ndarray
    return: flow rate vector
    :rtype: numpy.ndarray
    """

    Nt = kwargs['Nt']
    dt = kwargs['T'] / Nt
    T = kwargs['T']
    N_periods = kwargs['N_periods']
    T_period = T / N_periods
    times = np.linspace(dt, T, Nt)

    flow = 10.0 + np.abs(param[1] * np.sin(2 * np.pi * param[0] * (times % T_period) / T_period))

    return flow


def flow_rate_systole(param, **kwargs):
    """Define the systolic parametric flow rate

    :param param: vector of parameters
    :type param: list or numpy.ndarray
    return: flow rate vector
    :rtype: numpy.ndarray
    """

    # reference values, computed from measured inflow
    V0_ref = 1.541
    TM_ref = 0.13375
    VM_ref = 14.16
    Ts_ref = 0.3075
    Tm_ref = 0.375
    Vm_ref = 0.626

    delta_t = param[0]
    delta_V0 = param[1]
    delta_VM = param[2]
    delta_Vm = param[3]

    V0 = V0_ref * (1.0 + delta_V0)      # initial flow
    TM = TM_ref * (1.0 + delta_t)       # time of systolic peak
    VM = VM_ref * (1.0 + delta_VM)      # peak systolic flow
    Ts = Ts_ref * (1.0 + delta_t)       # systolic time
    Tm = Tm_ref * (1.0 + delta_t)       # time to min flow
    Vm = Vm_ref * (1.0 + delta_Vm)      # min flow

    Nt = kwargs['Nt']
    dt = kwargs['T'] / Nt
    T = kwargs['T']
    Tr = kwargs['Nt_ramp'] * dt
    N_periods = kwargs['N_periods']
    T_period = (T - Tr) / N_periods
    times = np.arange(dt, T-Tr+dt, dt)

    ramp = lambda t, Tramp, v0: (v0 / 2) * (1 - np.cos((t + Tramp) * np.pi / Tramp))
    ramp_times = np.arange(-Tr+dt, dt, dt)

    M_sys = np.array([[0, 0, 0, 0, 0, 0, 0, 1],
                      [TM ** 7, TM ** 6, TM ** 5, TM ** 4, TM ** 3, TM ** 2, TM, 1],
                      [Ts ** 7, Ts ** 6, Ts ** 5, Ts ** 4, Ts ** 3, Ts ** 2, Ts, 1],
                      [Tm ** 7, Tm ** 6, Tm ** 5, Tm ** 4, Tm ** 3, Tm ** 2, Tm, 1],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [7 * TM ** 6, 6 * TM ** 5, 5 * TM ** 4, 4 * TM ** 3, 3 * TM ** 2, 2 * TM, 1, 0],
                      [7 * Tm ** 6, 6 * Tm ** 5, 5 * Tm ** 4, 4 * Tm ** 3, 3 * Tm ** 2, 2 * Tm, 1, 0],
                      [42 * Tm ** 5, 30 * Tm ** 4, 20 * Tm ** 3, 12 * Tm ** 2, 6 * Tm, 2, 0, 0]])
    b_sys = np.array([V0, VM, V0, Vm, 0, 0, 0, 0])

    a_sys = np.linalg.solve(M_sys, b_sys)
    systolic_flow = np.polyval(a_sys, times % T_period)

    ramp_flow = ramp(ramp_times, Tr, V0)

    flow = np.hstack([ramp_flow, systolic_flow])

    return flow


def flow_rate_heartbeat(param, **kwargs):
    """Define the heartbeat parametric flow rate

    :param param: vector of parameters
    :type param: list or numpy.ndarray
    return: flow rate vector
    :rtype: numpy.ndarray
    """

    # reference values, computed from measured inflow
    V0_ref = 1.541
    TM_ref = 0.13375
    VM_ref = 14.16
    Ts_ref = 0.3075
    Tm_ref = 0.375
    Vm_ref = 0.626
    TMd_ref = 0.63375
    VMd_ref = 2.092
    Tf_ref = 0.750
    Vf_ref = 1.527

    delta_t = param[0]
    delta_V0 = param[1]
    delta_VM = param[2]
    delta_Vm = param[3]
    delta_VMd = param[4]
    delta_Vf = param[1]

    V0 = V0_ref * (1.0 + delta_V0)  # initial flow
    TM = TM_ref * (1.0 + delta_t)  # time of systolic peak
    VM = VM_ref * (1.0 + delta_VM)  # peak systolic flow
    Ts = Ts_ref * (1.0 + delta_t)  # systolic time
    Tm = Tm_ref * (1.0 + delta_t)  # time to min flow
    Vm = Vm_ref * (1.0 + delta_Vm)  # min flow
    Td = Tm_ref if Tm > Tm_ref else Tm  # border between systole and diastole
    TMd = TMd_ref  # time of diastolic peak
    VMd = VMd_ref * (1.0 + delta_VMd)  # peak diastolic flow
    Tf = Tf_ref  # final time
    Vf = Vf_ref * (1 + delta_Vf)  # flow at final time

    Nt = kwargs['Nt']
    dt = kwargs['T'] / Nt
    T = kwargs['T']
    Nt_ramp = kwargs['Nt_ramp']
    Tr = kwargs['Nt_ramp'] * dt
    N_periods = kwargs['N_periods']
    T_period = (T - Tr) / N_periods
    times = np.arange(dt, T - Tr + dt, dt)
    Td_idx = np.where(times > Td)[0][0]

    ramp = lambda t, Tramp, v0: (v0 / 2) * (1 - np.cos((t + Tramp) * np.pi / Tramp))
    ramp_times = np.arange(-Tr + dt, dt, dt)

    # systolic flow
    M_sys = np.array([[0, 0, 0, 0, 0, 0, 0, 1],
                      [TM ** 7, TM ** 6, TM ** 5, TM ** 4, TM ** 3, TM ** 2, TM, 1],
                      [Ts ** 7, Ts ** 6, Ts ** 5, Ts ** 4, Ts ** 3, Ts ** 2, Ts, 1],
                      [Tm ** 7, Tm ** 6, Tm ** 5, Tm ** 4, Tm ** 3, Tm ** 2, Tm, 1],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [7 * TM ** 6, 6 * TM ** 5, 5 * TM ** 4, 4 * TM ** 3, 3 * TM ** 2, 2 * TM, 1, 0],
                      [7 * Tm ** 6, 6 * Tm ** 5, 5 * Tm ** 4, 4 * Tm ** 3, 3 * Tm ** 2, 2 * Tm, 1, 0],
                      [42 * Tm ** 5, 30 * Tm ** 4, 20 * Tm ** 3, 12 * Tm ** 2, 6 * Tm, 2, 0, 0]])
    b_sys = np.array([V0, VM, V0, Vm, 0, 0, 0, 0])
    a_sys = np.linalg.solve(M_sys, b_sys)

    # diastolic flow
    Vd = np.polyval(a_sys, times[Td_idx])
    Vdp = (np.polyval(a_sys, times[Td_idx]) - np.polyval(a_sys, times[Td_idx] - dt)) / dt

    M_dia = np.array([[Td ** 4, Td ** 3, Td ** 2, Td, 1.0],
                      [4 * Td ** 3, 3 * Td ** 2, 2 * Td, 1.0, 0.0],
                      [TMd ** 4, TMd ** 3, TMd ** 2, TMd, 1.0],
                      [4 * TMd ** 3, 3 * TMd ** 2, 2 * TMd, 1.0, 0.0],
                      [Tf ** 4, Tf ** 3, Tf ** 2, Tf, 1.0]])
    b_dia = np.array([Vd, Vdp, VMd, 0.0, Vf])
    a_dia = np.linalg.solve(M_dia, b_dia)

    flow = []
    for t in times:
        if t % T_period < Td:
            flow.append(np.polyval(a_sys, t % T_period))
        else:
            flow.append(np.polyval(a_dia, t % T_period))

    ramp_flow = ramp(ramp_times, Tr, V0)

    flow = np.concatenate([ramp_flow, np.array(flow)])

    return flow


def __splineBasisFunction(i, p, knots, t):
    """Compute the i-th B-spline basis function of degree p at time t"""
    if p == 0:
        return 1.0 if (knots[i] <= t < knots[i + 1]) else 0.0

    denom1 = knots[i + p] - knots[i]
    denom2 = knots[i + p + 1] - knots[i + 1]

    alpha1 = 0.0 if (denom1 == 0) else (t - knots[i]) / denom1
    alpha2 = 0.0 if (denom2 == 0) else (knots[i + p + 1] - t) / denom2

    retVal = (alpha1 * __splineBasisFunction(i, p - 1, knots, t) +
              alpha2 * __splineBasisFunction(i + 1, p - 1, knots, t))

    return retVal


def __evaluateBSpline(knots, control_points, degree, t):
    """Evaluate the B-spline of given degree defined by the control points and knots at time t"""
    result = 0.0
    for i in range(len(knots) - degree - 1):
        result += __splineBasisFunction(i, degree, knots, t) * control_points[i]

    return result


def inflow_rate_bypass(param, **kwargs):
    """Define the bypass inflow rate

    :param param: vector of parameters
    :type param: list or numpy.ndarray
    return: flow rate vector
    :rtype: numpy.ndarray
    """

    Nt = kwargs['Nt']
    dt = kwargs['T'] / Nt
    T = kwargs['T']
    N_periods = kwargs['N_periods']
    T_period = T / N_periods
    times = np.linspace(dt, T, Nt)

    knots = [0.000000, 0.000000, 0.000000, 0.000000, 0.013009, 0.019696,
             0.026383, 0.033678, 0.040365, 0.043404, 0.048875, 0.053131,
             0.059210, 0.064073, 0.070152, 0.075623, 0.079271, 0.083526,
             0.093252, 0.102979, 0.107842, 0.112705, 0.120608, 0.130334,
             0.142492, 0.152827, 0.163161, 0.168632, 0.176535, 0.183830,
             0.191125, 0.197204, 0.201459, 0.206930, 0.214833, 0.220304,
             0.224559, 0.230638, 0.235502, 0.240365, 0.244620, 0.247660,
             0.251307, 0.255562, 0.260426, 0.266505, 0.270152, 0.275623,
             0.279271, 0.285957, 0.292644, 0.299331, 0.304802, 0.310274,
             0.315137, 0.320000, 0.323040, 0.324255, 0.326687, 0.327903,
             0.330942, 0.332766, 0.333982, 0.335198, 0.337021, 0.340669,
             0.341884, 0.344924, 0.349179, 0.357082, 0.364377, 0.374711,
             0.383222, 0.389301, 0.397812, 0.403283, 0.407538, 0.412401,
             0.414225, 0.416049, 0.419088, 0.422736, 0.424559, 0.431246,
             0.440973, 0.443404, 0.446444, 0.453739, 0.465897, 0.481702,
             0.495076, 0.502371, 0.513921, 0.522432, 0.533982, 0.547356,
             0.563769, 0.580182, 0.588693, 0.597204, 0.607538, 0.619088,
             0.633070, 0.644012, 0.651307, 0.661641, 0.673799, 0.685957,
             0.698723, 0.716960, 0.733982, 0.743100, 0.753435, 0.764985,
             0.800000, 0.800000, 0.800000, 0.800000]

    control_points = [0.000000, 0.107424, 0.270066, 0.686590, 1.079219, 1.529738,
                      1.665662, 2.479341, 2.526228, 3.203902, 3.330118, 3.823002,
                      4.134922, 4.419553, 4.866427, 5.221333, 5.486763, 5.489934,
                      5.871342, 6.121987, 6.415745, 6.376537, 6.730727, 6.422683,
                      6.752654, 6.440733, 6.332223, 6.216715, 5.968089, 5.894305,
                      5.600864, 5.428963, 5.369481, 5.061707, 4.794320, 4.624618,
                      4.313969, 4.128478, 3.923578, 3.642798, 3.488203, 3.259535,
                      2.999609, 2.896841, 2.565957, 2.385971, 1.959699, 1.810823,
                      1.575021, 1.394963, 1.243907, 1.005325, 0.896644, 0.603720,
                      0.632064, 0.022120, 0.143219, -0.469150, -0.431882, -0.740658,
                      -1.010876, -1.553619, -1.760727, -1.723078, -2.329966, -2.367022,
                      -2.828083, -2.655143, -2.557138, -2.541135, -2.543657, -2.546730,
                      -2.530823, -2.369128, -2.139345, -2.180916, -1.869986, -1.511433,
                      -1.222019, -1.160536, -0.412524, -0.584778, -0.679712, -0.082808,
                      0.411558, 0.472179, 0.526822, 0.530207, 0.607694, 0.662179,
                      0.784511, 0.610707, 0.861281, 0.720515, 0.747265, 0.519615,
                      0.806465, 0.930635, 0.743332, 0.720410, 0.765768, 0.603564,
                      0.547469, 0.661133, 0.604920, 0.617585, 0.628546, 0.546654,
                      0.432488, 0.558382, 0.369634, 0.407125, 0.174742, 0.000000]

    flow = np.array([__evaluateBSpline(knots, control_points, 3, t) for t in times % T_period])

    return flow


def outflow_rate_bypass(param, **kwargs):
    """Define the bypass outflow rate (applicable only with membrane model!)

    :param param: vector of parameters
    :type param: list or numpy.ndarray
    return: flow rate vector
    :rtype: numpy.ndarray
    """

    Nt = kwargs['Nt']
    dt = kwargs['T'] / Nt
    T = kwargs['T']
    N_periods = kwargs['N_periods']
    T_period = T / N_periods
    times = np.linspace(dt, T, Nt)

    knots = [0.        , 0.        , 0.        , 0.        , 0.00222222,
             0.00444444, 0.00666667, 0.00888889, 0.01111111, 0.01333333,
             0.01555556, 0.01777778, 0.02000000, 0.03048327, 0.03866171,
             0.04460967, 0.04907063, 0.05576208, 0.06171004, 0.07137546,
             0.07732342, 0.08550186, 0.09293680, 0.10111524, 0.10706320,
             0.11375465, 0.12342007, 0.13457249, 0.14498141, 0.15167286,
             0.15390335, 0.15687732, 0.15836431, 0.15985130, 0.16059480,
             0.16133829, 0.16356877, 0.16431227, 0.16802974, 0.17100372,
             0.17323420, 0.17397770, 0.17546468, 0.17620818, 0.17769517,
             0.17843866, 0.18066914, 0.18215613, 0.18289963, 0.18513011,
             0.18959108, 0.19330855, 0.20817844, 0.21412639, 0.22007435,
             0.22379182, 0.23048327, 0.23717472, 0.23940520, 0.24312268,
             0.24535316, 0.24832714, 0.25204461, 0.25576208, 0.26245353,
             0.26691450, 0.27286245, 0.27509294, 0.28029740, 0.28475836,
             0.28698885, 0.29591078, 0.31078067, 0.32490706, 0.34052045,
             0.35687732, 0.36802974, 0.37695167, 0.38513011, 0.39405204,
             0.40520446, 0.41412639, 0.42230483, 0.42750929, 0.44386617,
             0.45427509, 0.46394052, 0.47063197, 0.47881041, 0.49144981,
             0.49665428, 0.50631970, 0.51598513, 0.52565056, 0.53457249,
             0.54200743, 0.54795539, 0.55687732, 0.56133829, 0.57323420,
             0.58364312, 0.59107807, 0.60371747, 0.61338290, 0.62453532,
             0.63568773, 0.64684015, 0.65576208, 0.66245353, 0.67509294,
             0.68475836, 0.69070632, 0.69368030, 0.69442379, 0.69591078,
             0.70185874, 0.70631970, 0.71003717, 0.71747212, 0.72416357,
             0.72936803, 0.73085502, 0.73605948, 0.73977695, 0.74126394,
             0.74646840, 0.75910781, 0.76802974, 0.77546468, 0.78000000,
             0.78222222, 0.78444444, 0.78666667, 0.78888889, 0.79111111,
             0.79333333, 0.79555556, 0.79777778, 0.80000000, 0.80000000,
             0.80000000, 0.80000000]

    control_points = [0.        ,  0.01140310,  0.03420929,  0.21153903,  0.47112086,
                      0.79231050,  1.13319945,  1.45467187,  1.71312260,  1.89469396,
                      1.93583610,  2.66445842,  3.00006167,  3.41034204,  4.03832239,
                      4.27416276,  4.83065416,  4.78006153,  5.35344510,  5.47558548,
                      5.76167234,  5.72553231,  6.04687280,  6.22133741,  5.86138032,
                      6.28198243,  6.06018535,  6.07167588,  6.87074263,  6.89855466,
                      7.46259226,  7.61483051,  8.12473030,  8.78240768,  8.63081528,
                      9.40117935,  9.26463161,  8.99007553,  9.11903575,  8.28653217,
                      8.37461547,  7.52550251,  7.62907148,  6.67511634,  6.90081240,
                      6.53362503,  5.81516343,  5.81198389,  5.46865751,  4.90407916,
                      5.42525376,  4.79951252,  4.59973213,  3.97788861,  3.60759734,
                      3.65470833,  2.79043679,  2.87292451,  2.20343042,  2.00655357,
                      1.59605413,  1.21879554,  1.10980417,  0.40644433,  0.51974372,
                      -0.59203009, -0.70570290, -0.90252017, -1.92253192, -1.59956304,
                      -1.79772976, -1.62592415, -1.55993208, -1.43202631, -1.71555109,
                      -1.41927449, -1.20820251, -0.95761819, -1.07167944, -0.81594861,
                      -0.71835637,  0.00508391, -0.32814105, -0.00461953,  0.15338827,
                      0.53195530,  0.80974215,  0.69682703,  1.04553653,  1.12543914,
                      1.14329098,  1.06404570,  0.81610174,  1.12405914,  0.80334519,
                      0.86028034,  1.35298399,  1.01541468,  0.98609567,  1.32340832,
                      1.36757105,  1.32441407,  1.46684230,  1.35005258,  1.33145863,
                      1.28879385,  1.06799196,  1.17147405,  1.33184902,  0.90971089,
                      1.10924362,  0.44565021,  0.15958389,  0.07643125,  0.34714656,
                      0.58679833,  0.73667865,  0.42956946,  0.52633164, -0.14420232,
                      -0.12714337, -0.06883760, -0.94910189, -0.88223117, -0.92263130,
                      -0.64090488, -0.52991735, -0.30170202, -0.31616280, -0.28245275,
                      -0.24071168, -0.18729726, -0.13100867, -0.07788697, -0.03497508,
                      -0.00565560, -0.00188520,  0.]

    flow = np.array([__evaluateBSpline(knots, control_points, 3, t) for t in times % T_period])

    return flow


class InflowFactory:
    def __init__(self):
        return

    def __call__(self, rb_manager, flow_type, **kwargs):

        weak_inflow = rb_manager.n_weak_inlets > 0
        weak_outflow = rb_manager.n_weak_outlets > 0

        if flow_type == 'default':
            inflow_rate_fun = outflow_rate_fun = lambda mu: flow_rate_sinusoidal(mu, **kwargs)
            n_params_in, n_params_out = 2, 0
        elif flow_type == 'periodic':
            inflow_rate_fun = outflow_rate_fun = lambda mu: flow_rate_periodic(mu, **kwargs)
            n_params_in, n_params_out = 2, 0
        elif flow_type == 'systolic':
            inflow_rate_fun = outflow_rate_fun = lambda mu: flow_rate_systole(mu, **kwargs)
            n_params_in, n_params_out = 4, 0
        elif flow_type == 'heartbeat':
            inflow_rate_fun = outflow_rate_fun = lambda mu: flow_rate_heartbeat(mu, **kwargs)
            n_params_in, n_params_out = 5, 0
        elif flow_type == 'bypass':
            inflow_rate_fun = lambda mu: inflow_rate_bypass(mu, **kwargs)
            outflow_rate_fun = lambda mu: outflow_rate_bypass(mu, **kwargs)
            n_params_in, n_params_out = 0, 0
        else:
            raise ValueError(f"Unrecognized flow rate type: {flow_type}")

        rb_manager.set_flow_rate(inflow_rate=(inflow_rate_fun, n_params_in) if weak_inflow else (None, 0),
                                 outflow_rate=(outflow_rate_fun, n_params_out) if weak_outflow else (None, 0))

        return
