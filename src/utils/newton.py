import os
import warnings
import numpy as np
import scipy

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class Newton:
    def __init__(self, tol=1e-4, tol_abs=1e-8, max_iter=10, max_err=1e-2, max_err_abs=1e-6, jac_iter=1, alpha=1.0):

        self.tol = tol  # convergence tolerance on relative error
        self.tol_abs = tol_abs  # convergence tolerance on absolute error
        self.max_iter = max_iter  # maximal iterations number
        self.max_err = max_err  # maximal relative error
        self.max_err_abs = max_err_abs  # maximal absolute error
        self.jac_iter = jac_iter  # number of iterations after which recomputing the jacobian
        self.alpha = alpha  # relaxation parameter

        return

    def __call__(self, fun, jac, initial_guess, pre_iter=None, post_iter=None, use_lu=False, verbose=True):
        sol = initial_guess
        cnt = 0

        if pre_iter is not None:
            pre_iter(sol)
        curFun = fun(sol)
        curJac = jac(sol)
        err = np.linalg.norm(curFun)
        err0 = err
        if post_iter is not None:
            post_iter(sol)

        if use_lu:
            assert type(curJac) is tuple

        def __is_invalid_error(error):
            return np.isnan(error) or np.isinf(error)

        if verbose:
            logger.debug(f"Newton's method. Iteration: {cnt}  -  Relative Error: {err/err0:.2e}  -  "
                         f"Absolute Error: {err:.2e}")

        while (((err / err0 > self.max_err or err > self.max_err_abs) or
                (err / err0 > self.tol and err > self.tol_abs))
               and cnt < self.max_iter):

            # from scipy.linalg import cho_factor, cho_solve
            # is_symmetric = (np.max(np.abs(curJac - curJac.T)) <= 1e-14)
            # fac1, fac2 = cho_factor(curJac)
            # incr = -cho_solve((fac1, fac2), curFun)

            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    incr = -scipy.linalg.lu_solve(curJac, curFun) if use_lu else -scipy.linalg.solve(curJac, curFun)
                except scipy.linalg.LinAlgWarning:
                    logger.warning(f"The linear system is ill-conditioned and the solution is inaccurate! -- "
                                   f"Condition number: {np.linalg.cond(curJac):.2e}")
                    err = np.nan
                else:
                    sol += self.alpha * incr
                    cnt += 1

                    if pre_iter is not None:
                        pre_iter(sol)
                    curFun = fun(sol)
                    curJac = jac(sol) if cnt % self.jac_iter == 0 else curJac
                    err = np.linalg.norm(curFun)
                    if post_iter is not None:
                        post_iter(sol)

                    if verbose:
                        logger.debug(f"Newton's method. Iteration: {cnt}  -  Relative Error: {err/err0:.2e}  -  "
                                     f"Absolute Error: {err:.2e}")

        if __is_invalid_error(err) or (cnt == self.max_iter and (err / err0 > self.tol and err > self.tol_abs)):
            logger.critical(f"Newton's method has failed after {cnt} iterations!")
            converged = False
        else:
            if verbose:
                logger.debug(f"Newton's method converged after {cnt} iterations! -  Relative Error: {err/err0:.2e}  -  "
                             f"Absolute Error: {err:.2e}")
            converged = True

        return sol, converged
