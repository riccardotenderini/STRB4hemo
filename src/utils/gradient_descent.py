import numpy as np
import os

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class GradientDescent:
    def __init__(self, tol=1e-5, min_err=1e-14, max_iter=10, alpha=0.1):

        self.tol = tol  # convergence tolerance
        self.min_err = min_err  # minimal absolute error
        self.max_iter = max_iter  # maximal iterations number
        self.alpha = alpha  # learning rate

        return

    def __call__(self, fun, grad, initial_guess, pre_iter=None, post_iter=None):
        sol = initial_guess
        cnt = 0

        if pre_iter is not None:
            pre_iter(sol)
        curFun = fun(sol)
        curGrad = grad(sol)
        err = np.linalg.norm(curFun)
        err0 = err
        if post_iter is not None:
            post_iter(sol)

        def is_invalid_error(error):
            return np.isnan(error) or np.isinf(error)

        logger.debug(f"Gradient Descent method. Iteration: {cnt}  -  Relative Error: {err/err0:.2e}  -  "
                     f"Absolute Error: {err:.2e}")

        while (err / err0 > self.tol and err > self.min_err and not is_invalid_error(err)) and cnt < self.max_iter:

            sol -= self.alpha * curGrad
            cnt += 1

            if pre_iter is not None:
                pre_iter(sol)
            curFun = fun(sol)
            curGrad = grad(sol)
            err = np.linalg.norm(curFun)
            if post_iter is not None:
                post_iter(sol)

            logger.debug(f"Gradient Descent method. Iteration: {cnt}  -  Relative Error: {err / err0:.2e}  -  "
                         f"Absolute Error: {err:.2e}")

        if (cnt == self.max_iter and (err / err0 > self.tol or err > self.min_err or is_invalid_error(err))) \
           or is_invalid_error(err):
            logger.critical(f"Gradient Descent method has failed after {cnt} iterations!")
            converged = False
        else:
            logger.debug(f"Gradient Descent method converged after {cnt} iterations! -  Relative Error: {err/err0:.2e}  -  "
                         f"Absolute Error: {err:.2e}")
            converged = True

        return sol, converged
