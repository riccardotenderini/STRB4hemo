#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 8 17:21:31 2018
@author: Niccolò Dal Santo
@email : niccolo.dalsanto@epfl.ch
"""
import numpy as np
import os
import time
import warnings

from scipy.sparse import issparse, csc_matrix
from sksparse.cholmod import cholesky
try:
    from sklearn.decomposition import PCA
except ImportError:
    warnings.warn("Cannot import PCA from sklearn!")
    PCA = None

import logging.config
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class ProperOrthogonalDecomposition:
    """Class (callable object) which handles the computation of the Proper Orthogonal Decomposition of matrices. In the
    context of the current project, such procedure is used to derive a Reduced Basis, starting from a set of snapshots,
    computed by solving a parametric FOM problem for different parameter values and, eventually, at different time
    instants if the considered problem is time-dependent.
    """

    def __init__(self):
        """Initialization of the ProperOrthogonalDecomposition class
        """

        self.M_snapshots_matrix = np.zeros(0)
        self.M_basis = np.zeros(0)
        self.M_singular_values = np.zeros(0)
        self.M_N = 0
        self.M_ns = 0
        self.M_Nh = 0
        return

    @property
    def basis(self):
        return self.M_basis

    @property
    def singular_values(self):
        return self.M_singular_values

    def __call__(self, _snapshots_matrix, _tol=1e-4, _norm_matrix=None):
        """Call method of the class, which allows to compute the Proper Orthogonal Decomposition of the snapshots matrix
        given as input. The POD is performed by first computing the Singular Value Decomposition (SVD) of the input
        matrix and lately performing a dimensionality reduction, using the squared l2-norm of the singular values as
        energy metric; in particular, after the SVD has been performed, the singular values (ordered from the bigger to
        the smaller) are scanned, until the relative energy (i.e. squared l2-norm) of the already scanned ones over the
        energy of the full set of singular values overcomes a certain threshold value. The number of scanned singular
        values defines then the dimensionality of the reduced basis (say N), while the sub-matrix constructed by
        extracting the first N columns from the left-most matrix arising from the SVD of the snapshots' matrix encodes
        the Reduced Basis. If '_norm_matrix' is a symmetric and positive-definite matrix, the orthonormalization is done
        with respect to the norm that it defines; otherwise orthonormalization in l2-norm is performed.
        :param _snapshots_matrix: matrix of the snapshots, over which the POD has to be performed
        :type _snapshots_matrix: numpy.ndarray
        :param _tol: tolerance for the relative energy of the already scanned singular values with respect to the
          energy of the full set of singular values. Energy is interpreted as the squared l2-norm of the singular values.
          The actual tolerance used in the algorithm is tolerance = 1.0 - _tol, thus '_tol' is intended to be close
          to 0. Defaults to 1e-5
        :type _tol: float
        :param _norm_matrix: matrix defining a norm with respect to which the resulting bases vectors will be
          orthonormalized. If None, the orthonormalization is performed in l2-norm. Defaults to None.
        :type _norm_matrix: numpy.ndarray or scipy.sparse or NoneType
        :return: the result of the POD
        :rtype: numpy.ndarray
        """

        start = time.time()

        if type(self.M_snapshots_matrix) is not np.ndarray:
            raise TypeError(f"The snapshots matrix must be of type numpy.ndarray, while here it is of type"
                            f"{type(self.M_snapshots_matrix)}")

        if _norm_matrix is not None and not issparse(_norm_matrix):
            _norm_matrix = csc_matrix(_norm_matrix)

        self.M_snapshots_matrix = _snapshots_matrix
        self.M_Nh = self.M_snapshots_matrix.shape[0]
        self.M_ns = self.M_snapshots_matrix.shape[1]

        if _norm_matrix is not None:
            try:
                assert _norm_matrix.shape[0] == _norm_matrix.shape[1] == self.M_Nh
            except AssertionError:
                raise TypeError(f"The norm matrix for the POD must be a 2D numpy array of shape "
                                f"{self.M_Nh} x {self.M_Nh}")

            factor = cholesky(_norm_matrix)

            k = 0
            while k <= 4:
                try:
                    self.M_snapshots_matrix = factor.L().transpose().dot(factor.apply_P(self.M_snapshots_matrix[:, ::2**k]))
                    k = 99
                except Exception as e:
                    logger.warning(f"{e}\nImpossible to perform custom norm POD considering 1 snapshot every {2**k}!")
                    k += 1

            if k < 99:
                raise ValueError("Impossible to perform custom norm POD; the snapshots matrix is too big!")

        if PCA is not None:
            # 200 is set as maximal number of components --> change if necessary !!
            pca = PCA(n_components=min([200, self.M_snapshots_matrix.shape[0], self.M_snapshots_matrix.shape[1]]),
                      svd_solver='auto', random_state=0).fit(self.M_snapshots_matrix.T)
            U, s = pca.components_.T, pca.singular_values_
        else:
            # OLD CODE (deterministic)
            U, s, _ = np.linalg.svd(self.M_snapshots_matrix, full_matrices=False)

        if _norm_matrix is not None:
            U = factor.apply_Pt(factor.solve_Lt(U, use_LDLt_decomposition=False))

        total_energy = np.dot(s, np.transpose(s))
        logger.debug(f"The total energy of the field is {total_energy:.4e}")

        self.M_N = 0
        cumulative_energy = 0.0
        record_cumulative_energy = np.ones(self.M_ns)

        while cumulative_energy / total_energy < 1. - _tol ** 2 and self.M_N < self.M_ns:
            logger.debug(f"N = {self.M_N} -- "
                         f"Relative cumulative energy: {(cumulative_energy / total_energy):.4e}; "
                         f"Current SV {s[self.M_N]:.4e}")

            record_cumulative_energy[self.M_N] = cumulative_energy
            cumulative_energy = cumulative_energy + s[self.M_N] * s[self.M_N]  # add the energy of next basis
            self.M_N += 1  # add a basis in the count

        logger.info(f"Final N is {self.M_N}")

        self.M_basis = U[:, :self.M_N]
        self.M_basis[np.abs(self.M_basis) < 1e-14] = 0.0
        self.M_singular_values = s[:self.M_N]

        logger.debug(f"POD elapsed time: {(time.time() - start):.6f} s")

        return self.M_basis


__all__ = [
    "ProperOrthogonalDecomposition"
]