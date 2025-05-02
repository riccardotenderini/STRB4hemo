#!\usr\bin\env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 10:23:20 2019
@author: Niccolo' Dal Santo
@email : niccolo.dalsanto@epfl.ch
"""
import os

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix, coo_matrix, issparse
import scipy.sparse.linalg

import logging.config

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.normpath('../../log.cfg'))
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def is_empty(_matrix):
    """ Utility method to check whether a numpy matrix (or vector) is empty or not

    :param _matrix: matrix to be checked
    :type _matrix: numpy.ndarray
    :return: True if the matrix is empty, False otherwise
    :rtype: bool
    """

    return not _matrix.size or not _matrix.ndim


def save_array(_array, _file_name):
    """Utility method, which allows to save 1D/2D numpy arrays in text files. If the path to the text file does not exist,
    it displays an error message, but it does not raise any error.

    :param _array: array to be saved
    :type _array: numpy.ndarray
    :param _file_name: path to the file where the array has to be saved, provided that the path is a valid path
    :type _file_name: str
    """

    try:
        if os.path.splitext(_file_name)[1] in {'.txt', ''}:
            np.savetxt(_file_name, _array.squeeze(), fmt='%.10g')
        elif os.path.splitext(_file_name)[1] == '.npy':
            np.save(_file_name, _array)
        else:
            logger.warning(f"Non-default file extension for the file {_file_name}. "
                           f"Saving it as a text file.")
            np.savetxt(_file_name, _array.squeeze(), fmt='%.10g')
    except (OSError, IOError, FileNotFoundError, TypeError) as e:
        logger.critical(e)
        raise ValueError(f"Impossible to save the array at {_file_name}.")

    return


def read_matrix(name, matrixtype=csc_matrix, shapem=None, tol=None):
    """ Utility method, which allows to read a sparse matrix from a text file

    :param name: path to the .txt file where te matrix is stored
    :type name: str
    :param matrixtype: type of the sparse matrix. It defaults to scipy.sparse.csc_matrix
    :type scipy.sparse.lil_matrix or scipy.sparse.csr_matrix or scipy.sparse.csc_matrix or scipy.sparse.coo_matrix
    :param shapem: number of rows of the sparse matrix. If None, it is inferred automatically and errors may arise if
       the final rows are entirely made of zeros. It defaults to None
    :type shapem: int or NoneType
    :param tol: tolerance of zero values. If None, no thresholding is applied. It defaults to None.
    :type tol: float or NoneType
    :return the sparse matrix read from the text file
    :rtype: scipy.sparse.lil_matrix or scipy.sparse.csr_matrix or scipy.sparse.csc_matrix or scipy.sparse.coo_matrix
    """

    try:
        assert matrixtype in {lil_matrix, csr_matrix, csc_matrix, coo_matrix}
    except AssertionError:
        logger.error(f"Invalid matrix type!")
        raise TypeError

    i, j, value = np.loadtxt(name).T

    # setting to 0 very small entries, in order to increase the sparsity and reduce memory load
    if tol is not None:
        idx_nonzero = np.where(np.abs(value) > tol)[0]
        i, j, value = i[idx_nonzero], j[idx_nonzero], value[idx_nonzero]

    # indices are relative to matlab so we subtract 1
    return matrixtype((value, (i.astype(int) - 1, j.astype(int) - 1)), shape=shapem)


def __check_vector(vec):
    """Check if vector shapes are suitable for inner product

    :param vec: vector to be checked
    :type vec: np.ndarray
    :return: checked vector
    :rtype: np.ndarray
    """

    vec = np.squeeze(np.array(vec))
    if len(vec.shape) == 1:
        vec = vec[None]
    elif len(vec.shape) > 2:
        raise ValueError(f"Invalid vector shape {vec.shape}")
    return vec


def mydot(vec1, vec2, norm_matrix=None):
    """It computes the inner product between 'vec1' and 'vec2', defined by the (positive definite) matrix 'norm_matrix'.
    If 'norm_matrix' is None (default), the standard inner product between 'vec1' and 'vec2' is returned.

    :param vec1: first vector(s)
    :type vec1: np.ndarray
    :param vec2: second vector(s)
    :type vec2: np.ndarray
    :param norm_matrix: positive definite matrix, defining the inner product. If None, it defaults to the identity.
    :type norm_matrix: scipy.sparse.csc_matrix or np.ndarray or NoneType
    :return: inner product between 'vec1' and 'vec2', defined by 'norm_matrix'
    :rtype: float
    """

    vec1 = __check_vector(vec1)
    vec2 = __check_vector(vec2)

    if norm_matrix is not None:
        res = np.einsum('ij,ij->i', vec1, norm_matrix.dot(vec2.T).T)
    else:
        res = np.einsum('ij,ij->i', vec1, vec2)

    res = np.squeeze(res)

    return res


def mynorm(vec, norm_matrix=None):
    """It computes the norm of 'vec', defined by the (positive definite) matrix 'norm_matrix'.
    If 'norm_matrix' is None (default), the Euclidean norm of 'vec' is returned.

    :param vec: vector(s)
    :type vec: np.ndarray
    :param norm_matrix: positive definite matrix, defining the norm. If None, it defaults to the identity.
    :type norm_matrix: scipy.sparse.csc_matrix or np.ndarray or NoneType
    :return: norm of 'vec', defined by 'norm_matrix'
    :rtype: double
    """
    return np.sqrt(mydot(vec, vec, norm_matrix))


def sparse_matrix_matrix_mul(mat1, mat2):
    """Computes the matrix multiplication between mat1 and mat2, assuming that mat1 is sparse and mat2 is full.
    mat1 can be either in 'raw' CO format or a scipy.sparse matrix.

     :param mat1: pre-multiplicative matrix, supposed to be sparse
     :type mat1: numpy.ndarray or scipy.sparse
     :param mat2: post-multiplicative matrix, supposed to be full
     :type mat2: numpy.ndarray
     :return: result of the sparse-full matrix multiplication mat1*mat2, given as a full matrix
     :rtype: numpy.ndarray
     """

    if type(mat1) is np.ndarray:  # assuming raw COO format
        Av = np.zeros(mat2.shape)
        nnz = mat1.shape[0]
        for i in range(nnz):
            Av[int(mat1[i, 0]), :] = Av[int(mat1[i, 0]), :] + mat1[i, 2] * mat2[int(mat1[i, 1]), :]
    elif issparse(mat1):
        Av = mat1.dot(mat2)
    else:
        logger.error(f"Error: impossible to perform sparse-full matrix-matrix multiplication")
        raise TypeError

    return Av


def sparse_matrix_vector_mul(mat, vec):
    """ Computes the matrix-vector multiplication between mat and vec, assuming that mat1 is sparse and vec is full.
    mat1 can be either in 'raw' CO format or a scipy.sparse matrix.

    :param mat: pre-multiplicative matrix, supposed to be sparse
    :type mat: numpy.ndarray or scipy.sparse
    :param vec: post-multiplicative vector, supposed to be full
    :type vec: numpy.ndarray
    :return: result of the sparse-full matrix-vector multiplication mat*vec, given as a full vector
    :rtype: numpy.ndarray
    """

    if type(mat) is np.ndarray:  # assuming raw COO format
        Av = np.zeros(vec.shape)
        nnz = mat.shape[0]
        for i in range(nnz):
            Av[int(mat[i, 0])] = Av[int(mat[i, 0])] + mat[i, 2] * vec[int(mat[i, 1])]
    elif issparse(mat):
        Av = mat.dot(vec)
    else:
        logger.error(f"Error: impossible to perform sparse-full matrix-vector multiplication")
        raise TypeError

    return Av


def sparse_to_full_matrix(mat, dims):
    """Method to convert a sparse matrix, given in COO format or as scipy.sparse matrix, to a full matrix,
    given its dimensions

    :param mat: input matrix, supposed to be sparse, to be converted to a full matrix
    :type mat: numpy.ndarray or scipy.sparse
    :param dims: dimensions of the final full matrix
    :type dims: tuple(int, int)
    :return: full version of the input matrix, of dimensions 'dims'
    :rtype: numpy.ndarray
    """

    if type(mat) is np.ndarray:  # assuming raw COO format
        result = np.zeros(dims)
        result[mat[:, 0].astype(np.int64), mat[:, 1].astype(np.int64)] = mat[:, 2]
    elif issparse(mat):
        result = mat.toarray()
    else:
        logger.error(f"Error: impossible to perform sparse-to-full matrix conversion")
        raise TypeError

    return result


def solve_sparse_system(mat, vec, direct=True):
    """
    Solve sparse linear system using either a direct method (scipy.sparse.linalg.spsolve), if ``direct`` is True,
    or an iterative method (MinRes from scipy.sparse.linalg.minres) if ``direct`` is False
    """

    assert issparse(mat) and len(mat.shape) == 2 and 1 <= len(vec.shape) <= 2

    if direct:
        sol = scipy.sparse.linalg.spsolve(mat, vec)

    else:
        maxiter, tol = 1000, 1e-10
        solver = lambda _mat, _vec: scipy.sparse.linalg.minres(_mat, _vec, maxiter=maxiter, tol=tol)

        if len(vec.shape) == 1:
            sol, _ = solver(mat, vec)
        else:
            if issparse(vec):
                vec = vec.tocsc().todense()
            sol = np.vstack([sol[0] if not sol[1] else np.nan * np.ones_like(sol[0])
                             for sol in (solver(mat, vec[:, i]) for i in range(vec.shape[1]))]).T
            if np.any(np.isnan(sol)):
                raise ValueError(f"Sparse linear system MinRes solver did not converge "
                                 f"for tolerance={tol} and maximal iterations number maxiter={maxiter}!")

    return sol.todense() if issparse(sol) else sol


def is_symmetric(m):
    """
    MODIFY
    """

    if m.shape[0] != m.shape[1]:
        raise ValueError('m must be a square matrix')

    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)

    r, c, v = m.row, m.col, m.data
    tril_no_diag = r > c
    triu_no_diag = c > r

    if triu_no_diag.sum() != tril_no_diag.sum():
        return False

    rl = r[tril_no_diag]
    cl = c[tril_no_diag]
    vl = v[tril_no_diag]
    ru = r[triu_no_diag]
    cu = c[triu_no_diag]
    vu = v[triu_no_diag]

    sortl = np.lexsort((cl, rl))
    sortu = np.lexsort((ru, cu))
    vl = vl[sortl]
    vu = vu[sortu]

    check = np.allclose(vl, vu)

    return check


def is_positive_definite(A, tol=1e-10):
    vals, _ = scipy.sparse.linalg.arpack.eigsh(A, k=1, which='SA')
    return np.all(vals > -tol)


__all__ = [
    "is_empty",
    "is_symmetric",
    "is_positive_definite",
    "save_array",
    "read_matrix",
    "mydot",
    "mynorm",
    "sparse_matrix_matrix_mul",
    "sparse_matrix_vector_mul",
    "sparse_to_full_matrix",
    "solve_sparse_system"
]
