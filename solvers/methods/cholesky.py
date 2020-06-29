from solvers.utils.decomposition import cholesky_decomposition
from copy import deepcopy
import numpy as np


def cholesky_LU(A, coeff_b):
    """
    :param A: positive definite matrix otherwise it fails
    :param coeff_b: Array B in A * x = B
    :return: computed x values[ndarray]

    Uses same principle in LU decomposition
    For A * x = B
    1. It decomposes A into L, L' | A = L * L'
    2. From (1) => L * L' * x = B. Notation: L' * x = y
    3. From (1), (2) =>
                        L * y = B (I)
                        L' * x = y (II)
    4. First forward solve (I), than back solve (II)
    """
    try:
        L = cholesky_decomposition(A)

        assert np.dot(L, L.T) == A

        y = __forward_substitution(L, coeff_b)
        x = __back_substitution(L.T, y)

        return x

    except Exception as e:
        raise e


def __forward_substitution(lower_t, coeff, d_type=np.float64):
    """
    :param lower_t: Lower triangular matrix
    :param coeff: Array B in L * y = B
    :param d_type: Elements data type
    :return: computed y values for back substitution
    """
    n = len(lower_t)
    lt = deepcopy(lower_t)
    res = np.zeros(shape=(n, 1))
    for i in range(n):
        terms_sum = sum(lt[i][j] * lt[j][j] for j in range(i))
        res[i] = d_type((coeff[i][0] - terms_sum) / lt[i][i])
        lt[i][i] = res[i]
    return res


def __back_substitution(upper_t, coeff, d_type=np.float64):
    """
    :param upper_t: Upper triangular matrix
    :param coeff: Array y in L' * x = y
    :param d_type: Elements data type
    :return: computed x values, solution
    """
    n = len(upper_t)
    ut = deepcopy(upper_t)
    res = np.zeros(shape=(n, 1))
    for i in range(n - 1, -1, -1):
        terms_sum = sum(ut[i][j] * ut[j][j] for j in range(n - 1, i, -1))
        res[i] = d_type((coeff[i][0] - terms_sum) / ut[i][i])
        ut[i][i] = res[i]
    return res
