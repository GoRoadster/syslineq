import numpy as np
import cmath


def cholesky_decomposition(M, d_type=np.float64):
    """
    :param M: Symmetric, positive definite Matrix
    :param d_type: Data type for the the elements of the output
    :return: Lower triangular (L), Transpose of lower triangular (L')
    :Cholesky Decomposition Algorithm:

    L[j][j] = sqrt(M[j][j]-sum(L[j][k]**2)); k=0...j-1
    L[i][j] = (M[i][j]-sum(L[i][k]*L[j][k])*1/L[i][j]; k=0...j-1

    """

    if not np.all(np.linalg.eigvals(M) > 0):
        raise Exception("Matrix is not positive definite")

    L = np.zeros_like(M)
    for i in range(len(M)):
        for j in range(i + 1):
            _sum = sum((L[i][k] * L[j][k]) for k in range(j))
            if i == j:
                L[j][j] = d_type(cmath.sqrt(M[j][j] - _sum).real)
            else:
                L[i][j] = d_type(1 / L[j][j] * (M[i][j] - _sum))

    return L
