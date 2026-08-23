import numpy as np


def diis(e_list: list) -> np.ndarray:
    """
    Takes in a list of np.ndarray objects with matching shapes as error vectors.
    Construct the DIIS matrix and solve the Pulay equations for the coefficients.
    Returns a coefficient vector matching the arrays in the list.
    """
    size = len(e_list)

    B = np.full((size + 1, size + 1), -1.0)
    B[-1, -1] = 0.0

    flattened_e = [e.ravel() for e in e_list]
    e_array = np.vstack(flattened_e)

    # Compute pairwise dot products directly into B
    B[:size, :size] = e_array @ e_array.T

    rhs = np.zeros(size + 1)
    rhs[-1] = -1.0

    c = np.linalg.solve(B, rhs)
    # Return only the coefficients corresponding to the error vectors
    return c[:size]


def sym_ortho(S):
    # Symmetric orthogonalization
    # s = U.T @ S @ U
    # X = S^-0.5 = U @ s^-0.5 @ U.T
    s, U = np.linalg.eigh(S)
    s_inv_sqrt = 1.0 / np.sqrt(s)
    # Reset the inf values to 0
    s_inv_sqrt[np.isinf(s_inv_sqrt)] = 0.0
    X = U @ np.diag(s_inv_sqrt) @ U.T

    return X


def solve_F(F, X, n_occ):
    F_p = X.T @ F @ X
    eps, C_p = np.linalg.eigh(F_p)

    # C_p.shape = (K: original AO, K: new AO)
    # Truncate C to only keep occupied columns
    C = X @ C_p
    C_occ = C[:, :n_occ]

    # 2. Compute new Density matrix D
    D = C_occ @ C_occ.T
    return eps, C, D
