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


def roothan_solver(mol, S, T, V, I):
    max_iter = mol.max_iter
    startup_iter = mol.startup_iter
    e_conv = mol.e_conv
    grad_max = mol.grad_max
    grad_rms = mol.grad_rms
    verbose = mol.verbose
    ndocc = mol.ndocc
    V_nn = mol.V_nn

    # Symmetric orthogonalization
    # s = U.T @ S @ U
    # X = S^-0.5 = U @ s^-0.5 @ U.T
    s, U = np.linalg.eigh(S)
    s_inv_sqrt = 1.0 / np.sqrt(s)
    # Reset the inf values to 0
    s_inv_sqrt[np.isinf(s_inv_sqrt)] = 0.0
    X = U @ np.diag(s_inv_sqrt) @ U.T

    H = T + V
    # Initial guess, no electron interaction
    F = H
    E0_last = np.inf

    F_list = []
    err_list = []

    for i in range(1, max_iter+1):
        if (i == max_iter + 1):
            raise Exception("Maximum number of SCF iterations exceeded.")
        
        if (i == startup_iter + 1) and verbose == 0:
            print ("DIIS turned on!")

        # 1. Diagonalize current Fock matrix F
        F_p = X.T @ F @ X
        e, C_p = np.linalg.eigh(F_p)

        # C_p.shape = (K: original AO, K: new AO)
        # Truncate C to only keep occupied columns
        C = X @ C_p
        C_docc = C[:, :ndocc]

        # 2. Compute new Density matrix D
        D = C_docc @ C_docc.T

        # 3. Compute energy
        E0 = np.sum(D * (H + F)) + V_nn
        dE = E0 - E0_last

        # 4. Compute new F and error vector and append to list
        J = np.einsum('pqrs,rs->pq', I, D, optimize=True)
        K = np.einsum('prqs,rs->pq', I, D, optimize=True)
        F = H + 2 * J - K

        F_list.append(F)

        err_vector = F @ D @ S - S @ D @ F
        err_list.append(err_vector)

        max_error = np.max(np.abs(err_vector))
        rms_error = np.sqrt(np.mean(err_vector**2))

        if verbose == 0:
            print(f"SCF Iteration {i:3d}: | Energy = {E0:4.16f} | dE = {dE: 1.5E} | Max_E = {max_error: 1.5E} | RMS_E = {rms_error: 1.5E}")

        # 5. Convergence check
        if np.abs(dE) < e_conv and max_error < grad_max and rms_error < grad_rms:
            if verbose <= 1:
                print("SCF Converged!")
                print('Final RHF Energy: %.10f a.u.' % E0)
            return E0

        E0_last = E0

        if i < startup_iter:
            pass
        else:
            # Use a slice of the history to limit DIIS memory
            # Avoid old iterations from polluting the subspace
            history = slice(-8, None) 
            coeffs = diis(err_list[history])
            
            # Extrapolate F
            F = np.einsum('i,ijk->jk', coeffs, np.array(F_list[history]))