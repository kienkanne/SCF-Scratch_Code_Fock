import numpy as np
from scratch_code_fock.mol_basis_builder import Molecule
from scratch_code_fock.scf_utils import diis, sym_ortho, solve_F


def rhf(mol: Molecule, S, T, V, I,
                   max_iter=100, e_conv=1e-6, startup_iter=5, grad_max=1e-6, grad_rms=1e-6, verbose=0):

    ndocc = mol.ndocc
    V_nn = mol.V_nn

    X = sym_ortho(S)

    H = T + V
    # Initial guess, no electron interaction
    F = H
    E0_last = np.inf

    F_list = []
    err_list = []

    for i in range(1, max_iter+1):        
        if (i == startup_iter + 1) and verbose >= 2:
            print ("DIIS turned on!")

        # 1. Diagonalize current Fock matrix F
        eps, C, D = solve_F(F, X, ndocc)

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

        if verbose >=2:
            print(f"SCF Iteration {i:3d}: | Energy = {E0:.8f} | dE = {dE: 1.5E} | Max_E = {max_error: 1.5E} | RMS_E = {rms_error: 1.5E}")

        # 5. Convergence check
        if np.abs(dE) < e_conv and max_error < grad_max and rms_error < grad_rms:
            if verbose >= 1:
                print("SCF Converged!")
                print('Final RHF Energy: %.10f a.u.' % E0)
            
            return {"E0": E0, "eps": eps, "C": C, "D": D}

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

    raise Exception("Maximum number of SCF iterations exceeded.")