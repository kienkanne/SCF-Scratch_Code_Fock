import numpy as np
from scratch_code_fock.mol_basis_builder import Molecule
from scratch_code_fock.scf_utils import diis, sym_ortho, solve_F_get_D


def uhf(mol: Molecule, S, T, V, I,
                   max_iter=100, e_conv=1e-6, startup_iter=5, grad_max=1e-6, grad_rms=1e-6, verbose=0):

    n_alpha = mol.n_alpha
    n_beta = mol.n_beta
    V_nn = mol.V_nn

    X = sym_ortho(S)

    H = T + V
    # Initial guess, no electron interaction
    Fa = Fb = H
    E0_last = np.inf

    Fa_list = []
    Fb_list = []
    # Combined
    errab_list = []


    for i in range(1, max_iter+1):        
        if (i == startup_iter + 1) and verbose >= 2:
            print ("DIIS turned on!")

        # 1. Diagonalize current Fock matrix F
        Da = solve_F_get_D(Fa, X, n_alpha)
        Db = solve_F_get_D(Fb, X, n_beta)

        # 3. Compute energy
        E0 = 0.5 * np.sum((Da + Db) * H + Da * Fa + Db * Fb) + V_nn
        dE = E0 - E0_last

        # 4. Compute new F and error vector and append to list
        Ja = np.einsum('pqrs,rs->pq', I, Da, optimize=True)
        Ka = np.einsum('prqs,rs->pq', I, Da, optimize=True)
        Jb = np.einsum('pqrs,rs->pq', I, Db, optimize=True)
        Kb = np.einsum('prqs,rs->pq', I, Db, optimize=True)

        # Update Fa, Fb
        Fa = H + Ja + Jb - Ka
        Fb = H + Ja + Jb - Kb

        Fa_list.append(Fa)
        Fb_list.append(Fb)

        erra_vector = Fa @ Da @ S - S @ Da @ Fa
        errb_vector = Fb @ Db @ S - S @ Db @ Fb

        errab_vector = np.concat((erra_vector, errb_vector))
        errab_list.append(erra_vector)

        max_error = np.max(np.abs(errab_vector))
        rms_error = np.sqrt(np.mean(errab_vector**2))

        if verbose >=2:
            print(f"SCF Iteration {i:3d}: | Energy = {E0:.8f} | dE = {dE: 1.5E} | Max_E = {max_error: 1.5E} | RMS_E = {rms_error: 1.5E}")

        # 5. Convergence check
        if np.abs(dE) < e_conv and max_error < grad_max and rms_error < grad_rms:
            if verbose >= 1:
                print("SCF Converged!")
                print('Final RHF Energy: %.10f a.u.' % E0)
            
            return E0, Da, Db

        E0_last = E0

        if i < startup_iter:
            pass
        else:
            # Use a slice of the history to limit DIIS memory
            # Avoid old iterations from polluting the subspace
            history = slice(-8, None) 

            # Shared coefficients for both a and b
            coeffs = diis(errab_list[history])
            
            # Extrapolate F
            Fa = np.einsum('i,ijk->jk', coeffs, np.array(Fa_list[history]))
            Fb = np.einsum('i,ijk->jk', coeffs, np.array(Fb_list[history]))

    raise Exception("Maximum number of SCF iterations exceeded.")