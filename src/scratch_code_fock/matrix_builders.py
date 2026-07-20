import numpy as np
from scratch_code_fock.mol_basis_builder import ANGSTROM_TO_BOHR, ATOMIC_NUMBERS
from scratch_code_fock.mol_basis_builder import Molecule, Basis # For type hints only


def angular_combinations(n: int):
    coms = []
    # Psi4 orders Cartesian components from highest z-power down to x
    for nx in range(n, -1, -1):
        for ny in range(n - nx, -1, -1):
            nz = n - nx - ny
            coms.append((nx, ny, nz))
    return coms


def build_S_T_V(mol: Molecule, basis: Basis):
    from scratch_code_fock.integral_solvers import get_overlap_integrals, get_kinetic_integrals, get_NAIs
    n_ao = basis.nao

    S = np.zeros((n_ao,n_ao))
    T = np.zeros((n_ao,n_ao))
    V = np.zeros((n_ao,n_ao))

    # 1st loop: over shells
    for a in range(basis.nshell()):
        for b in range(basis.nshell()):

            ashell = basis.shell(a) 
            bshell = basis.shell(b)

            AMa = ashell.am
            AMb = bshell.am

            # Basis function index where the shell begins
            a_idx = ashell.function_index  
            b_idx = bshell.function_index

            a_coms = angular_combinations(AMa)
            b_coms = angular_combinations(AMb)

            counta = len(a_coms)
            countb = len(b_coms)
            
            # Ensure 2-fold symmetry to avoid duplicate computation
            if b < a:
                S[a_idx:a_idx+counta, b_idx:b_idx+countb] = S[b_idx:b_idx+countb, a_idx:a_idx+counta].transpose((1, 0))
                T[a_idx:a_idx+counta, b_idx:b_idx+countb] = T[b_idx:b_idx+countb, a_idx:a_idx+counta].transpose((1, 0))
                V[a_idx:a_idx+counta, b_idx:b_idx+countb] = V[b_idx:b_idx+countb, a_idx:a_idx+counta].transpose((1, 0))
                continue

            A = np.array(ashell.center)
            B = np.array(bshell.center)

            nprima = ashell.nprimitive 
            nprimb = bshell.nprimitive

            # 2nd loop: over the primitives within a shell
            for a_prim in range(nprima):  
                for b_prim in range(nprimb):
                    expa = ashell.exponents[a_prim] 
                    expb = bshell.exponents[b_prim]

                    coefa = ashell.effective_coef[a_prim]
                    coefb = bshell.effective_coef[b_prim]
                    
                    zeta = expa + expb
                    xi = (expa * expb) / zeta
                    P = (expa * A + expb * B) / zeta
                    PA = P - A
                    PB = P - B
                    AB = A - B

                    S_base = (np.pi / zeta)**(3 / 2) * np.exp(-xi * (AB[0]**2 + AB[1]**2 + AB[2]**2))

                    V_base = 2 * (zeta / np.pi)**0.5 * S_base

                    # Calculates an additional layer for kinetic terms
                    SI = get_overlap_integrals(PA, PB, zeta, AMa, AMb)
                    TI = get_kinetic_integrals(PA, PB, expa, expb, AMa, AMb, SI)

                    # 3rd loop: over the all angular combinations
                    for counta, a_com in enumerate(a_coms):
                        for countb, b_com in enumerate(b_coms):
                            x, y, z = 0, 1, 2
                            ax, ay, az = a_com
                            bx, by, bz = b_com
                            
                            # 3D Overlap is purely multiplicative: Sx * Sy * Sz
                            S[a_idx + counta, b_idx + countb] += S_base \
                                                            * coefa * coefb \
                                                            * SI[x, ax, bx] \
                                                            * SI[y, ay, by] \
                                                            * SI[z, az, bz]

                            # 3D Kinetic Energy requires the Laplacian cross-terms:
                            # Tx*Sy*Sz + Sx*Ty*Sz + Sx*Sy*Tz
                            t_term = (TI[x, ax, bx] * SI[y, ay, by] * SI[z, az, bz]) + \
                                    (SI[x, ax, bx] * TI[y, ay, by] * SI[z, az, bz]) + \
                                    (SI[x, ax, bx] * SI[y, ay, by] * TI[z, az, bz])

                            T[a_idx + counta, b_idx + countb] += S_base \
                                                            * coefa * coefb \
                                                            * t_term
                            
                            for atom, C in zip(mol.atoms, mol.coords * ANGSTROM_TO_BOHR):
                                Z = ATOMIC_NUMBERS[atom.upper()]
                                PC = P - C

                                VI = get_NAIs(PA, PB, PC, zeta, AMa, AMb)

                                # Multiply by -Z_C, do not cube VI, and strictly query m=0
                                V[a_idx + counta, b_idx + countb] += (-Z) * V_base \
                                                                    * coefa \
                                                                    * coefb \
                                                                    * VI[ax, ay, az, bx, by, bz, 0]

    return S, T, V                      


def build_ERI(mol: Molecule, basis: Basis):
    from scratch_code_fock.integral_solvers import get_ERIs

    n_ao = basis.nao
    ERI = np.zeros((n_ao,n_ao,n_ao,n_ao))

    # 1st loop: over shells
    for a in range(basis.nshell()):
        for b in range(basis.nshell()):
            for c in range(basis.nshell()):
                for d in range(basis.nshell()):

                    ashell = basis.shell(a) 
                    bshell = basis.shell(b)
                    cshell = basis.shell(c)
                    dshell = basis.shell(d)

                    AMa = ashell.am
                    AMb = bshell.am
                    AMc = cshell.am
                    AMd = dshell.am

                    # Basis function index where the shell begins
                    a_idx = ashell.function_index  
                    b_idx = bshell.function_index
                    c_idx = cshell.function_index
                    d_idx = dshell.function_index

                    a_coms = angular_combinations(AMa)
                    b_coms = angular_combinations(AMb)
                    c_coms = angular_combinations(AMc)
                    d_coms = angular_combinations(AMd)

                    counta = len(a_coms)
                    countb = len(b_coms)
                    countc = len(c_coms)
                    countd = len(d_coms)

                    # Ensure 8-fold symmetry to avoid duplicate computation
                    if d < c:
                        ERI[a_idx:a_idx+counta, b_idx:b_idx+countb, c_idx:c_idx+countc, d_idx:d_idx+countd] = 1 * \
                        ERI[a_idx:a_idx+counta, b_idx:b_idx+countb, d_idx:d_idx+countd, c_idx:c_idx+countc].transpose((0, 1, 3, 2))
                        continue

                    elif b < a:
                        ERI[a_idx:a_idx+counta, b_idx:b_idx+countb, c_idx:c_idx+countc, d_idx:d_idx+countd] = 1 * \
                        ERI[b_idx:b_idx+countb, a_idx:a_idx+counta, c_idx:c_idx+countc, d_idx:d_idx+countd].transpose((1, 0, 2, 3))
                        continue

                    elif (a, b) > (c, d):
                        ERI[a_idx:a_idx+counta, b_idx:b_idx+countb, c_idx:c_idx+countc, d_idx:d_idx+countd] = 1 * \
                        ERI[c_idx:c_idx+countc, d_idx:d_idx+countd, a_idx:a_idx+counta, b_idx:b_idx+countb].transpose((2, 3, 0, 1))
                        continue

                    A = np.array(ashell.center)
                    B = np.array(bshell.center)
                    C = np.array(cshell.center)
                    D = np.array(dshell.center)
            
                    nprima = ashell.nprimitive 
                    nprimb = bshell.nprimitive
                    nprimc = cshell.nprimitive
                    nprimd = dshell.nprimitive

                    # 2nd loop: over the primitives within a shell
                    for a_prim in range(nprima):  
                        for b_prim in range(nprimb):
                            for c_prim in range(nprimc):
                                for d_prim in range(nprimd):
                                    expa = ashell.exponents[a_prim] 
                                    expb = bshell.exponents[b_prim]
                                    expc = cshell.exponents[c_prim] 
                                    expd = dshell.exponents[d_prim]

                                    coefa = ashell.effective_coef[a_prim]
                                    coefb = bshell.effective_coef[b_prim]
                                    coefc = cshell.effective_coef[c_prim]
                                    coefd = dshell.effective_coef[d_prim]

                                    ERI_raw = get_ERIs(A, B, C, D, expa, expb, expc, expd, AMa, AMb, AMc, AMd)

                                    zeta = expa + expb
                                    xi = (expa * expb) / zeta
                                    eta = expc + expd
                                    xi_cd = (expc * expd) / eta # NOTE: In the original paper there is no notation for xi_cd

                                    S_base_ab = (np.pi / zeta)**(3 / 2) * np.exp(-xi * np.sum((A-B)**2))
                                    S_base_cd = (np.pi / eta)**(3 / 2) * np.exp(-xi_cd * np.sum((C-D)**2))

                                    rho = (zeta * eta) / (zeta + eta)

                                    ERI_base = 2 * (rho / np.pi)**0.5 * S_base_ab * S_base_cd

                                    # 3rd loop: over the all angular combinations
                                    for counta, a_com in enumerate(a_coms):
                                        for countb, b_com in enumerate(b_coms):
                                            for countc, c_com in enumerate(c_coms):
                                                for countd, d_com in enumerate(d_coms):
                                                    ax, ay, az = a_com
                                                    bx, by, bz = b_com
                                                    cx, cy, cz = c_com
                                                    dx, dy, dz = d_com
                                                    
                                                    ERI_term = ERI_raw[ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, 0]

                                                    ERI[a_idx + counta, b_idx + countb, c_idx + countc, d_idx + countd] += ERI_base \
                                                                                                                        * coefa * coefb * coefc * coefd \
                                                                                                                        * ERI_term
    return ERI
