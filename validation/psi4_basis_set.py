import psi4
import numpy as np
from scratch_code_fock.mol_basis_builder import Molecule

import sys
import logging
from pathlib import Path

logger = logging.Logger("")

log_path = Path(__file__).parent.resolve() / "psi4_basis_set.log"

logger.addHandler(logging.FileHandler(log_path, mode='w'))
logger.addHandler(logging.StreamHandler(sys.stdout))


def test_basis_set(xyz_str, basis_name):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()

    psi4_mol = psi4.core.Molecule.from_string(xyz_str)
    psi4.set_options({'basis': basis_name, 'puream': 0}, verbose=0)

    wfn = psi4.core.Wavefunction.build(psi4_mol, psi4.core.get_global_option('basis'), quiet=True)
    psi4_basis = wfn.basisset()
    
    my_mol = Molecule(xyz_str)
    my_basis = my_mol.build_basis(basis_name)

    logger.info("Psi4 total primitives")
    total_prim = 0
    for i in range(psi4_basis.nshell()):
        sh = psi4_basis.shell(i)
        total_prim += int(sh.nprimitive)
    logger.info(total_prim)

    total_prim = 0
    logger.info("My total primitives")
    for i in range(my_basis.nshell()):
        sh = my_basis.shell(i)
        total_prim += int(sh.nprimitive)
    logger.info(total_prim)

    # Check total number of shells
    if psi4_basis.nshell() != my_basis.nshell():
        logger.info(f"\n[MISMATCH] Shell count differs for {basis_name}!")
        logger.info(f"Psi4 has {psi4_basis.nshell()} shells, you have {my_basis.nshell()} shells.")
        return False

    for n in range(psi4_basis.nshell()):
        p4_shell = psi4_basis.shell(n)
        my_shell = my_basis.shell(n)
        
        am_map = {0: 'S', 1: 'P', 2: 'D', 3: 'F'}
        sh_type = am_map.get(p4_shell.am, f"L={p4_shell.am}")

        # Check Angular Momentum
        if p4_shell.am != my_shell.am:
            logger.info(f"\n[MISMATCH] Shell {n} Angular Momentum mismatch!")
            logger.info(f"Psi4: {p4_shell.am} ({sh_type}), Mine: {my_shell.am}")
            return False

        # Check Primitive Count
        if p4_shell.nprimitive != my_shell.nprimitive:
            logger.info(f"\n[MISMATCH] Shell {n} ({sh_type}) Primitive Count mismatch!")
            logger.info(f"Atom Index: {my_shell.atom_index}")
            logger.info(f"Psi4 primitive count: {p4_shell.nprimitive}")
            logger.info(f"My primitive count: {my_shell.nprimitive}")
            logger.info(f"Psi4 raw exponents: {[p4_shell.exp(i) for i in range(p4_shell.nprimitive)]}")
            logger.info(f"My raw exponents: {my_shell.exponents}")
            return False

        # Check individual Primitives within this shell
        for i in range(p4_shell.nprimitive):
            p4_exp = p4_shell.exp(i)
            p4_coef = p4_shell.coef(i)
            
            my_exp = my_shell.exponents[i]
            my_coef = my_shell.effective_coef[i]

            if not np.isclose(p4_exp, my_exp, rtol=1e-5, atol=1e-8):
                logger.info(f"\n[MISMATCH] Shell {n} ({sh_type}), Primitive {i} Exponent mismatch!")
                logger.info(f"Psi4: {p4_exp}")
                logger.info(f"Mine: {my_exp}")
                return False

            if not np.isclose(p4_coef, my_coef, rtol=1e-5, atol=1e-8):
                logger.info(f"\n[MISMATCH] Shell {n} ({sh_type}), Primitive {i} Coefficient mismatch!")
                logger.info(f"Psi4: {p4_coef}")
                logger.info(f"Mine: {my_coef}")
                return False
    return True

if __name__ == "__main__":
    formaldehyde_xyz = """
    C    0.000000    0.000000    0.000000
    O    0.000000    0.000000    1.203000
    H    0.000000    0.934000   -0.582000
    H    0.000000   -0.934000   -0.582000
    """

    basis_names = ["sto-3g", "6-31g", "6-31g**", "cc-pvdz"]

    for basis_name in basis_names:
        logger.info("=" * 50)
        logger.info(f"Testing {basis_name} ...")
        logger.info("=" * 50)

        if test_basis_set(formaldehyde_xyz, basis_name):
            logger.info(f"Basis set {basis_name} passed !!!")
        else:
            logger.info(f"Basis set {basis_name} failed completely.")
