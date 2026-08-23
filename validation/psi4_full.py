import time
import sys
import logging
from pathlib import Path

import numpy as np
import psi4

from scratch_code_fock.mol_basis_builder import Molecule
from scratch_code_fock.wfn import WaveFunction


logger = logging.Logger("psi4_full")

log_path = Path(__file__).parent.resolve() / "psi4_full.log"

logger.addHandler(logging.FileHandler(log_path, mode='w'))
logger.addHandler(logging.StreamHandler(sys.stdout))


def my_full_pipeline(xyz_str, basis_name, type):
    mol = Molecule(xyz_str)

    wfn = WaveFunction(mol, basis_name)

    S, T, V, I = wfn.calc_integrals()

    if type == "rhf":
        E0 = wfn.calc_rhf_energy(verbose=0)
    elif type == "uhf":
        E0 = wfn.calc_uhf_energy(verbose=0)
    elif type == "mp2":
        E0 = wfn.calc_mp2_energy(verbose=0)
    else:
        raise ValueError(f"Invalid type: {type}")

    MC = wfn.calc_mulliken_charges()
    
    return {"E0" :E0, "MC": MC, "S": S, "T": T, "V": V, "I": I}


def psi4_full_pipeline(xyz_str, basis_name, type):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()

    psi4.core.set_output_file('output.dat', False)
    psi4.set_memory(int(5e8))
    psi4.set_options({'basis': basis_name, 'puream': 0, 'scf_type': 'pk'})

    if type == "rhf":
        calc = "scf"
    elif type == "uhf":
        psi4.set_options({'reference': 'uhf'})
        calc = "scf"
    elif type == "mp2":
        psi4.set_options({'mp2_type': 'conv'})
        calc = "mp2"
    else:
        raise ValueError(f"Invalid type: {type}")

    xyz_str_no_sym = xyz_str + "\n    symmetry c1\n    no_reorient\n    no_com\n"

    mol = psi4.core.Molecule.from_string(xyz_str_no_sym)
    mol.update_geometry()

    wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
    mints = psi4.core.MintsHelper(wfn.basisset())

    S = np.asarray(mints.ao_overlap())
    T = np.asarray(mints.ao_kinetic())
    V = np.asarray(mints.ao_potential())
    I = np.asarray(mints.ao_eri())

    E0, wfn = psi4.energy(calc, molecule=mol, return_wfn=True)
    psi4.oeprop(wfn, 'MULLIKEN_CHARGES')

    MC = np.array(wfn.atomic_point_charges())
    return {"E0" :E0, "MC": MC, "S": S, "T": T, "V": V, "I": I}


def compare(name, a, b):
    allclose = np.allclose(a, b)
    max_err = np.max(np.abs(a - b))
    rms_err = np.sqrt(np.mean((a - b) ** 2))

    logger.info(f"Check {name:8} | Allclose: {str(allclose):5} | Max error: {max_err:.3e} | RMS error: {rms_err:.3e}")


def validate(basis_name, xyz, type, name):
    logger.info("=" * 30)
    logger.info(f"Test: {name}; Basis set {basis_name}")
    logger.info("=" * 30)

    start_time = time.perf_counter()

    my_result = my_full_pipeline(xyz, basis_name, type)

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    logger.info(f"My implementation runtime: {execution_time:.6f} seconds")

    start_time = time.perf_counter()

    psi4_result = psi4_full_pipeline(xyz, basis_name, type)

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    logger.info(f"Psi4 runtime: {execution_time:.6f} seconds")

    logger.info(f"My SCF energy:   {my_result['E0']}")
    logger.info(f"Psi4 SCF energy: {psi4_result['E0']}")

    for quantity in my_result.keys():
        compare(quantity, my_result[quantity], psi4_result[quantity])


def create_fm_xyz(charge, mult):
    return f"""
{charge} {mult}

C    0.000000    0.000000    0.000000
O    0.000000    0.000000    1.203000
H    0.000000    0.934000   -0.582000
H    0.000000   -0.934000   -0.582000
"""

def main():
    basis_names = ["sto-3g", "6-31g"]

    for basis_name in basis_names:
        validate(basis_name, create_fm_xyz(0, 1), "rhf", "RHF")
        validate(basis_name, create_fm_xyz(0, 1), "uhf", "UHF singlet")
        validate(basis_name, create_fm_xyz(0, 3), "uhf", "UHF triplet")
        validate(basis_name, create_fm_xyz(1, 2), "uhf", "UHF charged doublet")
        validate(basis_name, create_fm_xyz(0, 1), "mp2", "MP2")


if __name__ == "__main__":
    main()