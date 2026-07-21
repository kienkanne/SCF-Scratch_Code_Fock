import time
import numpy as np
import psi4

from scratch_code_fock.mol_basis_builder import Molecule
from scratch_code_fock.matrix_builders import build_S_T_V, build_ERI
from scratch_code_fock.roothan_solver import roothan_solver


def my_full_pipeline(xyz_str, basis_name):
    mol = Molecule(xyz_str)

    mol.get_nuclear_repulsion()

    basis = mol.build_basis(basis_name)

    S, T, V = build_S_T_V(mol, basis)
    I = build_ERI(mol, basis)

    scf_energy, mulliken_charges = roothan_solver(mol, S, T, V, I, basis=basis, return_mulliken=True, verbose=0)
    return scf_energy, mulliken_charges, S, T, V, I


def psi4_full_pipeline(xyz_str, basis_name):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()

    psi4.core.set_output_file('output.dat', False)
    psi4.set_memory(int(5e8))
    psi4.set_options({'basis': basis_name, 'puream': 0, 'scf_type': 'pk'})

    mol = psi4.core.Molecule.from_string(xyz_str)
    mol.update_geometry()

    wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
    mints = psi4.core.MintsHelper(wfn.basisset())

    S = np.asarray(mints.ao_overlap())
    T = np.asarray(mints.ao_kinetic())
    V = np.asarray(mints.ao_potential())
    I = np.asarray(mints.ao_eri())

    scf_energy, wfn = psi4.energy('SCF', molecule=mol, return_wfn=True)
    psi4.oeprop(wfn, 'MULLIKEN_CHARGES')

    mulliken_charges = np.array(wfn.atomic_point_charges())
    return scf_energy, mulliken_charges, S, T, V, I


def compare(name, a, b):
    allclose = np.allclose(a, b)
    max_err = np.max(np.abs(a - b))
    rms_err = np.sqrt(np.mean((a - b) ** 2))

    print(f"Check {name:8} | Allclose: {str(allclose):5} | Max error: {max_err:.3e} | RMS error: {rms_err:.3e}")


def main():
    formaldehyde_xyz = """
    C    0.000000    0.000000    0.000000
    O    0.000000    0.000000    1.203000
    H    0.000000    0.934000   -0.582000
    H    0.000000   -0.934000   -0.582000
    """
    basis_names = ["sto-3g", "6-31g", "6-31g**", "cc-pvdz"]

    for basis_name in basis_names:
        print ("=" * 30)
        print (f"Testing basis set {basis_name}")
        print ("=" * 30)

        start_time = time.perf_counter()

        my_scf_energy, my_mulliken_charges, my_S, my_T, my_V, my_I = my_full_pipeline(formaldehyde_xyz, basis_name)

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        print(f"My implementation runtime: {execution_time:.6f} seconds")

        start_time = time.perf_counter()

        psi4_scf_energy, psi4_mulliken_charges, psi4_S, psi4_T, psi4_V, psi4_I = psi4_full_pipeline(formaldehyde_xyz, basis_name)

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        print(f"Psi4 runtime: {execution_time:.6f} seconds")

        print (f"My SCF energy:   {my_scf_energy}")
        print (f"Psi4 SCF energy: {psi4_scf_energy}")

        outputs = {
            "Energy": (my_scf_energy, psi4_scf_energy),
            "Mulliken": (my_mulliken_charges, psi4_mulliken_charges),
            "S": (my_S, psi4_S),
            "T": (my_T, psi4_T),
            "V": (my_V, psi4_V),
            "I": (my_I, psi4_I)
        }

        for name, (my_data, psi4_data) in outputs.items():
            compare(name, my_data, psi4_data)


if __name__ == "__main__":
    main()