from tabnanny import verbose

from scratch_code_fock.mol_basis_builder import Molecule
from scratch_code_fock.matrix_builders import build_S_T_V, build_ERI
from scratch_code_fock.rhf import rhf
from scratch_code_fock.uhf import uhf
import numpy as np


class WaveFunction():
    def __init__(self, mol: Molecule, basis_name: str):
        self.mol = mol
        self.basis = mol.build_basis(basis_name)
        self.S = self.T = self.V = self.I = None

    def calc_integrals(self):
        self.S, self.T, self.V = build_S_T_V(self.mol, self.basis)
        self.I = build_ERI(self.mol, self.basis)

        return self.S, self.T, self.V, self.I

    def calc_energy(self, **kwargs):
        if self.S is None or self.T is None or self.V is None:
            self.S, self.T, self.V = build_S_T_V(self.mol, self.basis)
        if self.I is None:
            self.I = build_ERI(self.mol, self.basis)

        if self.mol.multiplicity != 1:
            self.energy, self.Da, self.Db = uhf(self.mol, self.S, self.T, self.V, self.I, **kwargs)
            self.D = 0.5 * (self.Da + self.Db)
        else:
            self.energy, self.D = rhf(self.mol, self.S, self.T, self.V, self.I, **kwargs)

        return self.energy

    def calc_mulliken_charges(self):
        from scratch_code_fock.mol_basis_builder import ATOMIC_NUMBERS
        if getattr(self, "D", None) is None:
            self.calc_energy(verbose=0)

        P = 2.0 * (self.D @ self.S)

        ao_populations = np.diag(P)

        num_atoms = len(self.mol.atoms)
        atom_electrons = np.zeros(num_atoms)

        nbf = self.S.shape[0]
        for ao_idx in range(nbf):
            atom_center = self.basis.function_to_center(ao_idx)
            atom_electrons[atom_center] += ao_populations[ao_idx]

        mulliken_charges = np.zeros_like(atom_electrons)
        for i in range(num_atoms):
            Z = ATOMIC_NUMBERS[self.mol.atoms[i].upper()]
            mulliken_charges[i] = Z - atom_electrons[i]

        return mulliken_charges