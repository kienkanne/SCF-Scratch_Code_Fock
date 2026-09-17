# SCF: Scratch-Code-Fock

A Self-Consistent-Field (SCF) engine built entirely from first principles in pure NumPy, with RHF MP2 recently added. Every integral (overlap, kinetic, nuclear attraction, and electron repulsion) is evaluated via explicit Obara-Saika recursion, with no dependency on a quantum chemistry package for the integral evaluation itself. It features RHF/SCF references, SCF/MP2 methods (UHF MP2 not yet available), and Mulliken population analysis.

> **This is an educational, unoptimized, scratch-built implementation.**
> The focus throughout is mathematical transparency and correctness, not speed. There is no primitive screening, no integral batching, and the electron repulsion integral recursion is a genuinely brute-force 13-dimensional Obara-Saika evaluation. The design was inspired from the Psi4 quantum chemistry package.

The integrals module of this repository was invited to contribute at Psi4Numpy, and is currently under reviewed by Dr. Crawford in [**this**](https://github.com/psi4/psi4numpy/pull/143) pull request.

## Dependencies

```
psi4                   # Validation benchmark
numpy                  # Main architecture
scipy                  # Boys function, via scipy.special.gamma / gammainc
basis_set_exchange     # Basis set data (exponents, contraction coefficients)
```

## Usage

```python
from scratch_code_fock.mol_basis_builder import Molecule
from scratch_code_fock.wfn import WaveFunction

mol = Molecule("""
0 1
O    0.000000    0.000000   -0.143225
H    0.000000    1.638036    1.136548
H    0.000000   -1.638036    1.136548
""")

wfn = WaveFunction(mol, "sto-3g")

scf_energy = wfn.calc_rhf_energy()
mulliken_charges = wfn.calc_mulliken_charges()
```

The first line of the molecule specification is its charge and multiplicity, and coordinates are in angstroms. `WaveFunction` builds the AO basis and integrals on demand for SCF methods. `calc_mulliken_charges()` returns one Mulliken charge per atom after an SCF calculation. Use `calc_uhf_energy()` for UHF and `wfn.calc_mp2_energy()` for RHF MP2.

See [psi4_full.py](validation/psi4_full.py) and [psi4_full.log](validation/psi4_full.log) for a worked example with validation against Psi4 reference matrices across STO-3G, 6-31G, 6-31G**, and cc-pVDZ.

## Files

See [SUPPLEMENTS.md](docs/SUPPLEMENTS.md) for the supplemental equations and derivations.

| File | Contents |
|---|---|
| [mol_basis_builder.py](src/scratch_code_fock/mol_basis_builder.py) | `Molecule` and `Basis`/`Shell` data structures; basis set loading via Basis Set Exchange |
| [integral_solvers.py](src/scratch_code_fock/integral_solvers.py) | Obara-Saika recursions for S, T, nuclear attraction, and ERI primitive integrals |
| [matrix_builders.py](src/scratch_code_fock/matrix_builders.py) | Shell-pair/quartet loops, symmetry exploitation, AO-basis matrix assembly |
| [wfn.py](src/scratch_code_fock/wfn.py) | High-level `WaveFunction` interface for RHF, UHF, MP2, and Mulliken charges |
| [rhf.py](src/scratch_code_fock/rhf.py), [uhf.py](src/scratch_code_fock/uhf.py) | RHF/UHF SCF loops: symmetric orthogonalization, Fock build, DIIS |
| [psi4_full.py](validation/psi4_full.py), [psi4_full.log](validation/psi4_full.log) | End-to-end run + validation against Psi4 |
