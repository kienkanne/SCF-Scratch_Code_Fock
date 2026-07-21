# SCF: Scratch-Code-Fock

A restricted Hartree-Fock (RHF) Self-Consistent-Field (SCF) engine, built entirely from first
principles in pure NumPy. Every integral — overlap, kinetic, nuclear
attraction, and electron repulsion — is evaluated via explicit Obara-Saika
recursion, with no dependency on a quantum chemistry package for the
integral evaluation itself.

> **This is an educational, unoptimized, scratch-built implementation.**
> The focus throughout is mathematical transparency and correctness, not
> speed. There is no primitive screening, no integral batching, and the
> electron repulsion integral recursion is a genuinely brute-force
> 13-dimensional Obara-Saika evaluation. The design was inspired from the 
> Psi4 quantum chemistry package.

## Dependencies

```
numpy
scipy                  # Boys function, via scipy.special.gamma / gammainc
basis_set_exchange     # basis set data (exponents, contraction coefficients)
```

## Usage

```python
from mol_basis_builder import Molecule
from matrix_builders import build_S_T_V, build_ERI
from roothan_solver import roothan_solver

mol = Molecule("""
O    0.000000    0.000000   -0.143225
H    0.000000    1.638036    1.136548
H    0.000000   -1.638036    1.136548
""")

basis = mol.build_basis("sto-3g")
S, T, V = build_S_T_V(mol, basis)
I = build_ERI(mol, basis)

E = roothan_solver(mol, S, T, V, I)
```

See [full_pipeline.py](full_pipeline.py) and [pipeline_output.txt](pipeline_output.txt) for a worked example with validation against Psi4
reference matrices across STO-3G, 6-31G, 6-31G**, and cc-pVDZ.

## Files

| File | Contents |
|---|---|
| [mol_basis_builder.py](src/scratch_code_fock/mol_basis_builder.py) | `Molecule` and `Basis`/`Shell` data structures; basis set loading via Basis Set Exchange |
| [integral_solvers.py](src/scratch_code_fock/integral_solvers.py) | Obara-Saika recursions for S, T, nuclear attraction, and ERI primitive integrals |
| [matrix_builders.py](src/scratch_code_fock/matrix_builders.py) | Shell-pair/quartet loops, symmetry exploitation, AO-basis matrix assembly |
| [roothan_solver.py](src/scratch_code_fock/roothan_solver.py) | SCF loop: symmetric orthogonalization, Fock build, DIIS |
| [full_pipeline.py](full_pipeline.py), [pipeline_output.txt](pipeline_output.txt) | End-to-end run + validation against Psi4 |