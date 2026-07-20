# DERIVATIONS.md

This is the mathematical supplements to the scratch-built RHF/SCF engine in this repository.
Every section below corresponds to one function or class in the codebase

**This is a from-scratch, first-principles, proof-of-concept-focused implementation.**
Every integral is evaluated via explicit Obara-Saika recursion in pure NumPy,
with no primitive screening or batching. It is not meant to be fast, it is
meant to be a direct, line-by-line map from the equations below to working code.

## Gaussian normalization — `Shell.__init__` (`mol_basis_builder.py`)