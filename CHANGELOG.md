# CHANGELOG

## 20th July 2026

- Reorganize repository with proper structure, moving computation units to src/, and keeping the full pipeline notebook in project root
- Past notebooks to verify directly with Psi4 results for each individual unit is added to `notebooks/`
- Moved Roothan solver parameters directly to `roothan_solver`, and add `return_mulliken` (requiring `basis` input also) parameter
- Added new methods and attributes to `Basis` and `Molecule` for Mulliken population analysis

## 10th July 2026

Initial commit: all four individual units and a full pipeline notebook saved.