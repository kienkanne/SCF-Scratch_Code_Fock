# CHANGELOG

## 20th July 2026

Reorganize repository with proper structure
- Moved computation units to src/, and changed main pipeline notebook to a python file with output saved to a text file in project's root
- Added [SUPPLEMENTS.md](SUPPLEMENTS.md), which is the math supplement to the codebase.
- Past notebooks to verify directly with Psi4 results for each individual unit is added to `notebooks/`
- Moved Roothan solver parameters directly to `roothan_solver`, and add `return_mulliken` (requiring `basis` input also) parameter
- Added new methods and attributes to `Basis` and `Molecule` for Mulliken population analysis
- Moved `get_NAIs` outside of the angular combinations loop for performance

## 10th July 2026

Initial commit: all four individual units and a full pipeline notebook saved.