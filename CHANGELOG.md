# CHANGELOG

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [2026-07-20]

Reorganize repository with proper structure

### Added
- [SUPPLEMENTS.md](SUPPLEMENTS.md), which is the math supplement to the codebase.
- Mulliken population analysis, as well as `function_to_center()` method for `Basis` class, mirroring Psi4
- Past notebooks to verify directly with Psi4 results for each individual unit is added to `notebooks/`

### Changed
- Moved computation units to src/, and changed main pipeline notebook to a python file with output saved to a text file in project's root
- Moved Roothan solver parameters directly to `roothan_solver`, and add `return_mulliken` (requiring `basis` input also) parameter
- Moved `get_NAIs` outside of the angular combinations loop for performance

## [2026-07-10]

Initial commit: all four individual units and a full pipeline notebook saved.