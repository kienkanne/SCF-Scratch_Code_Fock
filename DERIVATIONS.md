# DERIVATIONS.md

This is the mathematical supplements to the scratch-built RHF/SCF engine in this repository.
Every section below corresponds to one function or class in the codebase

**This is a from-scratch, first-principles, proof-of-concept-focused implementation.**
Every integral is evaluated via explicit Obara-Saika recursion in pure NumPy, with no primitive screening or batching. It is not meant to be fast, it is meant to be a direct, line-by-line map from the equations below to working code.

---

## Gaussian normalization — `Shell.__init__` (`mol_basis_builder.py`)

Every contracted Cartesian basis function is a normalized sum of primitives:

$$
\phi(\mathbf{r}, \mathbf{r_A}, \alpha_p, c_p) = N_{\text{cont}} \sum_p c_p \, N_{\text{prim}}(\alpha_p) \, P(\mathbf{r} - \mathbf{r_A}) \, \exp(- \alpha_p ||\mathbf{r} - \mathbf{r_A}||^2)
$$

where $P(\mathbf{r} - \mathbf{r_A})$ is the equivalent to the polynomial $P(x, y, z) = x^{n_x}y^{n_y}z^{n_z}$, and $n_x + n_y + n_z = l$, the total angular momentum.

**Contraction normalization** (self-overlap of the full contracted function $=1$), using the primitive-pair overlap matrix:

$$S_{pq} = \left(\dfrac{2\sqrt{\alpha_p\alpha_q}}{\alpha_p+\alpha_q}\right)^{l+3/2}$$:

$$
N_{\text{cont}} = \left( \sum_{p,q} c_p c_q S_{pq} \right)^{-1/2}
$$

`effective_coef` returns $N_{\text{cont}} \, c_p \, N_{\text{prim}}(\alpha_p)$ —
the single scalar multiplied into every integral involving this primitive in
`matrix_builders.py`.

**Primitive normalization** (self-overlap of a single primitive $=1$), for the "pure" axial component $(l,0,0)$:

$$
N_{\text{prim}}(\alpha) = \left(\frac{2\alpha}{\pi}\right)^{3/4} (4\alpha)^{l/2} \Big/ \sqrt{(2l-1)!!}
$$

$\alpha_p$ and $c_p$ are fetched from Basis Set Exchange, while both normalization are calculated following these formulas.

**NOTE (angular normalization):** the $1/\sqrt{(2l-1)!!}$ factor above is exact only for the "pure" Cartesian components where all angular momentum sits on one axis, for example $(2,0,0)$, $(0,2,0)$, $(0,0,2)$ for d functions. The general per-component formula, for Cartesian indices $(n_x,n_y,n_z)$, is

$$
\frac{1}{\sqrt{(2n_x-1)!!\,(2n_y-1)!!\,(2n_z-1)!!}}
$$

which reduces to $1/\sqrt{(2l-1)!!}$ for pure components but gives exactly $1$ for mixed components like $(1,1,0)$, not $1/\sqrt{3}$. `Shell` applies one `angular_norm` value per shell (for all combinations, computed from $l$ alone), so mixed Cartesian components (d$_{xy}$, d$_{xz}$, d$_{yz}$ and higher-$l$ analogues) are technically wrong. However, this normalization scheme (or any normalization scheme) leaves the SCF energy unaffected (see **Roothan-Hall equations** section) while still changing individual AO-matrix elements for any basis with $l \geq 2$ functions (verified directly against Psi4 reference matrices for cc-pVDZ). The normalization scheme, both primitive and contraction, was attempted to replicate Psi4's coefficients with already baked-in normalization. cc-pVDZ, or higher basis sets, have more specialized schemes, and currently fully replicating them is not a priority of this repository.

> Only the "superdiagonal" functions like or are normalized in Cartesian. The other functions have well-defined fractional normalizations that are typically accounted for automatically in the orthonormalization of the basis within the SCF code - *NVIDIA cuEST Documentation*

*Source: [IOData Documentation](https://iodata.readthedocs.io/en/latest/basis.html), [NVIDIA cuEST Documentation](https://docs.nvidia.com/cuda/cuest/gaussian_basis.html#gaussian-basis-functions)*

---

## Matrices construction, Roothaan-Hall equations, and symmetric orthogonalization — `roothan_solver` (`roothan_solver.py`)

The Hamiltonian matrix with no electron interaction is constructed directly from the kinetic integral matrix and the nuclear attraction (potential) integral matrix: $\mathbf{H} = \mathbf{T} + \mathbf{V}$. This is also used as the initial guess for the Fock matrix in this implementation, although there are different guess strategies.

Closed-shell density matrix from the occupied MO block $\mathbf{C}_{\text{occ}}$
($n_{\text{docc}}$ columns):

$$
D_{\mu\nu} = \sum_{i}^{\text{occ}} C_{\mu i} C_{\nu i} = (\mathbf{C}_{\text{occ}}\mathbf{C}_{\text{occ}}^T)_{\mu\nu}
$$

Coulomb and exchange matrices, contracted directly against the full 4-index
ERI tensor $I_{pqrs} = (pq|rs)$:

$$
J_{pq} = \sum_{rs} I_{pqrs}\, D_{rs}, \qquad K_{pq} = \sum_{rs} I_{prqs}\, D_{rs}, \qquad
\mathbf{F} = \mathbf{H} + 2\mathbf{J} - \mathbf{K}
$$

implemented as `np.einsum('pqrs,rs->pq', I, D)` and `np.einsum('prqs,rs->pq', I, D)`. Notice the index permutation between the two einsum strings ($q,r$ swap roles) is exactly what distinguishes the exchange intgral from the Coulomb integral.

The RHF equations in a non-orthogonal AO basis are a generalized eigenvalue problem,

$$
\mathbf{F}\mathbf{C} = \mathbf{S}\mathbf{C}\boldsymbol{\varepsilon}
$$

To eliminate S and make this an ordinary eigenvalue problem, we need to find a transformation matrix $\mathbf{X}$ such that $\mathbf{X}^T\mathbf{S}\mathbf{X} = \mathbf{I}$. Symmetric (Löwdin) orthogonalization does this by using 

$$
\mathbf{X} = \mathbf{S}^{-1/2} = \mathbf{U}\,\mathbf{s}^{-1/2}\,\mathbf{U}^T
$$

and $\mathbf{s}$ can be obtained by diagonalizing the overlap matrix, $\mathbf{S}=\mathbf{U}\mathbf{s}\mathbf{U}^T$, which is well-defined only because $\mathbf{S}$ is symmetric positive (semi)definite, making its eigenvalues positive and the elementwise inverse-square-root valid. 

With $\mathbf{F}' = \mathbf{X}^T\mathbf{F}\mathbf{X}$ and $\mathbf{C} = \mathbf{X}\mathbf{C}'$, the transformed problem $\mathbf{F}'\mathbf{C}' = \mathbf{C}'\boldsymbol{\epsilon}$ is ordinary and can be solved trivially.

The RHF energy can be obtained via constructed matrices:

$$
E_0 = \sum_{\mu\nu} D_{\mu\nu}\big(H_{\mu\nu}+F_{\mu\nu}\big) + V_{NN}
$$

*Source: [Szabo & Ostlund, Ch. 3.4]*

**Why different shell-level angular normalization schemes doesn't break the energy:** 
The generalized eigensolve above enforces $\mathbf{C}^T\mathbf{S}\mathbf{C}=\mathbf{I}$ using the $\mathbf{S}$ constructed from the AO basis, so even if any given AO is scaled by some constant, that scaling is automatically absorbed into the corresponding MO coefficients. The physical molecular orbitals and total energy come out identical either way. What is not identical are the electron integrals themselves, confirmed directly for cc-pVDZ, where individual $S,T,V,I$ entries differ from Psi4's reference values while the converged SCF energy still matches to $\sim\!10^{-8}\,E_h$. (See NVIDIA cuEST Documentation above)