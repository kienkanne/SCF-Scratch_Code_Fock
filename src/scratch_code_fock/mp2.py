import numpy as np

def transform_MO(I, C, n_occ):
    C_occ = C[:, :n_occ]
    C_virt = C[:, n_occ:]

    # O(N^5) transformation
    I_mo = np.einsum('pi,pqrs->iqrs', C_occ, I, optimize=True)
    I_mo = np.einsum('qa,iqrs->iars', C_virt, I_mo, optimize=True)
    I_mo = np.einsum('iars,rj->iajs', I_mo, C_occ, optimize=True)
    I_mo = np.einsum('iajs,sb->iajb', I_mo, C_virt, optimize=True)

    return I_mo


def calc_MP2_E(I_mo, n_occ, eps, E0):
    e_ij = eps[:n_occ]
    e_ab = eps[n_occ:]

    e_denom = 1 / (e_ij.reshape(-1, 1, 1, 1) - e_ab.reshape(1, -1, 1, 1) + e_ij.reshape(1, 1, -1, 1) - e_ab.reshape(1, 1, 1, -1))

    # Compute SS & OS MP2 Correlation with Einsum
    # `np.swapaxes` for correct indexing for the exchange integral
    os_E = np.einsum('iajb,iajb,iajb->', I_mo, I_mo, e_denom, optimize=True)
    ss_E = np.einsum('iajb,iajb,iajb->', I_mo, I_mo - I_mo.swapaxes(1,3), e_denom, optimize=True)

    # Total MP2 Energy
    MP2_E = E0 + os_E + ss_E

    return MP2_E