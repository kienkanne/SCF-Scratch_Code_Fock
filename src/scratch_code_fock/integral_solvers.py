import numpy as np

def get_overlap_integrals(PA, PB, zeta, AMa, AMb):

    overlap_integrals = np.zeros((3, AMa + 1, AMb + 1))
    
    overlap_integrals[:, 0, 0] = 1.0

    for ib in range(AMb + 1):
        for ia in range(AMa + 1):
            if ia == 0 and ib == 0:
                continue
                
            # If ib is 0, we are stepping up Center A
            if ib == 0:
                term1 = PA * overlap_integrals[:, ia - 1, 0]
                term2 = (0.5 / zeta) * (ia - 1) * overlap_integrals[:, ia - 2, 0] if ia > 1 else 0.0
                
                overlap_integrals[:, ia, 0] = term1 + term2

            # If ib > 0, we are stepping up Center B
            else:
                term1 = PB * overlap_integrals[:, ia, ib - 1]
                term2 = (0.5 / zeta) * ia * overlap_integrals[:, ia - 1, ib - 1] if ia > 0 else 0.0
                term3 = (0.5 / zeta) * (ib - 1) * overlap_integrals[:, ia, ib - 2] if ib > 1 else 0.0
                
                overlap_integrals[:, ia, ib] = term1 + term2 + term3

    return overlap_integrals


def get_kinetic_integrals(PA, PB, expa, expb, AMa, AMb, SI):
    zeta = expa + expb
    xi = (expa * expb) / zeta

    kinetic_integrals = np.zeros((3, AMa + 1, AMb + 1))

    # Seed base case
    kinetic_integrals[:, 0, 0] = xi * (1.0 - 2.0 * xi * (PA - PB)**2)

    for ib in range(AMb + 1):
        for ia in range(AMa + 1):
            if ia == 0 and ib == 0:
                continue

            # If ib is 0, stepping up Center A
            if ib == 0:
                term1 = PA * kinetic_integrals[:, ia - 1, ib]
                term2 = (0.5 / zeta) * (ia - 1) * kinetic_integrals[:, ia - 2, ib] if ia > 1 else 0.0
                term3 = (0.5 / zeta) * ib * kinetic_integrals[:, ia - 1, ib - 1] if ib > 0 else 0.0
                term4 = (2 * xi) * (SI[:, ia, ib] - (0.5 / expa) * (ia - 1) * SI[:, ia - 2, ib])

                kinetic_integrals[:, ia, 0] = term1 + term2 + term3 + term4

            # If ib > 0, stepping up Center B
            else:
                term1 = PB * kinetic_integrals[:, ia, ib - 1]
                term2 = (0.5 / zeta) * ia * kinetic_integrals[:, ia - 1, ib - 1] if ia > 0 else 0.0
                term3 = (0.5 / zeta) * (ib - 1) * kinetic_integrals[:, ia, ib - 2] if ib > 1 else 0.0
                term4 = (2 * xi) * (SI[:, ia, ib] - (0.5 / expb) * (ib - 1) * SI[:, ia, ib - 2])

                kinetic_integrals[:, ia, ib] = term1 + term2 + term3 + term4

    return kinetic_integrals


def boys_function(m, x):
    """Evaluates the Boys function F_m(x) using Scipy's error function."""
    if x < 1e-9:
        return 1.0 / (2 * m + 1)
   
    from scipy.special import gammainc, gamma
    return 0.5 * x**(-(m + 0.5)) * gamma(m + 0.5) * gammainc(m + 0.5, x)


def get_NAIs(PA, PB, PC, zeta, AMa, AMb):
    max_m = AMa + AMb
    NAI = np.zeros((AMa + 1, AMa + 1, AMa + 1, AMb + 1, AMb + 1, AMb + 1, max_m + 1))
    
    def shift_NAI(swap_dict: dict):
        """
        Returns the current NAI index but with shifted index
        """
        axis_dict = {
            "ax": 0, "ay": 1, "az": 2,
            "bx": 3, "by": 4, "bz": 5, 
            "m": 6
        }
        
        # Captures current loop variables
        curr_idx = [ax, ay, az, bx, by, bz, m]
        for axis, swap in swap_dict.items():
            idx = axis_dict[axis]
            curr_idx[idx] += swap

        if any(idx < 0 for idx in curr_idx):
            return 0.0

        return NAI[tuple(curr_idx)]
    
    # 1. Foundation: Seed the [0,0,0|0,0,0]^(m) base cases
    U = zeta * np.sum(PC**2)
    for m_idx in range(max_m + 1):
        NAI[0, 0, 0, 0, 0, 0, m_idx] = boys_function(m_idx, U) 
        
    # 2. Chained 3D Step-Up Engine
    # We must build L strictly from 1 up to (AMa + AMb) to ensure lower dependencies exist
    for L in range(1, AMa + AMb + 1):
        for ax in range(AMa + 1):
            for ay in range(AMa + 1 - ax):
                for az in range(AMa + 1 - ax - ay):
                    for bx in range(AMb + 1):
                        for by in range(AMb + 1 - bx):
                            for bz in range(AMb + 1 - bx - by):
                                
                                # Only process states that sum to the current total angular momentum
                                if ax + ay + az + bx + by + bz != L:
                                    continue
                                    
                                # The +1 ensures we only compute the max_m needed for the current total angular momentum
                                for m in range(max_m - L + 1):
                                # Step up the first available non-zero angular momentum component
                                    if ax > 0:
                                        term1 = PA[0] * shift_NAI({"ax": -1})
                                        term2 = PC[0] * shift_NAI({"ax": -1, "m": +1})
                                        term3 = (0.5 / zeta) * (ax - 1) * (shift_NAI({"ax": -2}) - shift_NAI({"ax": -2, "m": +1})) if ax > 1 else 0.0
                                        term4 = (0.5 / zeta) * bx * (shift_NAI({"ax": -1, "bx": -1}) - shift_NAI({"ax": -1, "bx": -1, "m": +1})) if bx > 0 else 0.0
                                        NAI[ax, ay, az, bx, by, bz, m] = term1 - term2 + term3 + term4
                                        
                                    elif ay > 0:
                                        term1 = PA[1] * shift_NAI({"ay": -1})
                                        term2 = PC[1] * shift_NAI({"ay": -1, "m": +1})
                                        term3 = (0.5 / zeta) * (ay - 1) * (shift_NAI({"ay": -2}) - shift_NAI({"ay": -2, "m": +1})) if ay > 1 else 0.0
                                        term4 = (0.5 / zeta) * by * (shift_NAI({"ay": -1, "by": -1}) - shift_NAI({"ay": -1, "by": -1, "m": +1})) if by > 0 else 0.0
                                        NAI[ax, ay, az, bx, by, bz, m] = term1 - term2 + term3 + term4
                                        
                                    elif az > 0:
                                        term1 = PA[2] * shift_NAI({"az": -1})
                                        term2 = PC[2] * shift_NAI({"az": -1, "m": +1})
                                        term3 = (0.5 / zeta) * (az - 1) * (shift_NAI({"az": -2}) - shift_NAI({"az": -2, "m": +1})) if az > 1 else 0.0
                                        term4 = (0.5 / zeta) * bz * (shift_NAI({"az": -1, "bz": -1}) - shift_NAI({"az": -1, "bz": -1, "m": +1})) if bz > 0 else 0.0
                                        NAI[ax, ay, az, bx, by, bz, m] = term1 - term2 + term3 + term4
                                        
                                    elif bx > 0:
                                        term1 = PB[0] * shift_NAI({"bx": -1})
                                        term2 = PC[0] * shift_NAI({"bx": -1, "m": +1})
                                        term3 = (0.5 / zeta) * (bx - 1) * (shift_NAI({"bx": -2}) - shift_NAI({"bx": -2, "m": +1})) if bx > 1 else 0.0
                                        NAI[ax, ay, az, bx, by, bz, m] = term1 - term2 + term3
                                        
                                    elif by > 0:
                                        term1 = PB[1] * shift_NAI({"by": -1})
                                        term2 = PC[1] * shift_NAI({"by": -1, "m": +1})
                                        term3 = (0.5 / zeta) * (by - 1) * (shift_NAI({"by": -2}) - shift_NAI({"by": -2, "m": +1})) if by > 1 else 0.0
                                        NAI[ax, ay, az, bx, by, bz, m] = term1 - term2 + term3
                                        
                                    elif bz > 0:
                                        term1 = PB[2] * shift_NAI({"bz": -1})
                                        term2 = PC[2] * shift_NAI({"bz": -1, "m": +1})
                                        term3 = (0.5 / zeta) * (bz - 1) * (shift_NAI({"bz": -2}) - shift_NAI({"bz": -2, "m": +1})) if bz > 1 else 0.0
                                        NAI[ax, ay, az, bx, by, bz, m] = term1 - term2 + term3

    return NAI


def get_ERIs(A, B, C, D, expa, expb, expc, expd, AMa, AMb, AMc, AMd):
    max_m = AMa + AMb + AMc + AMd
    
    ERI = np.zeros((AMa + 1, AMa + 1, AMa + 1, 
                  AMb + 1, AMb + 1, AMb + 1,
                  AMc + 1, AMc + 1, AMc + 1,
                  AMd + 1, AMd + 1, AMd + 1,
                  max_m + 1))

    def swap_ERI(swap_dict: dict):
        """
        Returns the current ERI index but with shifted index
        """
        axis_dict = {
            "ax": 0, "ay": 1, "az": 2,
            "bx": 3, "by": 4, "bz": 5,
            "cx": 6, "cy": 7, "cz": 8,
            "dx": 9, "dy": 10, "dz": 11, 
            "m": 12
        }
        
        # Captures current loop variables
        curr_idx = [ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, m]
        for axis, swap in swap_dict.items():
            idx = axis_dict[axis]
            curr_idx[idx] += swap

        if any(idx < 0 for idx in curr_idx):
            return 0.0

        return ERI[tuple(curr_idx)]


    zeta = expa + expb
    eta = expc + expd
    ez_sum = zeta + eta

    P = (expa * A + expb * B) / zeta
    Q = (expc * C + expd * D) / eta

    W = (zeta * P + eta * Q) / ez_sum
    WP = W - P
    WQ = W - Q

    PA = P - A
    PB = P - B
    QC = Q - C
    QD = Q - D
    
    # Seed base cases
    rho = (zeta * eta) / (zeta + eta)
    T = rho * np.sum((P-Q)**2)
    for m in range(max_m + 1):
        ERI[0, 0, 0, 
          0, 0, 0,
          0, 0, 0, 
          0, 0, 0,  
          m] = boys_function(m, T)
        
    # We must build L strictly from 1 up to (AMa + AMb) to ensure lower dependencies exist
    for L in range(1, AMa + AMb + AMc + AMd + 1):
        for ax in range(AMa + 1):
            for ay in range(AMa + 1 - ax):
                for az in range(AMa + 1 - ax - ay):
                    for bx in range(AMb + 1):
                        for by in range(AMb + 1 - bx):
                            for bz in range(AMb + 1 - bx - by):
                                for cx in range(AMc + 1):
                                    for cy in range(AMc + 1 - cx):
                                        for cz in range(AMc + 1 - cx - cy):
                                            for dx in range(AMd + 1):
                                                for dy in range(AMd + 1 - dx):
                                                    for dz in range(AMd + 1 - dx - dy):
                                
                                                        # Only process states that sum to the current total angular momentum
                                                        if ax + ay + az + bx + by + bz + cx + cy + cz + dx + dy + dz != L:
                                                            continue
                                                            
                                                        # The +1 ensures we only compute the max_m needed for the current total angular momentum
                                                        for m in range(max_m - L + 1):
                                                        # Step up the first available non-zero angular momentum component
                                                            if ax > 0:
                                                                term1 = PA[0] * swap_ERI({"ax": -1})
                                                                term2 = WP[0] * swap_ERI({"ax": -1, "m": +1})
                                                                term3 = (0.5 / zeta) * (ax - 1) * (swap_ERI({"ax": -2}) - (eta / ez_sum) * swap_ERI({"ax": -2, "m": +1})) if ax > 1 else 0.0
                                                                term4 = (0.5 / zeta) * bx * (swap_ERI({"ax": -1, "bx": -1}) - (eta / ez_sum) * swap_ERI({"ax": -1, "bx": -1, "m": +1})) if bx > 0 else 0.0
                                                                term5 = (0.5 / ez_sum) * cx * swap_ERI({"ax": -1, "cx": -1, "m": +1}) if cx > 0 else 0.0
                                                                term6 = (0.5 / ez_sum) * dx * swap_ERI({"ax": -1, "dx": -1, "m": +1}) if dx > 0 else 0.0
                                                                ERI[ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, m] = term1 + term2 + term3 + term4 + term5 + term6

                                                            elif ay > 0:
                                                                term1 = PA[1] * swap_ERI({"ay": -1})
                                                                term2 = WP[1] * swap_ERI({"ay": -1, "m": +1})
                                                                term3 = (0.5 / zeta) * (ay - 1) * (swap_ERI({"ay": -2}) - (eta / ez_sum) * swap_ERI({"ay": -2, "m": +1})) if ay > 1 else 0.0
                                                                term4 = (0.5 / zeta) * by * (swap_ERI({"ay": -1, "by": -1}) - (eta / ez_sum) * swap_ERI({"ay": -1, "by": -1, "m": +1})) if by > 0 else 0.0
                                                                term5 = (0.5 / ez_sum) * cy * swap_ERI({"ay": -1, "cy": -1, "m": +1}) if cy > 0 else 0.0
                                                                term6 = (0.5 / ez_sum) * dy * swap_ERI({"ay": -1, "dy": -1, "m": +1}) if dy > 0 else 0.0
                                                                ERI[ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, m] = term1 + term2 + term3 + term4 + term5 + term6

                                                            elif az > 0:
                                                                term1 = PA[2] * swap_ERI({"az": -1})
                                                                term2 = WP[2] * swap_ERI({"az": -1, "m": +1})
                                                                term3 = (0.5 / zeta) * (az - 1) * (swap_ERI({"az": -2}) - (eta / ez_sum) * swap_ERI({"az": -2, "m": +1})) if az > 1 else 0.0
                                                                term4 = (0.5 / zeta) * bz * (swap_ERI({"az": -1, "bz": -1}) - (eta / ez_sum) * swap_ERI({"az": -1, "bz": -1, "m": +1})) if bz > 0 else 0.0
                                                                term5 = (0.5 / ez_sum) * cz * swap_ERI({"az": -1, "cz": -1, "m": +1}) if cz > 0 else 0.0
                                                                term6 = (0.5 / ez_sum) * dz * swap_ERI({"az": -1, "dz": -1, "m": +1}) if dz > 0 else 0.0
                                                                ERI[ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, m] = term1 + term2 + term3 + term4 + term5 + term6

                                                            elif bx > 0:
                                                                term1 = PB[0] * swap_ERI({"bx": -1})
                                                                term2 = WP[0] * swap_ERI({"bx": -1, "m": +1})
                                                                term3 = (0.5 / zeta) * ax * (swap_ERI({"ax": -1, "bx": -1}) - (eta / ez_sum) * swap_ERI({"ax": -1, "bx": -1, "m": +1})) if ax > 0 else 0.0
                                                                term4 = (0.5 / zeta) * (bx - 1) * (swap_ERI({"bx": -2}) - (eta / ez_sum) * swap_ERI({"bx": -2, "m": +1})) if bx > 1 else 0.0
                                                                term5 = (0.5 / ez_sum) * cx * swap_ERI({"bx": -1, "cx": -1, "m": +1}) if cx > 0 else 0.0
                                                                term6 = (0.5 / ez_sum) * dx * swap_ERI({"bx": -1, "dx": -1, "m": +1}) if dx > 0 else 0.0
                                                                ERI[ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, m] = term1 + term2 + term3 + term4 + term5 + term6

                                                            elif by > 0:
                                                                term1 = PB[1] * swap_ERI({"by": -1})
                                                                term2 = WP[1] * swap_ERI({"by": -1, "m": +1})
                                                                term3 = (0.5 / zeta) * ay * (swap_ERI({"ay": -1, "by": -1}) - (eta / ez_sum) * swap_ERI({"ay": -1, "by": -1, "m": +1})) if ay > 0 else 0.0
                                                                term4 = (0.5 / zeta) * (by - 1) * (swap_ERI({"by": -2}) - (eta / ez_sum) * swap_ERI({"by": -2, "m": +1})) if by > 1 else 0.0
                                                                term5 = (0.5 / ez_sum) * cy * swap_ERI({"by": -1, "cy": -1, "m": +1}) if cy > 0 else 0.0
                                                                term6 = (0.5 / ez_sum) * dy * swap_ERI({"by": -1, "dy": -1, "m": +1}) if dy > 0 else 0.0
                                                                ERI[ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, m] = term1 + term2 + term3 + term4 + term5 + term6

                                                            elif bz > 0:
                                                                term1 = PB[2] * swap_ERI({"bz": -1})
                                                                term2 = WP[2] * swap_ERI({"bz": -1, "m": +1})
                                                                term3 = (0.5 / zeta) * az * (swap_ERI({"az": -1, "bz": -1}) - (eta / ez_sum) * swap_ERI({"az": -1, "bz": -1, "m": +1})) if az > 0 else 0.0
                                                                term4 = (0.5 / zeta) * (bz - 1) * (swap_ERI({"bz": -2}) - (eta / ez_sum) * swap_ERI({"bz": -2, "m": +1})) if bz > 1 else 0.0
                                                                term5 = (0.5 / ez_sum) * cz * swap_ERI({"bz": -1, "cz": -1, "m": +1}) if cz > 0 else 0.0
                                                                term6 = (0.5 / ez_sum) * dz * swap_ERI({"bz": -1, "dz": -1, "m": +1}) if dz > 0 else 0.0
                                                                ERI[ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, m] = term1 + term2 + term3 + term4 + term5 + term6

                                                            elif cx > 0:
                                                                term1 = QC[0] * swap_ERI({"cx": -1})
                                                                term2 = WQ[0] * swap_ERI({"cx": -1, "m": +1})
                                                                term3 = (0.5 / ez_sum) * ax * swap_ERI({"ax": -1, "cx": -1, "m": +1}) if ax > 0 else 0.0
                                                                term4 = (0.5 / ez_sum) * bx * swap_ERI({"bx": -1, "cx": -1, "m": +1}) if bx > 0 else 0.0
                                                                term5 = (0.5 / eta) * (cx - 1) * (swap_ERI({"cx": -2}) - (zeta / ez_sum) * swap_ERI({"cx": -2, "m": +1})) if cx > 1 else 0.0
                                                                term6 = (0.5 / eta) * dx * (swap_ERI({"cx": -1, "dx": -1}) - (zeta / ez_sum) * swap_ERI({"cx": -1, "dx": -1, "m": +1})) if dx > 0 else 0.0
                                                                ERI[ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, m] = term1 + term2 + term3 + term4 + term5 + term6

                                                            elif cy > 0:
                                                                term1 = QC[1] * swap_ERI({"cy": -1})
                                                                term2 = WQ[1] * swap_ERI({"cy": -1, "m": +1})
                                                                term3 = (0.5 / ez_sum) * ay * swap_ERI({"ay": -1, "cy": -1, "m": +1}) if ay > 0 else 0.0
                                                                term4 = (0.5 / ez_sum) * by * swap_ERI({"by": -1, "cy": -1, "m": +1}) if by > 0 else 0.0
                                                                term5 = (0.5 / eta) * (cy - 1) * (swap_ERI({"cy": -2}) - (zeta / ez_sum) * swap_ERI({"cy": -2, "m": +1})) if cy > 1 else 0.0
                                                                term6 = (0.5 / eta) * dy * (swap_ERI({"cy": -1, "dy": -1}) - (zeta / ez_sum) * swap_ERI({"cy": -1, "dy": -1, "m": +1})) if dy > 0 else 0.0
                                                                ERI[ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, m] = term1 + term2 + term3 + term4 + term5 + term6

                                                            elif cz > 0:
                                                                term1 = QC[2] * swap_ERI({"cz": -1})
                                                                term2 = WQ[2] * swap_ERI({"cz": -1, "m": +1})
                                                                term3 = (0.5 / ez_sum) * az * swap_ERI({"az": -1, "cz": -1, "m": +1}) if az > 0 else 0.0
                                                                term4 = (0.5 / ez_sum) * bz * swap_ERI({"bz": -1, "cz": -1, "m": +1}) if bz > 0 else 0.0
                                                                term5 = (0.5 / eta) * (cz - 1) * (swap_ERI({"cz": -2}) - (zeta / ez_sum) * swap_ERI({"cz": -2, "m": +1})) if cz > 1 else 0.0
                                                                term6 = (0.5 / eta) * dz * (swap_ERI({"cz": -1, "dz": -1}) - (zeta / ez_sum) * swap_ERI({"cz": -1, "dz": -1, "m": +1})) if dz > 0 else 0.0
                                                                ERI[ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, m] = term1 + term2 + term3 + term4 + term5 + term6

                                                            elif dx > 0:
                                                                term1 = QD[0] * swap_ERI({"dx": -1})
                                                                term2 = WQ[0] * swap_ERI({"dx": -1, "m": +1})
                                                                term3 = (0.5 / ez_sum) * ax * swap_ERI({"ax": -1, "dx": -1, "m": +1}) if ax > 0 else 0.0
                                                                term4 = (0.5 / ez_sum) * bx * swap_ERI({"bx": -1, "dx": -1, "m": +1}) if bx > 0 else 0.0
                                                                term5 = (0.5 / eta) * cx * (swap_ERI({"cx": -1, "dx": -1}) - (zeta / ez_sum) * swap_ERI({"cx": -1, "dx": -1, "m": +1})) if cx > 0 else 0.0
                                                                term6 = (0.5 / eta) * (dx - 1) * (swap_ERI({"dx": -2}) - (zeta / ez_sum) * swap_ERI({"dx": -2, "m": +1})) if dx > 1 else 0.0
                                                                ERI[ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, m] = term1 + term2 + term3 + term4 + term5 + term6

                                                            elif dy > 0:
                                                                term1 = QD[1] * swap_ERI({"dy": -1})
                                                                term2 = WQ[1] * swap_ERI({"dy": -1, "m": +1})
                                                                term3 = (0.5 / ez_sum) * ay * swap_ERI({"ay": -1, "dy": -1, "m": +1}) if ay > 0 else 0.0
                                                                term4 = (0.5 / ez_sum) * by * swap_ERI({"by": -1, "dy": -1, "m": +1}) if by > 0 else 0.0
                                                                term5 = (0.5 / eta) * cy * (swap_ERI({"cy": -1, "dy": -1}) - (zeta / ez_sum) * swap_ERI({"cy": -1, "dy": -1, "m": +1})) if cy > 0 else 0.0
                                                                term6 = (0.5 / eta) * (dy - 1) * (swap_ERI({"dy": -2}) - (zeta / ez_sum) * swap_ERI({"dy": -2, "m": +1})) if dy > 1 else 0.0
                                                                ERI[ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, m] = term1 + term2 + term3 + term4 + term5 + term6

                                                            elif dz > 0:
                                                                term1 = QD[2] * swap_ERI({"dz": -1})
                                                                term2 = WQ[2] * swap_ERI({"dz": -1, "m": +1})
                                                                term3 = (0.5 / ez_sum) * az * swap_ERI({"az": -1, "dz": -1, "m": +1}) if az > 0 else 0.0
                                                                term4 = (0.5 / ez_sum) * bz * swap_ERI({"bz": -1, "dz": -1, "m": +1}) if bz > 0 else 0.0
                                                                term5 = (0.5 / eta) * cz * (swap_ERI({"cz": -1, "dz": -1}) - (zeta / ez_sum) * swap_ERI({"cz": -1, "dz": -1, "m": +1})) if cz > 0 else 0.0
                                                                term6 = (0.5 / eta) * (dz - 1) * (swap_ERI({"dz": -2}) - (zeta / ez_sum) * swap_ERI({"dz": -2, "m": +1})) if dz > 1 else 0.0
                                                                ERI[ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, m] = term1 + term2 + term3 + term4 + term5 + term6
    return ERI