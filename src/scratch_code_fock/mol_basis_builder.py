import numpy as np
import basis_set_exchange as bse


# BSE dictionary uses atomic numbers (as strings) for keys. 
# This is a minimal lookup to map atomic symbols to numbers.
ATOMIC_NUMBERS = {
    'H': 1, 'HE': 2, 'LI': 3, 'BE': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'NE': 10,
    'NA': 11, 'MG': 12, 'AL': 13, 'SI': 14, 'P': 15, 'S': 16, 'CL': 17, 'AR': 18
}

ANGSTROM_TO_BOHR = 1.8897259886


def double_factorial(n):
    if n <= 0:
        return 1

    result = 1
    for value in range(n, 0, -2):
        result *= value
    return result


class Shell:
    def __init__(self, am, center, atom_index, exponents, coefficients, function_index):
        self.am = am
        self.center = center
        self.atom_index = atom_index
        self.function_index = function_index
        
        self.exponents = np.array(exponents)
        self.coefficients = np.array(coefficients)
        self.nprimitive = len(self.exponents)

        n_prim = self.nprimitive
        exponents = self.exponents
        coefficients = self.coefficients

        n_prim = self.nprimitive
        exponents = self.exponents
        coefficients = self.coefficients

        # Compute the overlap between primitives for contraction normalization
        S = np.zeros((n_prim, n_prim))
        for i in range(n_prim):
            for j in range(n_prim):
                S[i, j] = (2.0 * np.sqrt(exponents[i] * exponents[j]) / 
                            (exponents[i] + exponents[j]))**(am + 1.5)

        # Match Psi4's common angular normalization for Cartesian shells.
        angular_norm = 1.0 / np.sqrt(double_factorial(2 * am - 1))

        self.norm_prim = angular_norm * (2.0 * exponents / np.pi)**0.75 * (4.0 * exponents)**(am / 2.0)
        self.norm_cont = 1.0 / np.sqrt(np.sum(np.outer(coefficients, coefficients) * S))

    @property
    def effective_coef(self):
        return self.norm_cont * self.coefficients * self.norm_prim


class Basis:
    def __init__(self):
        self.shells: list[Shell] = []
        self.nao = 0
        self.name = ""
        self.ndocc = 0

    def nshell(self):
        return len(self.shells)

    def shell(self, idx):
        return self.shells[idx]


class Molecule:
    def __init__(self, xyz_string: str):
        lines = xyz_string.strip().split('\n')
        self.atoms = []
        self.coords = []
        
        # Handle optional header lines in standard XYZ files
        start_idx = 0
        if len(lines[0].split()) == 1: # First line is atom count
            start_idx = 2 if len(lines) > 1 else 1
            
        for line in lines[start_idx:]:
            parts = line.split()
            if not parts: continue
            self.atoms.append(parts[0])
            self.coords.append([float(x) for x in parts[1:4]])
        
        self.coords = np.array(self.coords)

        n_electrons = sum(ATOMIC_NUMBERS[atom.upper()] for atom in self.atoms)
        if n_electrons % 2 != 0:
            raise ValueError("RHF requires a closed-shell molecule with an even electron count.")
        self.ndocc = n_electrons // 2

        # Optionally set all to be called upon initialization
        self.V_nn = self.get_nuclear_repulsion()
        self.set_scf_settings()


    def set_scf_settings(self, max_iter=100, e_conv=1e-6, startup_iter=5, grad_max=1e-6, grad_rms=1e-6, verbose=0):
        self.max_iter = max_iter
        self.startup_iter = startup_iter
        self.e_conv = e_conv
        self.grad_max = grad_max
        self.grad_rms = grad_rms
        self.verbose = verbose


    def get_nuclear_repulsion(self):
        V_nn = 0.0
        n_atoms = len(self.atoms)
        
        coords_bohr = self.coords * ANGSTROM_TO_BOHR
        
        for i in range(n_atoms):
            z_i = ATOMIC_NUMBERS[self.atoms[i].upper()]
            
            for j in range(i + 1, n_atoms):
                z_j = ATOMIC_NUMBERS[self.atoms[j].upper()]
                
                dist = np.linalg.norm(coords_bohr[i] - coords_bohr[j])
                
                V_nn += (z_i * z_j) / dist
                
        self.V_nn = V_nn
        return V_nn


    def build_basis(self, basis_name):
        basis = Basis()
        
        unique_atoms = list(set(self.atoms))
        bse_dict = bse.get_basis(basis_name, elements=unique_atoms, fmt=None)

        function_index = 0
        coords_bohr = self.coords * ANGSTROM_TO_BOHR

        for atom_index, (atom, coord) in enumerate(zip(self.atoms, coords_bohr)):
            z_str = str(ATOMIC_NUMBERS.get(atom.upper()))
            
            if z_str not in bse_dict['elements']:
                raise ValueError(f"Basis {basis_name} does not have element {atom}")

            atom_data = bse_dict['elements'][z_str]

            for shell in atom_data['electron_shells']:
                exponents = np.array([float(e) for e in shell['exponents']])
                
                coefficients_list = shell['coefficients']
                am_list = shell['angular_momentum']

                for idx, coefficients in enumerate(coefficients_list):
                    # If there is only one am value, it applies to all coefficients in the shell 
                    if len(am_list) == 1:
                        am = am_list[0]
                    # Otherwise, BSE groups SP shells together (am = [0, 1]). 
                    elif len(am_list) == len(coefficients_list):
                        am = am_list[idx]
                    else:
                        raise ValueError(
                            f"Non-standard data organization for {basis_name}, please check. am_list: {am_list}, coefficients_list: {coefficients_list}")

                    am = int(am)
                    coefficients = np.array([float(c) for c in coefficients])

                    mask = np.abs(exponents) > 1e-6
                    exponents = exponents[mask]
                    coefficients = coefficients[mask]

                    if len(exponents) == 0:
                        continue

                    basis.shells.append(Shell(
                        am=am,
                        center=coord,
                        atom_index=atom_index,
                        exponents=exponents,
                        coefficients=coefficients,
                        function_index=function_index
                    ))
                    
                    # Advance function index by the number of Cartesian functions: (L+1)*(L+2)/2
                    function_index += (am + 1) * (am + 2) // 2
                    
        basis.nao = function_index
        basis.name = basis_name.upper()
        return basis
