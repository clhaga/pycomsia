import logging

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
import numpy as np

logger = logging.getLogger(__name__)


class MolecularFieldCalculator:
    def __init__(self):
        self.ALPHA = 0.3
        self.TOLERANCE = 1e-4

        self.donor_smarts = [
            # "[N&!H0&v3,N&!H0&+1&v4,O&H1&+0,S&H1&+0,n&H1&+0]"
            "[$([#7,#8,#15,#16]);H]"
        ]

        self.acceptor_smarts = [
            # "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$("
            # "[N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]"
            "[#8&!$(*~N~[OD1]),#7&H0;!$([D4]);!$([D3]-\*=,:[$([#7,#8,#15,#16])])]"
        ]

        self.donor_patterns = [Chem.MolFromSmarts(smart) for smart in self.donor_smarts]
        self.acceptor_patterns = [
            Chem.MolFromSmarts(smart) for smart in self.acceptor_smarts
        ]

    def calc_field(self, aligned_results, grid_spacing, grid_dimensions, grid_origin):
        """Calculate molecular fields for both training and prediction sets."""
        train_fields = {
            "steric_field": [],
            "electrostatic_field": [],
            "hydrophobic_field": [],
            "hbond_donor_field": [],
            "hbond_acceptor_field": [],
        }

        pred_fields = {
            "steric_field": [],
            "electrostatic_field": [],
            "hydrophobic_field": [],
            "hbond_donor_field": [],
            "hbond_acceptor_field": [],
        }

        # Calculate fields for all molecules
        n_train, n_pred = 0, 0
        for mol, is_training in aligned_results:
            if mol is not None:
                mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else "?"
                logger.debug(
                    f"  Calculating fields for {'train' if is_training else 'pred'} "
                    f"mol '{mol_name}' ({mol.GetNumAtoms()} atoms)"
                )
                fields = self._calc_single_molecule_field(
                    mol, grid_spacing, grid_dimensions, grid_origin
                )
                # Sort into training or prediction fields
                target_fields = train_fields if is_training else pred_fields
                for field_name in target_fields:
                    target_fields[field_name].append(fields[field_name].flatten())
                if is_training:
                    n_train += 1
                else:
                    n_pred += 1
        logger.debug(
            f"Field calculation complete: {n_train} train, {n_pred} pred molecules"
        )

        return {"train_fields": train_fields, "pred_fields": pred_fields}

    def _calc_single_molecule_field(
        self, mol, grid_spacing, grid_dimensions, grid_origin
    ):
        """Calculate fields for a single molecule."""
        dx, dy, dz = grid_spacing
        nx, ny, nz = grid_dimensions
        x0, y0, z0 = grid_origin

        # Initialize fields
        steric_field = np.zeros((nx, ny, nz))
        electrostatic_field = np.zeros((nx, ny, nz))
        hydrophobic_field = np.zeros((nx, ny, nz))
        hbond_donor_field = np.zeros((nx, ny, nz))
        hbond_acceptor_field = np.zeros((nx, ny, nz))

        # Get atom properties
        conformer = mol.GetConformer()
        atom_coords = np.array(
            [list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
        )

        # Compute Gasteiger charges
        AllChem.ComputeGasteigerCharges(mol)
        atom_charges = np.array(
            [float(atom.GetProp("_GasteigerCharge")) for atom in mol.GetAtoms()]
        )
        atom_charges = np.nan_to_num(atom_charges)

        # Calculate hydrophobicity
        atom_contribs = rdMolDescriptors._CalcCrippenContribs(mol)
        atom_hydrophobicities = np.array([contrib[0] for contrib in atom_contribs])

        # Get VdW radii
        ptable = Chem.GetPeriodicTable()
        atom_vdw = np.array(
            [ptable.GetRvdw(atom.GetAtomicNum()) for atom in mol.GetAtoms()]
        )

        # Generate grid points
        x = x0 + np.arange(nx) * dx
        y = y0 + np.arange(ny) * dy
        z = z0 + np.arange(nz) * dz
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing="ij")
        grid_points = np.vstack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel())).T

        # Calculate atomic contributions
        for i in range(mol.GetNumAtoms()):
            displacement = grid_points - atom_coords[i]
            distances_sq = np.sum(displacement**2, axis=1)
            gaussian = np.exp(-self.ALPHA * distances_sq)

            steric_field.ravel()[:] -= (atom_vdw[i] ** 3) * gaussian
            electrostatic_field.ravel()[:] += (atom_charges[i] * gaussian) * 337
            hydrophobic_field.ravel()[:] += (atom_hydrophobicities[i] * gaussian) * 25

        # Calculate H-bond fields
        total_donors, total_acceptors = 0, 0
        for pattern in self.donor_patterns:
            matches = mol.GetSubstructMatches(pattern, uniquify=1)
            pseudoatoms = self.generate_pseudoatoms(mol, matches, True)
            total_donors += len(pseudoatoms)
            for pos in pseudoatoms:
                displacement = grid_points - pos
                distances_sq = np.sum(displacement**2, axis=1)
                gaussian = np.exp(-self.ALPHA * distances_sq)
                hbond_donor_field.ravel()[:] += gaussian * 20

        for pattern in self.acceptor_patterns:
            matches = mol.GetSubstructMatches(pattern, uniquify=1)
            pseudoatoms = self.generate_pseudoatoms(mol, matches, False)
            total_acceptors += len(pseudoatoms)
            for pos in pseudoatoms:
                displacement = grid_points - pos
                distances_sq = np.sum(displacement**2, axis=1)
                gaussian = np.exp(-self.ALPHA * distances_sq)
                hbond_acceptor_field.ravel()[:] += gaussian * 20

        logger.debug(
            f"    H-bond pseudoatoms: {total_donors} donors, {total_acceptors} acceptors"
        )

        return {
            "steric_field": steric_field,
            "electrostatic_field": electrostatic_field,
            "hydrophobic_field": hydrophobic_field,
            "hbond_donor_field": hbond_donor_field,
            "hbond_acceptor_field": hbond_acceptor_field,
        }

    def generate_pseudoatoms(self, mol, matches, is_donor):
        """Generate pseudoatom positions for H-bond donors/acceptors."""
        pseudoatom_positions = []

        for match in matches:
            atom_idx = match[0]
            atom = mol.GetAtomWithIdx(atom_idx)
            atom_pos = np.array(mol.GetConformer().GetAtomPosition(atom_idx))
            positions = self._get_hbond_positions(mol, atom, atom_pos, is_donor)

            # Pass the donor/acceptor atom index
            filtered_positions = self._filter_positions(
                mol, positions, is_donor, donor_acceptor_idx=atom_idx
            )

            # Note: Suppressed warning about filtered donor positions - this is normal when
            # atoms are too close together and indicates proper steric filtering
            # if not filtered_positions and is_donor:
            #     print(f"Warning: All donor positions filtered out for atom {atom_idx}")

            pseudoatom_positions.extend(filtered_positions)

        return pseudoatom_positions

    def _get_hbond_positions(self, mol, atom, atom_pos, is_donor):
        """Generate positions for hydrogen bond interactions."""
        positions = []
        distance = 1.9  # Ångstroms

        if is_donor:
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 1:
                    h_pos = np.array(
                        mol.GetConformer().GetAtomPosition(neighbor.GetIdx())
                    )
                    vector = h_pos - atom_pos
                    vector /= np.linalg.norm(vector)
                    positions.append(atom_pos + distance * vector)
        else:
            vectors = self._get_hybridization_vectors(atom)
            for vector in vectors:
                positions.append(atom_pos + distance * vector)
        return positions

    def _get_hybridization_vectors(self, atom):
        """Get vectors based on atom hybridization."""
        vectors = []
        hybridization = atom.GetHybridization()

        if hybridization == Chem.rdchem.HybridizationType.SP2:
            vectors = [
                np.array([1, 0, 0]),
                np.array([-0.5, np.sqrt(3) / 2, 0]),
                np.array([-0.5, -np.sqrt(3) / 2, 0]),
            ]
        elif hybridization == Chem.rdchem.HybridizationType.SP3:
            vectors = [
                np.array([1, 1, 1]),
                np.array([-1, -1, 1]),
                np.array([-1, 1, -1]),
                np.array([1, -1, -1]),
            ]
        else:
            vectors = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

        return [vec / np.linalg.norm(vec) for vec in vectors]

    def _filter_positions(self, mol, positions, is_donor, donor_acceptor_idx=None):
        """
        Remove positions that are too close to any non-hydrogen atom in the molecule,
        excluding the donor/acceptor atom itself.

        Args:
            mol: RDKit molecule
            positions: List of candidate pseudoatom positions
            donor_acceptor_idx: Index of the donor/acceptor atom to exclude from checking
        """
        filtered_positions = []

        # Get positions of all non-hydrogen atoms except the donor/acceptor
        atom_positions = []
        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            # Skip hydrogens and the donor/acceptor atom
            if atom.GetAtomicNum() != 1 and i != donor_acceptor_idx:
                atom_positions.append(list(mol.GetConformer().GetAtomPosition(i)))

        atom_positions = np.array(atom_positions)

        min_distance = 1.5 if is_donor else 1.8  # Threshold for minimum distance

        for pos in positions:
            distances = np.linalg.norm(atom_positions - pos, axis=1)
            if np.all(distances > min_distance):
                filtered_positions.append(pos)

        return filtered_positions
