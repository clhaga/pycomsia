import logging

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self):
        self.smiles_list = []
        self.activities = None
        self.molecules = None

    def load_sdf_data(self, sdf_file, activity_property=None, is_training=True):
        """
        Load SMILES, 3D molecules, and activity data from SDF

        Parameters:
        -----------
        sdf_file : str
            Path to the SDF file
        activity_property : str, optional
            Name of the property in the SDF file that contains the activity data
            Required if is_training=True
        is_training : bool, default=True
            If True, loads both SMILES and activities
            If False, loads only SMILES (for prediction dataset)

        Returns:
        --------
        tuple
            If is_training=True: (smiles_list, [(mol, is_training)], activities)
            If is_training=False: (smiles_list, [(mol, is_training)], None)
        """
        # Read the SDF file
        logger.info("Reading SDF file...")
        suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
        self.smiles_list = []
        mols_with_flag = []  # List of tuples: (mol, is_training)
        self.activities = []
        original_mols = []  # Store original molecules for protonation validation
        _mol_idx = 0

        for mol in suppl:
            if mol is not None:
                # Store original molecule for protonation validation
                original_mols.append(mol)

                # Count hydrogen atoms and log per-molecule status at DEBUG level
                num_hs = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1)
                mol_name = (
                    mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{_mol_idx}"
                )
                logger.debug(
                    f"  Mol {_mol_idx} ({mol_name}): {mol.GetNumAtoms()} atoms, "
                    f"{num_hs} explicit H"
                )
                _mol_idx += 1

                if num_hs > 0:
                    # Molecule already has hydrogens, validate they have coordinates
                    mol_with_hs = mol
                    conf = mol_with_hs.GetConformer()
                    atoms_to_remove = []

                    for i in range(mol_with_hs.GetNumAtoms()):
                        atom = mol_with_hs.GetAtomWithIdx(i)
                        try:
                            pos = conf.GetAtomPosition(i)
                            # Check for invalid coordinates (NaN, infinity, or suspicious zeros)
                            if (
                                str(pos.x) == "nan"
                                or str(pos.y) == "nan"
                                or str(pos.z) == "nan"
                                or abs(pos.x) > 1000
                                or abs(pos.y) > 1000
                                or abs(pos.z) > 1000
                            ):
                                if (
                                    atom.GetAtomicNum() == 1
                                ):  # Only remove hydrogens with bad coordinates
                                    atoms_to_remove.append(i)
                        except:
                            if (
                                atom.GetAtomicNum() == 1
                            ):  # Only remove hydrogens with coordinate errors
                                atoms_to_remove.append(i)

                    # Remove problematic hydrogen atoms (preserve heavy atom structure)
                    if atoms_to_remove:
                        editable_mol = Chem.EditableMol(mol_with_hs)
                        for idx in sorted(atoms_to_remove, reverse=True):
                            editable_mol.RemoveAtom(idx)
                        mol_with_hs = editable_mol.GetMol()
                else:
                    # No hydrogens present - add them but preserve heavy atom coordinates
                    mol_with_hs = Chem.AddHs(mol, addCoords=False)

                    # Try to add hydrogen coordinates without changing heavy atoms
                    try:
                        from rdkit.Chem import rdDistGeom

                        # Use constrained embedding to preserve heavy atom positions
                        heavy_atom_coords = {}
                        orig_conf = mol.GetConformer()

                        # Store original heavy atom coordinates
                        for i in range(mol.GetNumAtoms()):
                            pos = orig_conf.GetAtomPosition(i)
                            heavy_atom_coords[i] = (pos.x, pos.y, pos.z)

                        # Embed with constraints to keep heavy atoms fixed
                        AllChem.EmbedMolecule(
                            mol_with_hs,
                            useExpTorsionAnglePrefs=True,
                            useBasicKnowledge=True,
                            randomSeed=42,
                        )

                        # Restore original heavy atom coordinates
                        new_conf = mol_with_hs.GetConformer()
                        for old_idx in heavy_atom_coords:
                            x, y, z = heavy_atom_coords[old_idx]
                            new_conf.SetAtomPosition(old_idx, [x, y, z])

                    except:
                        # If coordinate generation fails, fall back to molecule without hydrogens
                        mol_with_hs = mol

                # Generate SMILES from the processed molecule
                smiles = Chem.MolToSmiles(mol_with_hs)
                self.smiles_list.append(smiles)

                # Store the 3D molecule and the is_training flag
                mols_with_flag.append((mol_with_hs, is_training))

                # Extract activity if in training mode (from original mol to preserve properties)
                if is_training:
                    if activity_property is None:
                        raise ValueError(
                            "Activity property must be specified for training data"
                        )
                    if activity_property not in mol.GetPropNames():
                        raise ValueError(
                            f"Activity property '{activity_property}' not found in SDF file"
                        )
                    activity = float(mol.GetProp(activity_property))
                    self.activities.append(activity)

        # Validate protonation state of original input molecules (before hydrogen processing)
        self._validate_protonation_state_from_originals(original_mols)

        if not is_training:
            self.activities = None

        n_loaded = len(mols_with_flag)
        logger.debug(f"Loaded {n_loaded} molecules from {sdf_file}")
        if is_training and self.activities:
            acts = self.activities
            logger.debug(
                f"Activity range: min={min(acts):.3f}  max={max(acts):.3f}  "
                f"mean={sum(acts)/len(acts):.3f}"
            )

        return self.smiles_list, mols_with_flag, self.activities

    def _validate_protonation_state_from_originals(self, original_mols):
        """
        Validate that molecules appear to be properly protonated based on original input.

        Parameters:
        -----------
        original_mols : list
            List of original RDKit molecules as loaded from SDF (before hydrogen processing)

        Raises:
        -------
        ValueError
            If molecules appear to be significantly under-protonated
        """
        if not original_mols:
            return

        under_protonated_count = 0
        total_molecules = len(original_mols)
        protonation_stats = []

        for mol in original_mols:
            if mol is None:
                continue

            # Count heavy atoms (non-hydrogen) and hydrogen atoms
            heavy_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1)
            hydrogen_atoms = sum(
                1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1
            )

            if heavy_atoms == 0:
                continue

            # Calculate hydrogen-to-heavy-atom ratio
            h_to_heavy_ratio = hydrogen_atoms / heavy_atoms if heavy_atoms > 0 else 0
            protonation_stats.append(h_to_heavy_ratio)

            # Typical organic drug-like molecules should have H/heavy ratio > 0.5
            # Very under-protonated molecules will have ratio < 0.3
            if h_to_heavy_ratio < 0.3:
                under_protonated_count += 1

        if not protonation_stats:
            return

        # Calculate statistics
        avg_h_ratio = sum(protonation_stats) / len(protonation_stats)
        under_protonated_fraction = under_protonated_count / total_molecules

        # Raise error if dataset appears significantly under-protonated
        if under_protonated_fraction > 0.5 or avg_h_ratio < 0.4:
            raise ValueError(
                f"Input SDF appears to contain under-protonated molecules!\n"
                f"\n"
                f"Protonation analysis:\n"
                f"  - {under_protonated_count}/{total_molecules} molecules appear under-protonated\n"
                f"  - Average H/heavy-atom ratio: {avg_h_ratio:.2f} (expect ~1.0-2.0 for protonated molecules)\n"
                f"\n"
                f"PyCoMSIA requires properly protonated input structures for accurate field calculations.\n"
                f"\n"
                f"Solutions:\n"
                f"  1. Protonate your ligands using tools like:\n"
                f"     - ChemAxon's cxcalc: 'cxcalc majorms -H 7.4 input.sdf > output.sdf'\n"
                f"     - OpenEye OMEGA: 'omega2 -in input.sdf -out output.sdf -ph 7.4'\n"
                f"     - Schrödinger LigPrep with ionization at pH 7.4\n"
                f"     - RDKit: Chem.AddHs() followed by coordinate embedding\n"
                f"  2. Ensure your SDF contains explicit hydrogen atoms\n"
                f"  3. Verify protonation states are appropriate for your target pH\n"
                f"\n"
                f"If you believe your molecules are correctly protonated, you can bypass this check\n"
                f"by manually adding hydrogens before running PyCoMSIA."
            )
