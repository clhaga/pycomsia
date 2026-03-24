import logging

from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor, rdFMCS, rdDistGeom
from rdkit.Chem.rdForceFieldHelpers import (
    MMFFGetMoleculeProperties,
    MMFFGetMoleculeForceField,
)

logger = logging.getLogger(__name__)


class MoleculeAligner:
    def __init__(self):
        self.template = None
        self.core = None

    def align_molecules(self, train_smiles_list, pred_smiles_list=None):
        """
        ALIGNMENT DISABLED: Assuming SDF molecules are pre-aligned.
        This method now returns molecules without alignment.

        Returns a list of tuples: (mol, is_training)
        """
        logger.info("Alignment disabled - using pre-aligned SDF molecules")
        return []  # Return empty list since SDF molecules are handled directly

        # # COMMENTED OUT - Original alignment code
        # if not train_smiles_list:
        #     return []

        # print("Extracting SMILES...")
        # # Create combined list with tracking info
        # combined_data = [(smi, True) for smi in train_smiles_list]
        # if pred_smiles_list:
        #     combined_data.extend([(smi, False) for smi in pred_smiles_list])
        #
        # # Convert SMILES to molecules while preserving order
        # mols_with_info = [(Chem.MolFromSmiles(smi), is_training)
        #                  for smi, is_training in combined_data]
        #
        # if not mols_with_info or any(mol[0] is None for mol in mols_with_info):
        #     print("Error: Invalid SMILES strings provided.")
        #     return []

        # # Find the largest molecule (preferably from training set)
        # largest_mol_index = 0
        # largest_mol_num_atoms = 0
        # for i, (mol, is_training) in enumerate(mols_with_info):
        #     num_atoms = mol.GetNumAtoms()
        #     # Prefer training set molecules as template
        #     if num_atoms > largest_mol_num_atoms or \
        #        (num_atoms == largest_mol_num_atoms and is_training and not mols_with_info[largest_mol_index][1]):
        #         largest_mol_num_atoms = num_atoms
        #         largest_mol_index = i

        # # Use the largest molecule as the template
        # self.template = Chem.AddHs(mols_with_info[largest_mol_index][0])
        # ps = rdDistGeom.ETKDGv3()
        # ps.randomSeed = 0xf00d
        # #ps.randomSeed = 42
        # ps.useBasicKnowledge = True
        # ps.ignoreSmoothingFailures = True
        # AllChem.EmbedMolecule(self.template, ps)
        # # Initialize results list with None values to maintain order
        # aligned_results = [(None, info[1]) for info in mols_with_info]
        # # Add template molecule to its correct position
        # aligned_results[largest_mol_index] = (Chem.Mol(self.template),
        #                                     mols_with_info[largest_mol_index][1])

        # print("Aligning molecules...")
        # for i, (mol, is_training) in enumerate(mols_with_info):
        #     if i == largest_mol_index:  # Skip the template molecule
        #         continue

        #     mcs = rdFMCS.FindMCS([self.template, mol],
        #                         completeRingsOnly=True,
        #                         ringMatchesRingOnly=False,
        #                         atomCompare=rdFMCS.AtomCompare.CompareAny,
        #                         bondCompare=rdFMCS.BondCompare.CompareAny)
        #                         #ringCompare=rdFMCS.RingCompare.PermissiveRingFusion)

        #     if mcs.numAtoms > 0:
        #         mcs_smarts = mcs.smartsString
        #         core = Chem.MolFromSmarts(mcs_smarts)
        #         mol = Chem.AddHs(mol)
        #         template_match = self.template.GetSubstructMatch(core)
        #         mol_match = mol.GetSubstructMatch(core)

        #         if template_match and mol_match:
        #             cmap = {mol_match[k]: self.template.GetConformer().GetAtomPosition(template_match[k])
        #                    for k in range(len(template_match))}
        #             try:
        #                 #AllChem.EmbedMolecule(mol, coordMap=cmap,
        #                 #                    ignoreSmoothingFailures=True, useRandomCoords=True)
        #                 AllChem.EmbedMolecule(mol, coordMap=cmap,
        #                                       ignoreSmoothingFailures=True, useRandomCoords=True, useBasicKnowledge = True)
        #
        #                 mcp = Chem.Mol(mol)
        #                 # mmff_props = MMFFGetMoleculeProperties(mcp)
        #                 # ff = MMFFGetMoleculeForceField(mcp, mmff_props)
        #                 # for at_idx in mol_match:
        #                 #     ff.MMFFAddPositionConstraint(at_idx, 0.5, 200)
        #                 # max_iters = 100
        #                 # while ff.Minimize(maxIts=100) and max_iters > 0:
        #                 #     max_iters -= 1
        #                 aligned_results[i] = (mcp, is_training)
        #             except Exception as e:
        #                 print(f"Embedding/FF optimization failed for molecule {i}: {e}")
        #     else:
        #         print(f"No MCS found between template and molecule {i} using rdFMCS")

        # return aligned_results

    def get_template(self):
        return self.template

    def get_core(self):
        return self.core
