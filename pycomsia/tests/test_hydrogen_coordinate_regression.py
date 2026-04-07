"""
Regression test for hydrogen coordinate bug fix and PLS analysis stability.
This test ensures that:
1. Hydrogen atoms have valid coordinates after loading
2. PLS analysis produces reasonable Q2 scores 
3. The hydrogen fix doesn't negatively impact analysis quality
"""

import pytest
import tempfile
import os
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

from pycomsia.data_loader import DataLoader
from pycomsia.molecular_grid_calculator import MolecularGridCalculator
from pycomsia.molecular_field_calculator import MolecularFieldCalculator
from pycomsia.pls_analysis import PLSAnalysis


class TestHydrogenCoordinateRegression:
    """Tests for hydrogen coordinate handling and PLS analysis stability."""
    
    def test_hydrogen_coordinate_validation(self):
        """Test that all atoms (including hydrogens) have valid coordinates after loading."""
        # Create test molecules with known activities
        molecules_data = [
            ('CCO', 5.0),      # ethanol
            ('CCCO', 6.0),     # propanol  
            ('CCCCO', 7.0),    # butanol
            ('CCCCCO', 8.0),   # pentanol
            ('CCN', 5.5),      # ethylamine  
            ('CC(=O)O', 6.5)   # acetic acid
        ]

        # Create SDF with multiple molecules
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as f:
            writer = Chem.SDWriter(f.name)
            for smiles, activity in molecules_data:
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)
                mol.SetProp('activity', str(activity))
                writer.write(mol)
            writer.close()
            temp_file = f.name

        try:
            # Load using DataLoader with our hydrogen coordinate fix
            loader = DataLoader()
            smiles_list, molecules_with_flags, activities = loader.load_sdf_data(
                temp_file, 'activity', is_training=True
            )
            
            assert len(molecules_with_flags) == len(molecules_data), "All molecules should be loaded"
            
            # Check each molecule for coordinate validity
            for i, (mol, is_training) in enumerate(molecules_with_flags):
                assert mol is not None, f"Molecule {i} should not be None"
                
                # Count hydrogen atoms
                num_hydrogens = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1)
                
                if num_hydrogens > 0:
                    # Check that all atoms have valid, accessible coordinates
                    conf = mol.GetConformer()
                    
                    for j in range(mol.GetNumAtoms()):
                        atom = mol.GetAtomWithIdx(j)
                        
                        # This should not raise an exception
                        pos = conf.GetAtomPosition(j)
                        
                        # Coordinates should be finite numbers
                        assert np.isfinite(pos.x), f"Molecule {i}, atom {j} has non-finite x coordinate"
                        assert np.isfinite(pos.y), f"Molecule {i}, atom {j} has non-finite y coordinate" 
                        assert np.isfinite(pos.z), f"Molecule {i}, atom {j} has non-finite z coordinate"
                        
                        # Coordinates should be reasonable (not extreme values)
                        assert abs(pos.x) < 100, f"Molecule {i}, atom {j} has extreme x coordinate: {pos.x}"
                        assert abs(pos.y) < 100, f"Molecule {i}, atom {j} has extreme y coordinate: {pos.y}"
                        assert abs(pos.z) < 100, f"Molecule {i}, atom {j} has extreme z coordinate: {pos.z}"

        finally:
            os.unlink(temp_file)
    
    def test_pls_analysis_stability_ace_dataset(self):
        """Test that PLS analysis produces reasonable Q2 scores with real ACE data."""
        ace_train_file = Path(__file__).parent.parent / "data" / "ACE_train.sdf"
        
        # Skip if ACE dataset is not available
        if not ace_train_file.exists():
            pytest.skip("ACE training dataset not available")
        
        # Load ACE dataset
        loader = DataLoader()
        smiles, mols_flags, activities = loader.load_sdf_data(
            str(ace_train_file), 'Activity', is_training=True
        )
        
        assert len(mols_flags) > 50, "ACE dataset should have sufficient molecules"
        assert len(activities) == len(mols_flags), "Activities should match molecule count"
        
        # Calculate grid and fields
        grid_calc = MolecularGridCalculator()
        grid_data = grid_calc.generate_grid(mols_flags, resolution=2.0, padding=4.0)
        
        field_calc = MolecularFieldCalculator()
        field_results = field_calc.calc_field(mols_flags, grid_data[0], grid_data[1], grid_data[2])
        
        train_fields = field_results['train_fields']
        
        # Run PLS analysis
        pls = PLSAnalysis()
        X_train, X_pred = pls.convert_fields_to_X(train_fields)
        
        assert X_train is not None, "Feature matrix should be created"
        assert X_train.shape[0] == len(activities), "Feature matrix should match molecule count"
        
        # Perform LOO analysis
        pls.perform_loo_analysis(activities, max_components=5)
        
        # Check Q2 scores are reasonable
        assert hasattr(pls, 'q2_scores'), "Q2 scores should be calculated"
        assert len(pls.q2_scores) > 0, "Should have Q2 scores"
        
        # Q2 scores should be reasonable for ACE dataset
        # ACE is a well-studied dataset that should produce decent Q2 scores
        best_q2 = max(pls.q2_scores)
        assert best_q2 > 0.3, f"Best Q2 score should be > 0.3 for ACE dataset, got {best_q2}"
        
        # Optimal number of components should be reasonable
        assert pls.optimal_n_components > 0, "Should find optimal number of components"
        assert pls.optimal_n_components <= 5, "Optimal components should be within tested range"
        
    def test_hydrogen_processing_preserves_analysis_quality(self):
        """Test that hydrogen coordinate processing doesn't degrade analysis quality."""
        # This is a placeholder test that could be expanded to compare
        # Q2 scores before/after hydrogen fixes if needed
        
        # For now, we rely on the ACE dataset test to validate that
        # the hydrogen processing doesn't break PLS analysis
        pass
    
    def test_coordinate_edge_cases(self):
        """Test edge cases in coordinate handling."""
        # Test molecule with no hydrogens initially
        mol = Chem.MolFromSmiles('C')  # methane without explicit H
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        
        # Write to SDF
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as f:
            writer = Chem.SDWriter(f.name)
            mol.SetProp('activity', '5.0')
            writer.write(mol)
            writer.close()
            temp_file = f.name
        
        try:
            loader = DataLoader()
            smiles, mols_flags, activities = loader.load_sdf_data(
                temp_file, 'activity', is_training=True
            )
            
            assert len(mols_flags) == 1, "Should load single molecule"
            mol_processed, _ = mols_flags[0]
            
            # Should have hydrogen atoms 
            num_hs = sum(1 for atom in mol_processed.GetAtoms() if atom.GetAtomicNum() == 1)
            assert num_hs > 0, "Should have hydrogen atoms"
            
            # All atoms should have valid coordinates
            conf = mol_processed.GetConformer()
            for i in range(mol_processed.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                assert np.isfinite(pos.x) and np.isfinite(pos.y) and np.isfinite(pos.z), \
                    f"Atom {i} should have finite coordinates"
                    
        finally:
            os.unlink(temp_file)

    def test_protonation_state_validation_catches_unprotonated_molecules(self):
        """Test that the protonation validation correctly identifies under-protonated molecules."""
        # Create test molecules WITHOUT hydrogens (common mistake)
        molecules_data = [
            'CCO',      # ethanol - no explicit H
            'CCCO',     # propanol - no explicit H  
            'CCCCO',    # butanol - no explicit H
            'CCN',      # ethylamine - no explicit H
        ]

        # Create SDF with unprotonated molecules
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as f:
            writer = Chem.SDWriter(f.name)
            for i, smiles in enumerate(molecules_data):
                mol = Chem.MolFromSmiles(smiles)
                # Do NOT add hydrogens - this simulates user error
                AllChem.EmbedMolecule(mol, randomSeed=42)
                mol.SetProp('activity', str(5.0 + i))
                writer.write(mol)
            writer.close()
            temp_file = f.name

        try:
            # Loading should raise an error due to under-protonation
            loader = DataLoader()
            with pytest.raises(ValueError, match=r"under-protonated molecules"):
                loader.load_sdf_data(temp_file, 'activity', is_training=True)
                    
        finally:
            os.unlink(temp_file)
            
    def test_protonation_state_validation_passes_protonated_molecules(self):
        """Test that properly protonated molecules pass validation."""
        # Create test molecules WITH hydrogens
        molecules_data = [
            'CCO',      # ethanol
            'CCCO',     # propanol  
            'CCCCO',    # butanol
            'CCN',      # ethylamine
        ]

        # Create SDF with properly protonated molecules
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as f:
            writer = Chem.SDWriter(f.name)
            for i, smiles in enumerate(molecules_data):
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)  # ADD hydrogens - this is correct
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)
                mol.SetProp('activity', str(5.0 + i))
                writer.write(mol)
            writer.close()
            temp_file = f.name

        try:
            # Loading should succeed with protonated molecules
            loader = DataLoader()
            smiles, mols_flags, activities = loader.load_sdf_data(
                temp_file, 'activity', is_training=True
            )
            
            # Should successfully load all molecules
            assert len(mols_flags) == len(molecules_data)
            assert len(activities) == len(molecules_data)
                    
        finally:
            os.unlink(temp_file)
            
    def test_protonation_validation_with_mixed_molecules(self):
        """Test protonation validation with a mix of protonated and unprotonated molecules."""
        # Create mixed dataset - some protonated, some not
        molecules_data = [
            ('CCO', True),    # ethanol - will be protonated
            ('CCCO', False),  # propanol - will NOT be protonated  
            ('CCCCO', False), # butanol - will NOT be protonated
            ('CCN', False),   # ethylamine - will NOT be protonated
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as f:
            writer = Chem.SDWriter(f.name)
            for i, (smiles, add_hs) in enumerate(molecules_data):
                mol = Chem.MolFromSmiles(smiles)
                if add_hs:
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, randomSeed=42)
                    AllChem.MMFFOptimizeMolecule(mol)
                else:
                    # Don't add hydrogens
                    AllChem.EmbedMolecule(mol, randomSeed=42)
                mol.SetProp('activity', str(5.0 + i))
                writer.write(mol)
            writer.close()
            temp_file = f.name

        try:
            # Should raise error since 3/4 molecules are under-protonated (75% > 50% threshold)
            loader = DataLoader()
            with pytest.raises(ValueError, match=r"under-protonated molecules"):
                loader.load_sdf_data(temp_file, 'activity', is_training=True)
                    
        finally:
            os.unlink(temp_file)