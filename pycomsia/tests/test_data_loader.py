"""
Unit tests for DataLoader module.
"""

import pytest
import numpy as np
from pathlib import Path
from rdkit import Chem

from pycomsia.data_loader import DataLoader


@pytest.fixture
def test_data_dir():
    """Get path to test data directory."""
    return Path(__file__).parent.parent / "data"


@pytest.fixture
def data_loader():
    """Create DataLoader instance."""
    return DataLoader()


@pytest.fixture
def ace_train_file(test_data_dir):
    """Path to ACE training SDF file."""
    return test_data_dir / "ACE_train.sdf"


@pytest.fixture
def ace_test_file(test_data_dir):
    """Path to ACE test SDF file."""
    return test_data_dir / "ACE_test.sdf"


class TestDataLoader:
    """Test DataLoader functionality."""
    
    def test_init(self, data_loader):
        """Test DataLoader initialization."""
        assert data_loader.smiles_list == []
        assert data_loader.activities is None
        assert data_loader.molecules is None
    
    def test_load_training_data(self, data_loader, ace_train_file):
        """Test loading training data from SDF file."""
        smiles_list, mols, activities = data_loader.load_sdf_data(
            str(ace_train_file), "Activity", is_training=True
        )
        
        # Basic validation
        assert len(smiles_list) > 0
        assert len(mols) == len(smiles_list)
        assert len(activities) == len(smiles_list)
        assert all(isinstance(activity, (int, float)) for activity in activities)
        
        # Verify molecules have conformers
        for mol, is_training in mols:
            assert mol is not None
            assert mol.GetNumConformers() > 0
            assert is_training is True
            
        # Verify SMILES are valid
        for smiles in smiles_list:
            assert isinstance(smiles, str)
            assert len(smiles) > 0
            mol_from_smiles = Chem.MolFromSmiles(smiles)
            assert mol_from_smiles is not None
    
    def test_load_prediction_data(self, data_loader, ace_test_file):
        """Test loading prediction data from SDF file."""
        smiles_list, mols, activities = data_loader.load_sdf_data(
            str(ace_test_file), is_training=False
        )
        
        # Basic validation
        assert len(smiles_list) > 0
        assert len(mols) == len(smiles_list)
        assert activities is None
        
        # Verify molecules have conformers
        for mol, is_training in mols:
            assert mol is not None
            assert mol.GetNumConformers() > 0
            assert is_training is False
    
    def test_activity_property_missing_for_training(self, data_loader, ace_train_file):
        """Test that loading training data without activity property raises error."""
        with pytest.raises(ValueError, match="Activity property must be specified"):
            data_loader.load_sdf_data(str(ace_train_file), None, is_training=True)
    
    def test_specific_molecule_counts(self, data_loader, ace_train_file, ace_test_file):
        """Test that we get expected molecule counts for specific datasets."""
        # Test training set
        train_smiles, train_mols, train_activities = data_loader.load_sdf_data(
            str(ace_train_file), "Activity", is_training=True
        )
        
        # Test test set
        test_smiles, test_mols, test_activities = data_loader.load_sdf_data(
            str(ace_test_file), is_training=False
        )
        
        # ACE dataset should have reasonable numbers of molecules
        assert 50 <= len(train_smiles) <= 200  # Reasonable range
        assert 20 <= len(test_smiles) <= 100   # Reasonable range
        
        # Activities should be in reasonable range for pIC50/pKi values
        assert all(0 <= act <= 15 for act in train_activities)
    
    def test_hydrogen_addition(self, data_loader, ace_train_file):
        """Test that hydrogens are properly added to molecules."""
        _, mols, _ = data_loader.load_sdf_data(
            str(ace_train_file), "Activity", is_training=True
        )
        
        # Check that molecules have hydrogens
        for mol, _ in mols[:5]:  # Check first 5 molecules
            h_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1)
            assert h_count > 0, "Molecule should have hydrogens"
    
    def test_multiple_datasets(self, data_loader, test_data_dir):
        """Test loading different datasets to ensure consistency."""
        datasets = ["ACE", "AChE", "CCR5"]
        
        for dataset in datasets:
            train_file = test_data_dir / f"{dataset}_train.sdf"
            if train_file.exists():
                smiles_list, mols, activities = data_loader.load_sdf_data(
                    str(train_file), "Activity", is_training=True
                )
                
                assert len(smiles_list) > 0
                assert len(mols) == len(smiles_list)
                assert len(activities) == len(smiles_list)
                
                # Verify consistency
                for i, (mol, is_training) in enumerate(mols):
                    assert mol is not None
                    assert mol.GetNumConformers() > 0
                    assert is_training is True


class TestDataLoaderRegression:
    """Regression tests to prevent drift in data loading behavior."""
    
    def test_ace_training_exact_counts(self, data_loader, ace_train_file):
        """Test exact molecular counts for ACE training set (regression test)."""
        smiles_list, mols, activities = data_loader.load_sdf_data(
            str(ace_train_file), "Activity", is_training=True
        )
        
        # These are the expected exact counts based on the current data
        # Update these if the test data changes intentionally
        expected_train_count = 76  # Based on actual ACE_train.sdf
        
        assert len(smiles_list) == expected_train_count
        assert len(mols) == expected_train_count
        assert len(activities) == expected_train_count
    
    def test_ace_activity_range(self, data_loader, ace_train_file):
        """Test that ACE activities are in expected range (regression test).""" 
        _, _, activities = data_loader.load_sdf_data(
            str(ace_train_file), "Activity", is_training=True
        )
        
        # Check activity statistics
        activities_array = np.array(activities)
        assert 2.0 <= np.min(activities_array) <= 3.0  # Adjusted to match actual ACE data
        assert 8.0 <= np.max(activities_array) <= 12.0  # Reasonable pIC50 range
        assert 6.0 <= np.mean(activities_array) <= 9.0  # Reasonable mean
    
    def test_smiles_consistency(self, data_loader, ace_train_file):
        """Test that SMILES generation is consistent (regression test)."""
        smiles_list1, _, _ = data_loader.load_sdf_data(
            str(ace_train_file), "Activity", is_training=True
        )
        
        # Load again to test consistency
        data_loader2 = DataLoader()
        smiles_list2, _, _ = data_loader2.load_sdf_data(
            str(ace_train_file), "Activity", is_training=True
        )
        
        assert smiles_list1 == smiles_list2, "SMILES should be consistent across loads"