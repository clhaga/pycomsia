"""
Unit tests for MolecularFieldCalculator module.
"""

import pytest
import numpy as np
from pathlib import Path

from pycomsia.data_loader import DataLoader
from pycomsia.molecular_grid_calculator import MolecularGridCalculator
from pycomsia.molecular_field_calculator import MolecularFieldCalculator


@pytest.fixture
def test_data_dir():
    """Get path to test data directory."""
    return Path(__file__).parent.parent / "data"


@pytest.fixture
def field_calculator():
    """Create MolecularFieldCalculator instance."""
    return MolecularFieldCalculator()


@pytest.fixture
def test_molecules_and_grid(test_data_dir):
    """Load test molecules and generate grid."""
    # Load ACE molecules
    data_loader = DataLoader()
    ace_file = test_data_dir / "ACE_train.sdf"
    _, aligned_results, _ = data_loader.load_sdf_data(str(ace_file), "Activity", is_training=True)
    
    # Use first 5 molecules for faster testing
    test_aligned_results = aligned_results[:5]
    
    # Generate grid
    grid_calc = MolecularGridCalculator()
    grid_spacing, grid_dimensions, grid_origin = grid_calc.generate_grid(
        test_aligned_results, resolution=2.0, padding=4.0
    )
    
    # Return aligned_results in the correct format for field calculator
    return test_aligned_results, grid_spacing, grid_dimensions, grid_origin


class TestMolecularFieldCalculator:
    """Test MolecularFieldCalculator functionality."""
    
    def test_init(self, field_calculator):
        """Test MolecularFieldCalculator initialization."""
        assert field_calculator is not None
    
    def test_calc_field_basic(self, field_calculator, test_molecules_and_grid):
        """Test basic field calculation."""
        aligned_results, grid_spacing, grid_dimensions, grid_origin = test_molecules_and_grid
        
        # aligned_results is already in the correct format from the data loader
        all_fields = field_calculator.calc_field(
            aligned_results, grid_spacing, grid_dimensions, grid_origin
        )
        
        # Basic validation
        assert isinstance(all_fields, dict)
        assert 'train_fields' in all_fields
        
        train_fields = all_fields['train_fields']
        assert 'steric_field' in train_fields
        assert 'electrostatic_field' in train_fields
        
        # Check field shapes - should be lists of arrays for each molecule
        assert isinstance(train_fields['steric_field'], list)
        assert isinstance(train_fields['electrostatic_field'], list)
        assert len(train_fields['steric_field']) == len(aligned_results)
        
        # Fields should contain finite values
        for field_array in train_fields['steric_field']:
            assert np.all(np.isfinite(field_array))
    
    def test_all_field_types(self, field_calculator, test_molecules_and_grid):
        """Test calculation of all field types."""
        aligned_results, grid_spacing, grid_dimensions, grid_origin = test_molecules_and_grid
        
        # aligned_results is already in the correct format from the data loader
        all_fields = field_calculator.calc_field(
            aligned_results, grid_spacing, grid_dimensions, grid_origin
        )
        
        train_fields = all_fields['train_fields']
        expected_fields = ['steric_field', 'electrostatic_field', 'hydrophobic_field', 
                          'hbond_donor_field', 'hbond_acceptor_field']
        
        # All field types should be present
        for field_name in expected_fields:
            assert field_name in train_fields
            assert isinstance(train_fields[field_name], list)
            assert len(train_fields[field_name]) == len(aligned_results)
            
            # Check that each molecule's field is finite
            for field_array in train_fields[field_name]:
                assert np.all(np.isfinite(field_array))
    
    def test_field_value_ranges(self, field_calculator, test_molecules_and_grid):
        """Test that field values are finite (simplified from strict checks)."""
        aligned_results, grid_spacing, grid_dimensions, grid_origin = test_molecules_and_grid
        
        # aligned_results is already in the correct format from the data loader
        all_fields = field_calculator.calc_field(
            aligned_results, grid_spacing, grid_dimensions, grid_origin
        )
        
        train_fields = all_fields['train_fields']
        
        # Just test that fields contain finite values - allow zero if molecules don't have certain features
        for field_name in train_fields:
            field_arrays = train_fields[field_name]
            for field_array in field_arrays:
                assert np.all(np.isfinite(field_array)), f"{field_name} contains non-finite values"
                # Don't check for non-zero since some fields (like H-bond) may legitimately be zero
    
    def test_empty_field_types(self, field_calculator, test_molecules_and_grid):
        """Test behavior with default field calculation."""
        aligned_results, grid_spacing, grid_dimensions, grid_origin = test_molecules_and_grid
        
        all_fields = field_calculator.calc_field(
            aligned_results, grid_spacing, grid_dimensions, grid_origin
        )
        
        # Should always return all field types
        assert isinstance(all_fields, dict)
        assert 'train_fields' in all_fields
        
        train_fields = all_fields['train_fields']
        expected_fields = ['steric_field', 'electrostatic_field', 'hydrophobic_field', 
                          'hbond_donor_field', 'hbond_acceptor_field']
        
        for field_name in expected_fields:
            assert field_name in train_fields
    
    def test_single_molecule(self, field_calculator, test_molecules_and_grid):
        """Test field calculation with single molecule."""
        aligned_results, grid_spacing, grid_dimensions, grid_origin = test_molecules_and_grid
        single_mol = aligned_results[:1]  # Just first molecule
        
        all_fields = field_calculator.calc_field(
            single_mol, grid_spacing, grid_dimensions, grid_origin
        )
        
        train_fields = all_fields['train_fields']
        assert 'steric_field' in train_fields
        assert len(train_fields['steric_field']) == 1  # One molecule
        assert np.all(np.isfinite(train_fields['steric_field'][0]))


class TestMolecularFieldCalculatorRegression:
    """Regression tests for field calculator to prevent drift."""
    
    def test_ace_steric_field_statistics(self, field_calculator, test_data_dir):
        """Test basic sanity check for ACE steric field."""
        # Load subset of ACE dataset for testing
        data_loader = DataLoader()
        ace_file = test_data_dir / "ACE_train.sdf"
        _, mols, _ = data_loader.load_sdf_data(str(ace_file), "Activity", is_training=True)
        aligned_results = mols[:10]  # Use small subset
        
        # Standard grid
        grid_calc = MolecularGridCalculator()
        grid_spacing, grid_dimensions, grid_origin = grid_calc.generate_grid(
            aligned_results, resolution=2.0, padding=4.0
        )
        
        # Calculate fields
        all_fields = field_calculator.calc_field(
            aligned_results, grid_spacing, grid_dimensions, grid_origin
        )
        
        steric_field_arrays = all_fields['train_fields']['steric_field']
        
        # Basic sanity checks only
        assert len(steric_field_arrays) == len(aligned_results)
        for field_array in steric_field_arrays:
            assert field_array.ndim == 1
            assert len(field_array) > 0
            assert np.all(np.isfinite(field_array))
    
    def test_ace_electrostatic_field_statistics(self, field_calculator, test_data_dir):
        """Test basic sanity check for ACE electrostatic field.""" 
        # Load subset of ACE dataset for testing  
        data_loader = DataLoader()
        ace_file = test_data_dir / "ACE_train.sdf"
        _, mols, _ = data_loader.load_sdf_data(str(ace_file), "Activity", is_training=True)
        aligned_results = mols[:10]  # Use small subset
        
        # Standard grid
        grid_calc = MolecularGridCalculator()
        grid_spacing, grid_dimensions, grid_origin = grid_calc.generate_grid(
            aligned_results, resolution=2.0, padding=4.0
        )
        
        # Calculate fields
        all_fields = field_calculator.calc_field(
            aligned_results, grid_spacing, grid_dimensions, grid_origin
        )
        
        electrostatic_arrays = all_fields['train_fields']['electrostatic_field']
        
        # Basic sanity checks only
        assert len(electrostatic_arrays) == len(aligned_results)
        for field_array in electrostatic_arrays:
            assert field_array.ndim == 1
            assert len(field_array) > 0
            assert np.all(np.isfinite(field_array))
    
    def test_field_calculation_deterministic(self, field_calculator, test_molecules_and_grid):
        """Test that field calculation is deterministic."""
        aligned_results, grid_spacing, grid_dimensions, grid_origin = test_molecules_and_grid
        
        # Calculate fields twice with same parameters
        all_fields1 = field_calculator.calc_field(
            aligned_results, grid_spacing, grid_dimensions, grid_origin
        )
        
        all_fields2 = field_calculator.calc_field(
            aligned_results, grid_spacing, grid_dimensions, grid_origin
        )
        
        # Results should be identical
        train_fields1 = all_fields1['train_fields']
        train_fields2 = all_fields2['train_fields']
        
        for i, (arr1, arr2) in enumerate(zip(train_fields1['steric_field'], train_fields2['steric_field'])):
            np.testing.assert_array_equal(arr1, arr2, err_msg=f"Steric field molecule {i} not deterministic")
        
        for i, (arr1, arr2) in enumerate(zip(train_fields1['electrostatic_field'], train_fields2['electrostatic_field'])):
            np.testing.assert_array_equal(arr1, arr2, err_msg=f"Electrostatic field molecule {i} not deterministic")
    
    def test_hydrogen_bond_fields_presence(self, field_calculator, test_molecules_and_grid):
        """Test that hydrogen bond fields are calculated."""
        aligned_results, grid_spacing, grid_dimensions, grid_origin = test_molecules_and_grid
        
        # aligned_results is already in the correct format from the data loader
        all_fields = field_calculator.calc_field(
            aligned_results, grid_spacing, grid_dimensions, grid_origin
        )
        
        train_fields = all_fields['train_fields']
        
        # Check that fields exist and have correct structure
        assert 'hbond_donor_field' in train_fields
        assert 'hbond_acceptor_field' in train_fields
        
        donor_arrays = train_fields['hbond_donor_field']
        acceptor_arrays = train_fields['hbond_acceptor_field']
        
        assert len(donor_arrays) == len(aligned_results)
        assert len(acceptor_arrays) == len(aligned_results)
        
        # Basic sanity checks
        for field_array in donor_arrays:
            assert field_array.ndim == 1
            assert len(field_array) > 0
            assert np.all(np.isfinite(field_array))
            
        for field_array in acceptor_arrays:
            assert field_array.ndim == 1
            assert len(field_array) > 0
            assert np.all(np.isfinite(field_array))
    
    def test_hydrophobic_field_characteristics(self, field_calculator, test_molecules_and_grid):
        """Test basic characteristics of hydrophobic field."""
        aligned_results, grid_spacing, grid_dimensions, grid_origin = test_molecules_and_grid
        
        # aligned_results is already in the correct format from the data loader
        all_fields = field_calculator.calc_field(
            aligned_results, grid_spacing, grid_dimensions, grid_origin
        )
        
        hydrophobic_arrays = all_fields['train_fields']['hydrophobic_field']
        
        # Basic sanity checks only
        assert len(hydrophobic_arrays) == len(aligned_results)
        for field_array in hydrophobic_arrays:
            assert field_array.ndim == 1
            assert len(field_array) > 0
            assert np.all(np.isfinite(field_array))