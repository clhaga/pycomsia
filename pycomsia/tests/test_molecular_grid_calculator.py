"""
Unit tests for MolecularGridCalculator module.
"""

import pytest
import numpy as np
from pathlib import Path

from pycomsia.data_loader import DataLoader
from pycomsia.molecular_grid_calculator import MolecularGridCalculator


@pytest.fixture
def test_data_dir():
    """Get path to test data directory."""
    return Path(__file__).parent.parent / "data"


@pytest.fixture
def grid_calculator():
    """Create MolecularGridCalculator instance."""
    return MolecularGridCalculator()


@pytest.fixture
def ace_molecules(test_data_dir):
    """Load ACE training molecules for testing."""
    data_loader = DataLoader()
    ace_file = test_data_dir / "ACE_train.sdf"
    _, mols, _ = data_loader.load_sdf_data(str(ace_file), "Activity", is_training=True)
    return mols[:10]  # Use first 10 molecules for faster testing


class TestMolecularGridCalculator:
    """Test MolecularGridCalculator functionality."""
    
    def test_init(self, grid_calculator):
        """Test MolecularGridCalculator initialization."""
        assert grid_calculator is not None
    
    def test_generate_grid_basic(self, grid_calculator, ace_molecules):
        """Test basic grid generation."""
        grid_spacing, grid_dimensions, grid_origin = grid_calculator.generate_grid(
            ace_molecules, resolution=2.0, padding=4.0
        )
        
        # Basic validation
        assert isinstance(grid_spacing, (tuple, list, np.ndarray))
        assert isinstance(grid_dimensions, (tuple, list, np.ndarray))
        assert isinstance(grid_origin, (tuple, list, np.ndarray))
        
        assert len(grid_spacing) == 3
        assert len(grid_dimensions) == 3
        assert len(grid_origin) == 3
        
        # All dimensions should be positive
        assert all(dim > 0 for dim in grid_dimensions)
        assert all(spacing > 0 for spacing in grid_spacing)
    
    def test_grid_spacing_matches_resolution(self, grid_calculator, ace_molecules):
        """Test that grid spacing matches requested resolution."""
        resolution = 1.5
        grid_spacing, _, _ = grid_calculator.generate_grid(
            ace_molecules, resolution=resolution, padding=4.0
        )
        
        # Grid spacing should be close to resolution
        np.testing.assert_allclose(grid_spacing, [resolution, resolution, resolution])
    
    def test_padding_effect(self, grid_calculator, ace_molecules):
        """Test that padding affects grid dimensions."""
        # Generate grids with different padding
        _, dims_small, _ = grid_calculator.generate_grid(
            ace_molecules, resolution=2.0, padding=2.0
        )
        _, dims_large, _ = grid_calculator.generate_grid(
            ace_molecules, resolution=2.0, padding=6.0
        )
        
        # Larger padding should result in larger dimensions
        assert all(dims_large[i] >= dims_small[i] for i in range(3))
    
    def test_resolution_effect(self, grid_calculator, ace_molecules):
        """Test that resolution affects grid dimensions."""
        # Generate grids with different resolutions
        _, dims_coarse, _ = grid_calculator.generate_grid(
            ace_molecules, resolution=3.0, padding=4.0
        )
        _, dims_fine, _ = grid_calculator.generate_grid(
            ace_molecules, resolution=1.0, padding=4.0
        )
        
        # Finer resolution should result in more grid points
        assert all(dims_fine[i] >= dims_coarse[i] for i in range(3))


class TestMolecularGridCalculatorRegression:
    """Regression tests for grid calculator to prevent drift."""
    
    def test_ace_grid_dimensions_standard(self, grid_calculator, test_data_dir):
        """Test exact grid dimensions for ACE dataset with standard parameters."""
        data_loader = DataLoader()
        ace_file = test_data_dir / "ACE_train.sdf"
        _, mols, _ = data_loader.load_sdf_data(str(ace_file), "Activity", is_training=True)
        
        grid_spacing, grid_dimensions, grid_origin = grid_calculator.generate_grid(
            mols, resolution=2.0, padding=4.0
        )
        
        # Expected values for ACE dataset with these parameters
        # These should be stable unless the algorithm or data changes
        expected_dimensions = (12, 13, 11)  # Based on current ACE dataset
        
        assert tuple(grid_dimensions) == expected_dimensions, \
            f"Expected {expected_dimensions}, got {tuple(grid_dimensions)}"
        
        # Grid spacing should be exactly 2.0
        np.testing.assert_allclose(grid_spacing, [2.0, 2.0, 2.0], rtol=1e-10)
    
    def test_ace_grid_origin_range(self, grid_calculator, test_data_dir):
        """Test that grid origin is in expected range for ACE dataset."""
        data_loader = DataLoader()
        ace_file = test_data_dir / "ACE_train.sdf"
        _, mols, _ = data_loader.load_sdf_data(str(ace_file), "Activity", is_training=True)
        
        _, _, grid_origin = grid_calculator.generate_grid(
            mols, resolution=2.0, padding=4.0
        )
        
        # Grid origin should be reasonable for molecular coordinates
        assert -20 <= grid_origin[0] <= 20
        assert -20 <= grid_origin[1] <= 20  
        assert -20 <= grid_origin[2] <= 20
    
    def test_multiple_datasets_consistency(self, grid_calculator, test_data_dir):
        """Test grid generation consistency across different datasets."""
        data_loader = DataLoader()
        datasets = ["ACE", "AChE", "CCR5"]
        
        for dataset in datasets:
            train_file = test_data_dir / f"{dataset}_train.sdf" 
            if train_file.exists():
                _, mols, _ = data_loader.load_sdf_data(
                    str(train_file), "Activity", is_training=True
                )
                
                grid_spacing, grid_dimensions, grid_origin = grid_calculator.generate_grid(
                    mols[:5], resolution=2.0, padding=4.0  # Use subset for speed
                )
                
                # Basic consistency checks
                assert len(grid_spacing) == 3
                assert len(grid_dimensions) == 3
                assert len(grid_origin) == 3
                assert all(dim > 0 for dim in grid_dimensions)
                np.testing.assert_allclose(grid_spacing, [2.0, 2.0, 2.0])
    
    def test_grid_volume_calculation(self, grid_calculator, ace_molecules):
        """Test that grid volume is reasonable and consistent."""
        grid_spacing, grid_dimensions, _ = grid_calculator.generate_grid(
            ace_molecules, resolution=2.0, padding=4.0
        )
        
        # Calculate grid volume
        volume = np.prod(grid_dimensions) * np.prod(grid_spacing)
        
        # Volume should be reasonable for molecular systems
        assert 1000 <= volume <= 50000  # Cubic angstroms, reasonable range
        
        # Test with different resolution
        grid_spacing2, grid_dimensions2, _ = grid_calculator.generate_grid(
            ace_molecules, resolution=1.0, padding=4.0
        )
        volume2 = np.prod(grid_dimensions2) * np.prod(grid_spacing2)
        
        # Volume should be similar (same molecular ensemble)
        assert abs(volume - volume2) / volume < 0.5  # Within 50%
    
    def test_deterministic_behavior(self, grid_calculator, ace_molecules):
        """Test that grid generation is deterministic."""
        # Generate grid twice with same parameters
        result1 = grid_calculator.generate_grid(
            ace_molecules, resolution=2.0, padding=4.0
        )
        result2 = grid_calculator.generate_grid(
            ace_molecules, resolution=2.0, padding=4.0
        )
        
        # Results should be identical
        np.testing.assert_array_equal(result1[0], result2[0])  # spacing
        np.testing.assert_array_equal(result1[1], result2[1])  # dimensions
        np.testing.assert_allclose(result1[2], result2[2])     # origin