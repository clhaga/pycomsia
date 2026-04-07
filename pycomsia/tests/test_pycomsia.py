"""
Unit and regression test for the pycomsia package.
"""

# Import package, test suite, and other packages as needed
import sys
import pytest
import numpy as np
from pathlib import Path

import pycomsia
from pycomsia import DataLoader, MolecularGridCalculator, MolecularFieldCalculator, PLSAnalysis


def test_pycomsia_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "pycomsia" in sys.modules


class TestPackageStructure:
    """Test package structure and basic functionality."""
    
    def test_core_modules_importable(self):
        """Test that core modules can be imported."""
        # These should all be importable
        from pycomsia.data_loader import DataLoader
        from pycomsia.molecular_grid_calculator import MolecularGridCalculator
        from pycomsia.molecular_field_calculator import MolecularFieldCalculator
        from pycomsia.pls_analysis import PLSAnalysis
        from pycomsia.contour_plot_visualizer import ContourPlotVisualizer
        
        # Should be able to instantiate
        assert DataLoader() is not None
        assert MolecularGridCalculator() is not None
        assert MolecularFieldCalculator() is not None
        assert PLSAnalysis() is not None
        assert ContourPlotVisualizer() is not None
    
    def test_data_directory_exists(self):
        """Test that data directory exists with expected files."""
        data_dir = Path(__file__).parent.parent / "data"
        assert data_dir.exists(), "Data directory should exist"
        assert data_dir.is_dir(), "Data path should be a directory"
        
        # Check for key datasets
        expected_files = ["ACE_train.sdf", "ACE_test.sdf", "README.md"]
        for filename in expected_files:
            file_path = data_dir / filename
            assert file_path.exists(), f"Expected data file {filename} should exist"
    
    def test_package_version_info(self):
        """Test package version and metadata."""
        # Package should have basic attributes
        assert hasattr(pycomsia, '__name__')
        
        # Check that main classes are accessible from package
        assert hasattr(pycomsia, 'DataLoader')
        assert hasattr(pycomsia, 'MolecularGridCalculator')
        assert hasattr(pycomsia, 'MolecularFieldCalculator')
        assert hasattr(pycomsia, 'PLSAnalysis')


class TestBasicFunctionality:
    """Test basic functionality across modules."""
    
    @pytest.fixture
    def test_data_dir(self):
        """Get path to test data directory."""
        return Path(__file__).parent.parent / "data"
    
    def test_data_loading_basic(self, test_data_dir):
        """Test basic data loading functionality."""
        data_loader = DataLoader()
        ace_file = test_data_dir / "ACE_train.sdf"
        
        smiles_list, mols, activities = data_loader.load_sdf_data(
            str(ace_file), "Activity", is_training=True
        )
        
        # Basic checks
        assert len(smiles_list) > 10, "Should load reasonable number of molecules"
        assert len(mols) == len(smiles_list)
        assert len(activities) == len(smiles_list)
        assert all(isinstance(activity, (int, float)) for activity in activities)
    
    def test_grid_calculation_basic(self, test_data_dir):
        """Test basic grid calculation."""
        # Load small dataset
        data_loader = DataLoader()
        ace_file = test_data_dir / "ACE_train.sdf"
        _, mols, _ = data_loader.load_sdf_data(str(ace_file), "Activity", is_training=True)
        test_mols = mols[:5]  # Small subset
        
        # Calculate grid
        grid_calc = MolecularGridCalculator()
        grid_spacing, grid_dimensions, grid_origin = grid_calc.generate_grid(
            test_mols, resolution=2.0, padding=4.0
        )
        
        # Basic validation
        assert len(grid_spacing) == 3
        assert len(grid_dimensions) == 3
        assert len(grid_origin) == 3
        assert all(dim > 0 for dim in grid_dimensions)
        assert np.allclose(grid_spacing, [2.0, 2.0, 2.0])
    
    def test_field_calculation_basic(self, test_data_dir):
        """Test basic field calculation."""
        # Load and prepare data
        data_loader = DataLoader()
        ace_file = test_data_dir / "ACE_train.sdf"
        _, mols, _ = data_loader.load_sdf_data(str(ace_file), "Activity", is_training=True)
        test_aligned_results = mols[:3]  # Very small subset for speed
        
        # Grid - pass aligned_results since generate_grid expects this format
        grid_calc = MolecularGridCalculator()
        grid_spacing, grid_dimensions, grid_origin = grid_calc.generate_grid(
            test_aligned_results, resolution=3.0, padding=4.0
        )
        
        # Fields
        field_calc = MolecularFieldCalculator()
        all_fields = field_calc.calc_field(
            test_aligned_results, grid_spacing, grid_dimensions, grid_origin
        )
        
        # Validation
        assert 'train_fields' in all_fields
        train_fields = all_fields['train_fields']
        assert 'steric_field' in train_fields
        assert len(train_fields['steric_field']) == len(test_aligned_results)
        assert np.all(np.isfinite(train_fields['steric_field'][0]))
    
    def test_pls_analysis_basic(self, test_data_dir):
        """Test basic PLS analysis functionality."""
        # Load and prepare data
        data_loader = DataLoader()
        ace_file = test_data_dir / "ACE_train.sdf"
        _, mols, activities = data_loader.load_sdf_data(str(ace_file), "Activity", is_training=True)
        test_aligned_results = mols[:10]  # Small subset
        test_activities = activities[:10]
        
        # Grid and fields - pass aligned_results since generate_grid expects this format
        grid_calc = MolecularGridCalculator()
        grid_spacing, grid_dimensions, grid_origin = grid_calc.generate_grid(
            test_aligned_results, resolution=3.0, padding=4.0
        )
        
        field_calc = MolecularFieldCalculator()
        all_fields = field_calc.calc_field(
            test_aligned_results, grid_spacing, grid_dimensions, grid_origin
        )
        
        # Extract training fields for PLS analysis
        fields = all_fields['train_fields']
        
        # PLS analysis
        pls_analysis = PLSAnalysis()
        pls_analysis.convert_fields_to_X(fields)
        pls_analysis.perform_loo_analysis(test_activities, max_components=3)
        pls_analysis.fit_final_model(test_activities)
        
        # Basic validation
        assert pls_analysis.pls_model is not None
        assert pls_analysis.optimal_n_components > 0
        assert pls_analysis.r2_train is not None
        assert pls_analysis.r2_test is not None
        
        # Should be able to get coefficients
        coefficients = pls_analysis.get_coefficient_fields()
        assert 'steric_field' in coefficients
        assert 'electrostatic_field' in coefficients


class TestRegressionBaseline:
    """Baseline regression tests to catch major changes."""
    
    @pytest.fixture
    def test_data_dir(self):
        """Get path to test data directory."""
        return Path(__file__).parent.parent / "data"
    
    def test_ace_dataset_basic_stats(self, test_data_dir):
        """Test that ACE dataset has expected basic statistics."""
        data_loader = DataLoader()
        ace_file = test_data_dir / "ACE_train.sdf"
        _, mols, activities = data_loader.load_sdf_data(
            str(ace_file), "Activity", is_training=True
        )
        
        # Dataset size should be stable
        expected_molecule_count = 76  # Update if dataset changes intentionally
        assert len(mols) == expected_molecule_count, \
            f"ACE molecule count changed: expected {expected_molecule_count}, got {len(mols)}"
        
        # Activity statistics should be in expected ranges
        activities_array = np.array(activities)
        assert 2.0 <= np.min(activities_array) <= 3.0  # Adjusted to match actual data
        assert 9.5 <= np.max(activities_array) <= 10.5  # Adjusted to match actual data  
        assert 5.5 <= np.mean(activities_array) <= 7.5  # Adjusted to match actual data
    
    def test_standard_grid_dimensions(self, test_data_dir):
        """Test that standard grid dimensions are stable."""
        data_loader = DataLoader()
        ace_file = test_data_dir / "ACE_train.sdf"
        _, mols, _ = data_loader.load_sdf_data(str(ace_file), "Activity", is_training=True)
        
        # Standard grid calculation
        grid_calc = MolecularGridCalculator()
        _, grid_dimensions, _ = grid_calc.generate_grid(
            mols, resolution=2.0, padding=4.0
        )
        
        # These dimensions should be stable for ACE dataset
        expected_dims = (12, 13, 11)
        assert tuple(grid_dimensions) == expected_dims, \
            f"Grid dimensions changed: expected {expected_dims}, got {tuple(grid_dimensions)}"
