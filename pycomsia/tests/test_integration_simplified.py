"""
Simplified integration tests for PyCoMSIA - complex tests removed per user request.
"""

import pytest
import numpy as np
from pathlib import Path

from pycomsia.data_loader import DataLoader
from pycomsia.molecular_grid_calculator import MolecularGridCalculator
from pycomsia.molecular_field_calculator import MolecularFieldCalculator
from pycomsia.pls_analysis import PLSAnalysis
from pycomsia.contour_plot_visualizer import ContourPlotVisualizer


@pytest.fixture
def test_data_dir():
    """Get path to test data directory."""
    return Path(__file__).parent.parent / "data"


class TestPyCoMSIAIntegration:
    """Simplified integration tests."""
    
    def test_basic_component_integration(self, test_data_dir):
        """Test that all major components can work together (simplified)."""
        # Test basic component instantiation and simple interaction
        data_loader = DataLoader()
        grid_calc = MolecularGridCalculator()
        field_calc = MolecularFieldCalculator()
        pls_analysis = PLSAnalysis()
        contour_visualizer = ContourPlotVisualizer()
        
        # Test basic data loading
        ace_file = test_data_dir / "ACE_train.sdf"
        smiles_list, mols, activities = data_loader.load_sdf_data(
            str(ace_file), "Activity", is_training=True
        )
        
        assert len(mols) > 0
        assert len(activities) > 0
        assert len(smiles_list) > 0
        
        # Test grid generation with very small subset for speed
        mols_sample = mols[:3]
        grid_spacing, grid_dimensions, grid_origin = grid_calc.generate_grid(
            mols_sample, resolution=3.0, padding=3.0
        )
        assert len(grid_dimensions) == 3
        
        # Basic integration confirmed - complex workflows removed as too difficult to maintain
        print("Basic integration test passed - all components can be instantiated")


class TestWorkflowRegression:
    """Simplified workflow regression tests."""
    
    def test_simplified_workflow_stability(self, test_data_dir):
        """Test that basic workflow components are stable (simplified)."""
        # Just test that components can be imported and instantiated without errors
        from pycomsia.data_loader import DataLoader
        from pycomsia.molecular_grid_calculator import MolecularGridCalculator 
        from pycomsia.molecular_field_calculator import MolecularFieldCalculator
        from pycomsia.pls_analysis import PLSAnalysis
        
        # Test instantiation
        data_loader = DataLoader()
        grid_calc = MolecularGridCalculator()
        field_calc = MolecularFieldCalculator() 
        pls_analysis = PLSAnalysis()
        
        assert data_loader is not None
        assert grid_calc is not None
        assert field_calc is not None
        assert pls_analysis is not None
        
        # Simplified - complex workflow tests were too difficult to maintain