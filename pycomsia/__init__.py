"""PyCoMSIA: Comparative Molecular Similarity Indices Analysis."""

# Core modules
from .data_loader import DataLoader
# from .molecule_aligner import MoleculeAligner  # Temporarily disabled due to conflicts
from .molecular_grid_calculator import MolecularGridCalculator
from .molecular_field_calculator import MolecularFieldCalculator
from .molecular_visualizer import MolecularVisualizer
from .pls_analysis import PLSAnalysis
from .pls_analysis_test_sets import PLSAnalysisTestSets
from .contour_plot_visualizer import ContourPlotVisualizer

# Legacy interface
from .pycomsia import *

from ._version import __version__

# Export all classes
__all__ = [
    'DataLoader',
    'MoleculeAligner', 
    'MolecularGridCalculator',
    'MolecularFieldCalculator',
    'MolecularVisualizer', 
    'PLSAnalysis',
    'PLSAnalysisTestSets',
    'ContourPlotVisualizer'
]
