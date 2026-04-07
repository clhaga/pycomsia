"""PyCoMSIA main interface module."""

# Import all core classes
from .data_loader import DataLoader
# from .molecule_aligner import MoleculeAligner  # Temporarily disabled due to conflicts
from .molecular_grid_calculator import MolecularGridCalculator
from .molecular_field_calculator import MolecularFieldCalculator
from .molecular_visualizer import MolecularVisualizer
from .pls_analysis import PLSAnalysis
from .pls_analysis_test_sets import PLSAnalysisTestSets
from .contour_plot_visualizer import ContourPlotVisualizer


def get_version():
    """Get the version of PyCoMSIA."""
    try:
        from ._version import __version__
        return __version__
    except ImportError:
        return "unknown"


def list_modules():
    """List available PyCoMSIA modules."""
    modules = [
        'DataLoader - Load molecular data from SDF files',
        'MoleculeAligner - Alignment functionality disabled (assumes pre-aligned SDFs)', 
        'MolecularGridCalculator - Generate 3D grids for molecular fields',
        'MolecularFieldCalculator - Calculate steric, electrostatic, and other fields',
        'MolecularVisualizer - Visualize molecules and fields using PyVista',
        'PLSAnalysis - Partial Least Squares analysis with train/test split',
        'PLSAnalysisTestSets - PLS analysis for separate training and test sets',
        'ContourPlotVisualizer - Generate contour plots of coefficient fields'
    ]
    return modules


def canvas(with_attribution=True):
    """
    Legacy placeholder function.
    
    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from.

    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution.
    """
    quote = "PyCoMSIA: The code is but a canvas to our molecular imagination."
    if with_attribution:
        quote += "\n\t- Adapted from Henry David Thoreau"
    return quote


if __name__ == "__main__":
    print(canvas())
    print("\nAvailable modules:")
    for module in list_modules():
        print(f"  - {module}")
