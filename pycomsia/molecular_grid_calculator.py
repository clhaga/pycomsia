import logging

import numpy as np
from rdkit import Chem

logger = logging.getLogger(__name__)


class MolecularGridCalculator:
    def __init__(self):
        """Initialize MolecularGridCalculator"""
        pass

    def _snap_coords_to_grid(self, coords, grid_spacing):
        """
        Snap coordinates to the nearest grid point.
        Args:
            coords: Numpy array of shape (n_atoms, 3) containing atomic coordinates
            grid_spacing: Grid spacing (float)
        Returns:
            snapped_coords: Numpy array of shape (n_atoms, 3) with snapped coordinates
        """
        return np.round(coords / grid_spacing) * grid_spacing

    def generate_grid(self, aligned_results, resolution=1.0, padding=3):
        """
        Generate grid parameters that encompass all aligned molecules.
        Args:
            aligned_results: List of tuples (mol, is_training) from pre-aligned SDF files
            resolution: Grid spacing (float)
            padding: Padding around the molecules (float)
        Returns:
            grid_spacing: Tuple of (x, y, z) spacing
            grid_dimensions: Tuple of (nx, ny, nz) points
            grid_origin: Tuple of (x, y, z) origin coordinates
        """
        # Collect coordinates from all aligned molecules
        coords = []
        for mol, _ in aligned_results:
            if mol is not None:  # Check for valid molecules
                conf = mol.GetConformer()
                for i in range(mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(i)
                    coords.append([pos.x, pos.y, pos.z])

        if not coords:
            raise ValueError("No valid molecules found in aligned_results")

        coords = np.array(coords)
        n_mols = sum(1 for _, _ in aligned_results)
        logger.debug(f"Grid input: {n_mols} molecules, {len(coords)} atomic coords")
        logger.debug(
            f"Coordinate range before snap: "
            f"x=[{coords[:,0].min():.2f}, {coords[:,0].max():.2f}]  "
            f"y=[{coords[:,1].min():.2f}, {coords[:,1].max():.2f}]  "
            f"z=[{coords[:,2].min():.2f}, {coords[:,2].max():.2f}]"
        )

        # Snap coordinates to the grid
        snapped_coords = self._snap_coords_to_grid(coords, resolution)

        # Find the extremes of snapped coordinates
        min_coords = np.min(snapped_coords, axis=0) - padding
        max_coords = np.max(snapped_coords, axis=0) + padding

        # Calculate grid parameters
        grid_spacing = (resolution, resolution, resolution)

        # Calculate number of points in each dimension
        nx = int(np.ceil((max_coords[0] - min_coords[0]) / resolution))
        ny = int(np.ceil((max_coords[1] - min_coords[1]) / resolution))
        nz = int(np.ceil((max_coords[2] - min_coords[2]) / resolution))

        grid_dimensions = (nx, ny, nz)
        grid_origin = tuple(min_coords)

        logger.debug(
            f"Grid origin: ({grid_origin[0]:.2f}, {grid_origin[1]:.2f}, {grid_origin[2]:.2f})"
        )
        logger.debug(f"Grid dimensions: {nx} x {ny} x {nz} = {nx*ny*nz} points")
        logger.debug(f"Grid spacing: {resolution} Å  padding: {padding} Å")

        return grid_spacing, grid_dimensions, grid_origin
