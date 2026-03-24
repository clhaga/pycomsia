import logging

import numpy as np
import pyvista as pv
from rdkit import Chem
from rdkit.Chem import AllChem
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import PyMOL, handle gracefully if not available
try:
    import pymol2

    PYMOL_AVAILABLE = True
except ImportError:
    PYMOL_AVAILABLE = False


class ContourPlotVisualizer:
    def __init__(self):
        self.significant_ranges = None

    def calculate_significant_ranges(
        self, reconstructed_coeffs, top_percent=0.5, bottom_percent=0.5
    ):
        """Calculate significant value ranges for each field"""
        significant_ranges = {}

        for field_name, field_values in reconstructed_coeffs.items():
            flat_values = field_values.flatten()
            sorted_values = np.sort(flat_values)

            n = len(sorted_values)
            bottom_index = int(n * bottom_percent / 100)
            top_index = int(n * (1 - top_percent / 100))

            low_range = sorted_values[:bottom_index]
            high_range = sorted_values[top_index:]

            # Handle empty ranges
            if len(low_range) == 0:
                low_range = [np.min(sorted_values)]
            if len(high_range) == 0:
                high_range = [np.max(sorted_values)]

            lo = (np.min(low_range), np.max(low_range))
            hi = (np.min(high_range), np.max(high_range))
            significant_ranges[field_name] = {"low": lo, "high": hi}
            logger.debug(f"  {field_name}: low=[{lo[0]:.4f}, {lo[1]:.4f}]  "
                         f"high=[{hi[0]:.4f}, {hi[1]:.4f}]  "
                         f"(top/bottom {top_percent}%)")

        self.significant_ranges = significant_ranges
        return significant_ranges

    def visualize_contour_plots(
        self,
        mol,
        reconstructed_coeffs,
        grid_dimensions,
        grid_origin,
        grid_spacing,
        output,
        significant_ranges=None,
    ):
        """Visualize molecular fields using marching cubes algorithm with field-specific colors and legends"""

        color_map = {
            "steric_field": {"low": "green", "high": "yellow"},
            "electrostatic_field": {"low": "blue", "high": "red"},
            "hydrophobic_field": {"low": "purple", "high": "orange"},
            "hbond_acceptor_field": {"low": "purple", "high": "teal"},
            "hbond_donor_field": {"low": "grey", "high": "yellow"},
        }

        if significant_ranges is None:
            significant_ranges = self.significant_ranges

        for field_name, field_values in reconstructed_coeffs.items():
            if field_values.ndim == 1:
                field_values = field_values.reshape(grid_dimensions)

            grid = pv.ImageData()
            grid.dimensions = np.array(grid_dimensions)
            grid.origin = grid_origin
            grid.spacing = grid_spacing
            grid.point_data[field_name] = field_values.flatten(order="F")

            plotter = pv.Plotter(off_screen=True)
            pv.global_theme.allow_empty_mesh = True
            low_range = significant_ranges[field_name]["low"]
            high_range = significant_ranges[field_name]["high"]

            # Get colors for the specific field
            low_color = color_map.get(field_name, {}).get("low", "blue")
            high_color = color_map.get(field_name, {}).get("high", "red")

            # Create isosurfaces for low values
            contours_low = grid.contour(
                [low_range[0], low_range[1]],
                scalars=field_name,
                method="marching_cubes",
            )
            contours_low = contours_low.interpolate(grid)
            plotter.add_mesh(
                contours_low, color=low_color, opacity=0.7, smooth_shading=True
            )

            # Create isosurfaces for high values
            contours_high = grid.contour(
                [high_range[0], high_range[1]],
                scalars=field_name,
                method="marching_cubes",
            )
            contours_high = contours_high.interpolate(grid)

            plotter.add_mesh(
                contours_high, color=high_color, opacity=0.7, smooth_shading=True
            )

            # Add molecule visualization
            self._add_molecule_to_plot(plotter, mol)

            # Set visualization properties
            plotter.camera_position = "iso"
            plotter.camera.zoom(1.2)
            plotter.background_color = "white"

            filename = f"{output}/Contour_Plots/{field_name}_contourplot.png"
            plotter.screenshot(filename, scale=5)
            logger.info(f"Saved contour plot for {field_name} field to {filename}")
            plotter.close()

    def create_pymol_session(
        self,
        molecules,
        output_dir,
        session_name="pycomsia_ligands",
        coefficients=None,
        grid_dimensions=None,
        grid_origin=None,
        grid_spacing=None,
        significant_ranges=None,
        protein_file=None,
    ):
        """
        Create a PyMOL session file (.pse) containing all input ligands and contour plots.

        Args:
            molecules: List of RDKit molecule objects
            output_dir: Output directory path
            session_name: Base name for the session file (without extension)
            coefficients: Field coefficients for contour generation
            grid_dimensions: Grid dimensions tuple
            grid_origin: Grid origin coordinates
            grid_spacing: Grid spacing values
            significant_ranges: Significant value ranges for contours
            protein_file: Optional path to a protein structure file (e.g. PDB) to include
                          in the session. When provided the protein is loaded as a separate
                          object and displayed as cartoon.
        """
        if not PYMOL_AVAILABLE:
            logger.warning(
                "PyMOL not available. Install pymol-open-source to create PyMOL sessions."
            )
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Create ContourPlots subdirectory if it doesn't exist
        contour_plots_dir = output_dir / "Contour_Plots"
        contour_plots_dir.mkdir(exist_ok=True)

        # Create temporary SDF file with all molecules
        temp_sdf_file = output_dir / f"{session_name}_temp.sdf"
        session_file = contour_plots_dir / f"{session_name}.pse"

        try:
            # Write all molecules to temporary SDF file
            writer = Chem.SDWriter(str(temp_sdf_file))
            for i, mol in enumerate(molecules):
                if mol is not None:
                    # Add molecule name if not present
                    if not mol.HasProp("_Name"):
                        mol.SetProp("_Name", f"Ligand_{i+1:03d}")
                    writer.write(mol)
            writer.close()

            logger.info(
                f"Created temporary SDF with {len(molecules)} molecules: {temp_sdf_file}"
            )

            # Initialize PyMOL instance
            with pymol2.PyMOL() as pymol:
                cmd = pymol.cmd

                # Load molecules from SDF file
                cmd.load(str(temp_sdf_file), "ligands")

                # Load optional protein structure
                protein_loaded = False
                protein_object_name = None
                if protein_file is not None:
                    protein_path = Path(protein_file)
                    if not protein_path.exists():
                        logger.warning(f"Protein file not found, skipping: {protein_file}")
                    else:
                        protein_object_name = protein_path.stem
                        cmd.load(str(protein_path), protein_object_name)
                        cmd.hide("everything", protein_object_name)
                        cmd.show("cartoon", protein_object_name)
                        cmd.color("grey80", protein_object_name)
                        protein_loaded = True
                        logger.info(
                            f"Protein loaded: {protein_path.name} (object: {protein_object_name})"
                        )

                # Remove non-polar hydrogens (keep only polar hydrogens bonded to N, O, S)
                cmd.remove("elem H and not (neighbor elem N+O+S)")

                # Set all_states to 1
                cmd.set("all_states", 1)

                # Set display preferences
                cmd.hide("everything", "ligands")
                cmd.show("sticks", "ligands")
                cmd.hide("lines", "ligands")  # Hide wires/wireframe
                cmd.color("atomic", "ligands")

                # Restore protein display (the broad hide above would have cleared it)
                if protein_loaded:
                    cmd.hide("everything", protein_object_name)
                    cmd.show("cartoon", protein_object_name)
                    cmd.color("grey80", protein_object_name)

                # Add contours for all available fields if data is provided
                if (
                    coefficients is not None
                    and grid_dimensions is not None
                    and significant_ranges is not None
                ):
                    self._add_all_field_contours_to_pymol(
                        cmd,
                        coefficients,
                        grid_dimensions,
                        grid_origin,
                        grid_spacing,
                        significant_ranges,
                    )

                # Compute and add binding site volume when a protein is available
                if protein_loaded:
                    logger.info("Computing binding site volume...")
                    # Use 'polymer' to select only protein chains — excludes any co-crystallised
                    # ligands, waters, and other HETATM records present in the PDB file.
                    protein_coords_raw = cmd.get_coords(
                        f"{protein_object_name} and polymer and not elem H", 1
                    )
                    if protein_coords_raw is not None:
                        binding_site_points = self._compute_binding_site_volume(
                            molecules, np.array(protein_coords_raw)
                        )
                        if len(binding_site_points) > 0:
                            self._add_binding_site_to_pymol(
                                cmd, binding_site_points, "binding_site_volume", "cyan"
                            )
                            logger.info(
                                f"Binding site volume: {len(binding_site_points)} grid points"
                            )

                            # Add per-field excluded volume objects when field data is available
                            if (
                                coefficients is not None
                                and grid_dimensions is not None
                                and significant_ranges is not None
                            ):
                                logger.info("Computing per-field excluded volumes...")
                                self._add_excluded_field_volumes_to_pymol(
                                    cmd,
                                    binding_site_points,
                                    coefficients,
                                    grid_dimensions,
                                    grid_origin,
                                    grid_spacing,
                                    significant_ranges,
                                )
                        else:
                            logger.warning(
                                "No binding site volume points found (check protein proximity to ligands)"
                            )
                    else:
                        logger.warning("Could not retrieve protein coordinates")

                # Center and zoom on all molecules
                cmd.center("ligands")
                cmd.zoom("ligands", buffer=5.0)

                # Set background to white for better visibility
                cmd.bg_color("white")

                # Save PyMOL session
                cmd.save(str(session_file))

                logger.info(f"PyMOL session saved: {session_file}")
                logger.info(f"Contains {len(molecules)} ligands")
                if coefficients is not None:
                    field_count = len(
                        [f for f in coefficients.keys() if f in significant_ranges]
                    )
                    logger.info(f"Field contours included: {field_count} field types")
                logger.info(f"Open with: pymol {session_file}")

        except Exception as e:
            logger.error(f"Error creating PyMOL session: {e}")

        finally:
            # Clean up temporary file
            if temp_sdf_file.exists():
                temp_sdf_file.unlink()
                logger.debug(f"Cleaned up temporary file: {temp_sdf_file}")

    def _add_all_field_contours_to_pymol(
        self,
        cmd,
        coefficients,
        grid_dimensions,
        grid_origin,
        grid_spacing,
        significant_ranges,
    ):
        """
        Add contours for all available molecular fields to PyMOL session.

        Args:
            cmd: PyMOL command object
            coefficients: Dictionary of field coefficients
            grid_dimensions: Grid dimensions tuple
            grid_origin: Grid origin coordinates
            grid_spacing: Grid spacing values
            significant_ranges: Significant value ranges for all fields
        """
        # Color mapping for different field types
        field_colors = {
            "steric_field": {"low": "green", "high": "yellow"},
            "electrostatic_field": {"low": "blue", "high": "red"},
            "hydrophobic_field": {"low": "purple", "high": "orange"},
            "hbond_acceptor_field": {"low": "purple", "high": "teal"},
            "hbond_donor_field": {"low": "grey", "high": "yellow"},
        }

        for field_name, field_data in coefficients.items():
            if field_name in significant_ranges:
                try:
                    logger.info(f"Adding {field_name} contours...")

                    # Get colors for this field
                    colors = field_colors.get(
                        field_name, {"low": "cyan", "high": "magenta"}
                    )

                    # Reshape field data if needed
                    if field_data.ndim == 1:
                        field_values = field_data.reshape(grid_dimensions)
                    else:
                        field_values = field_data

                    # Create grid for contour generation
                    try:
                        import pyvista as pv

                        grid = pv.ImageData()
                        grid.dimensions = np.array(grid_dimensions)
                        grid.origin = grid_origin
                        grid.spacing = grid_spacing
                        grid.point_data[field_name] = field_values.flatten(order="F")

                        # Generate contours for low values
                        low_range = significant_ranges[field_name]["low"]
                        contours_low = grid.contour(
                            [low_range[0], low_range[1]],
                            scalars=field_name,
                            method="marching_cubes",
                        )

                        # Generate contours for high values
                        high_range = significant_ranges[field_name]["high"]
                        contours_high = grid.contour(
                            [high_range[0], high_range[1]],
                            scalars=field_name,
                            method="marching_cubes",
                        )

                        # Add contours as surfaces
                        field_short = field_name.replace("_field", "")
                        if contours_low.n_points > 0:
                            self._add_contour_surfaces_to_pymol(
                                cmd,
                                contours_low,
                                f"{field_short}_negative",
                                colors["low"],
                            )

                        if contours_high.n_points > 0:
                            self._add_contour_surfaces_to_pymol(
                                cmd,
                                contours_high,
                                f"{field_short}_positive",
                                colors["high"],
                            )

                        logger.info(
                            f"{field_short}: {contours_low.n_points} low, {contours_high.n_points} high points"
                        )

                    except ImportError:
                        logger.warning(
                            f"PyVista not available for {field_name} contour generation"
                        )

                except Exception as e:
                    logger.warning(f"Could not add {field_name} contours: {e}")

    def _add_contour_surfaces_to_pymol(self, cmd, contour_mesh, object_name, color):
        """
        Add contour surface points as pseudoatoms displayed as surfaces in PyMOL.

        Args:
            cmd: PyMOL command object
            contour_mesh: PyVista mesh containing contour surface
            object_name: Name for the PyMOL object
            color: Color for the surfaces
        """
        points = contour_mesh.points

        # Subsample points if there are too many (for performance)
        max_points = 2000
        if len(points) > max_points:
            step = len(points) // max_points
            points = points[::step]

        # Create pseudoatoms at each contour point
        if len(points) > 0:
            # Create a new object for the contour points
            for i, point in enumerate(points):
                # Create pseudoatom at each point
                cmd.pseudoatom(
                    f"{object_name}_temp",
                    pos=[float(point[0]), float(point[1]), float(point[2])],
                    state=1,
                )

            # Show as surface and set properties
            cmd.show("surface", f"{object_name}_temp")
            cmd.color(color, f"{object_name}_temp")
            cmd.set("transparency", 0.5, f"{object_name}_temp")

            # Rename the object to final name
            cmd.set_name(f"{object_name}_temp", object_name)

            logger.debug(
                f"Added {len(points)} {color} surface points as '{object_name}'"
            )

    def _add_contour_spheres_to_pymol(
        self, cmd, contour_mesh, object_name, color, radius=0.1
    ):
        """
        Add contour surface points as small spheres in PyMOL.

        Args:
            cmd: PyMOL command object
            contour_mesh: PyVista mesh containing contour surface
            object_name: Name for the PyMOL object
            color: Color for the spheres
            radius: Radius of spheres
        """
        points = contour_mesh.points

        # Subsample points if there are too many (for performance)
        max_points = 2000
        if len(points) > max_points:
            step = len(points) // max_points
            points = points[::step]

        # Create pseudoatoms at each contour point
        if len(points) > 0:
            # Create a new object for the contour points
            for i, point in enumerate(points):
                # Create pseudoatom at each point
                cmd.pseudoatom(
                    f"{object_name}_temp",
                    pos=[float(point[0]), float(point[1]), float(point[2])],
                    state=1,
                )

            # Show as spheres and set properties
            cmd.show("spheres", f"{object_name}_temp")
            cmd.set("sphere_scale", radius, f"{object_name}_temp")
            cmd.color(color, f"{object_name}_temp")
            cmd.set("sphere_transparency", 0.3, f"{object_name}_temp")

            # Rename the object to final name
            cmd.set_name(f"{object_name}_temp", object_name)

            logger.debug(f"Added {len(points)} {color} spheres as '{object_name}'")

    def _compute_binding_site_volume(
        self,
        molecules,
        protein_coords,
        padding=3.0,
        grid_resolution=1.0,
        protein_proximity=5.0,
        protein_exclusion_radius=2.8,
    ):
        """
        Compute binding site volume as a point cloud.

        The volume is defined as the set of grid points that:
          1. Fall within the bounding box of all input ligands expanded by
             ``padding`` angstroms.
          2. Are within ``protein_proximity`` angstroms of any protein
             heavy atom (i.e. near the protein surface).
          3. Are further than ``protein_exclusion_radius`` angstroms from
             every protein heavy atom (i.e. not buried inside the protein).

        ``protein_exclusion_radius`` defaults to 2.8 Å, which approximates the
        van der Waals radius of a carbon atom (~1.7 Å) plus a 1.1 Å probe, so
        that densely packed interior regions of the protein are reliably excluded
        even when no single atom centre is closer than 2.0 Å to the grid point.

        Args:
            molecules: List of RDKit molecule objects (all input ligands).
            protein_coords: (N, 3) numpy array of protein heavy-atom coordinates.
            padding: Expansion around ligand bounding box in angstroms (default 5.0).
            grid_resolution: Spacing between grid points in angstroms (default 1.0).
            protein_proximity: Maximum distance to protein to keep a point (default 5.0).
            protein_exclusion_radius: Points closer to the protein than this value
                are considered inside the protein and are excluded (default 2.8).

        Returns:
            (M, 3) numpy array of binding site volume points, or an empty
            array when insufficient data is available.
        """
        # Collect all heavy-atom coordinates across all ligands
        all_ligand_coords = []
        for mol in molecules:
            if mol is None:
                continue
            mol_noh = Chem.RemoveHs(mol)
            try:
                conf = mol_noh.GetConformer()
            except ValueError:
                continue
            for i in range(mol_noh.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                all_ligand_coords.append([pos.x, pos.y, pos.z])

        if not all_ligand_coords:
            return np.empty((0, 3))

        ligand_coords = np.array(all_ligand_coords)

        # Compute the ligand bounding box centre and expand by padding.
        bbox_min = ligand_coords.min(axis=0) - padding
        bbox_max = ligand_coords.max(axis=0) + padding
        bbox_centre = (bbox_min + bbox_max) / 2.0
        half_extent = (bbox_max - bbox_min) / 2.0

        # Build a regular grid spanning the expanded rectangular envelope first,
        # then clip it to an octagonal prism shape by removing corners.
        # Offsets are normalised to [-1, 1] per axis so the clipping is invariant
        # to the box aspect ratio.  The L1 constraint |n_i| + |n_j| <= sqrt(2)
        # in normalised space produces the standard 45-degree diagonal cuts that
        # turn a rectangle into an octagon in each projection plane, giving a
        # truncated-octahedron-like shape in 3D.
        x = np.arange(bbox_min[0], bbox_max[0] + grid_resolution, grid_resolution)
        y = np.arange(bbox_min[1], bbox_max[1] + grid_resolution, grid_resolution)
        z = np.arange(bbox_min[2], bbox_max[2] + grid_resolution, grid_resolution)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        candidate_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        # Normalised offsets from the bounding-box centre (safe: half_extent > 0)
        norm_offsets = (candidate_points - bbox_centre) / half_extent
        diag_limit = np.sqrt(2)

        octagon_mask = (
            (np.abs(norm_offsets[:, 0]) + np.abs(norm_offsets[:, 1]) <= diag_limit)
            & (np.abs(norm_offsets[:, 0]) + np.abs(norm_offsets[:, 2]) <= diag_limit)
            & (np.abs(norm_offsets[:, 1]) + np.abs(norm_offsets[:, 2]) <= diag_limit)
        )
        candidate_points = candidate_points[octagon_mask]

        if len(protein_coords) == 0:
            return candidate_points

        # Compute per-point minimum distance to any protein heavy atom.
        # Processed in chunks to keep memory usage bounded.
        chunk_size = 2000
        min_dists_protein = np.empty(len(candidate_points))
        for start in range(0, len(candidate_points), chunk_size):
            chunk = candidate_points[start : start + chunk_size]
            dists = np.linalg.norm(
                chunk[:, np.newaxis, :] - protein_coords[np.newaxis, :, :], axis=2
            )
            min_dists_protein[start : start + chunk_size] = dists.min(axis=1)

        # Compute per-point minimum distance to any ligand heavy atom.
        # Points within ligand_exclusion_radius are excluded (inside a ligand atom VDW shell).
        # Keeping this radius tight (1.4 Å) ensures the candidate region stays close to the
        # ligands and does not extend to the far side of the protein.
        ligand_exclusion_radius = 1.4
        min_dists_ligand = np.empty(len(candidate_points))
        for start in range(0, len(candidate_points), chunk_size):
            chunk = candidate_points[start : start + chunk_size]
            dists = np.linalg.norm(
                chunk[:, np.newaxis, :] - ligand_coords[np.newaxis, :, :], axis=2
            )
            min_dists_ligand[start : start + chunk_size] = dists.min(axis=1)

        mask = (
            (min_dists_protein <= protein_proximity)
            & (min_dists_protein > protein_exclusion_radius)
            & (min_dists_ligand > ligand_exclusion_radius)
        )
        return candidate_points[mask]

    def _add_binding_site_to_pymol(
        self, cmd, points, object_name="binding_site_volume", color="cyan"
    ):
        """
        Add binding site volume points as a surface object in PyMOL.

        Points are added as pseudoatoms and rendered as a transparent surface,
        consistent with the contour-plot visualisation style.

        Args:
            cmd: PyMOL command object.
            points: (N, 3) numpy array of volume grid points.
            object_name: Name for the PyMOL object (default 'binding_site_volume').
            color: PyMOL colour name (default 'lightcyan').
        """
        if len(points) == 0:
            return

        # Randomly subsample to keep PyMOL performance acceptable
        max_points = 3000
        if len(points) > max_points:
            rng = np.random.default_rng(seed=0)
            indices = rng.choice(len(points), max_points, replace=False)
            points = points[indices]

        tmp_name = f"{object_name}_temp"
        for point in points:
            cmd.pseudoatom(
                tmp_name,
                pos=[float(point[0]), float(point[1]), float(point[2])],
                state=1,
            )

        cmd.show("surface", tmp_name)
        cmd.color(color, tmp_name)
        cmd.set("transparency", 0.7, tmp_name)
        cmd.set_name(tmp_name, object_name)

        logger.debug(
            f"Added {len(points)} {color} surface points as '{object_name}'"
        )

    def _add_excluded_field_volumes_to_pymol(
        self,
        cmd,
        binding_site_points,
        coefficients,
        grid_dimensions,
        grid_origin,
        grid_spacing,
        significant_ranges,
        exploration_radius=2.0,
    ):
        """
        For each field type create a PyMOL object (e.g. 'steric_excluded') that
        shows the portion of the binding site NOT covered by the field's significant
        contours (both positive and negative).

        A binding site point is considered 'explored' if it lies within
        ``exploration_radius`` Å of any vertex of the field's isosurface meshes
        (both low and high contours).  Points further away than this from any
        isosurface are 'excluded' / unexplored.
        """
        # Colours match the existing field contour colour map; use the 'low' colour
        # as the representative shade for the excluded object.
        field_colors = {
            "steric_field": "green",
            "electrostatic_field": "blue",
            "hydrophobic_field": "purple",
            "hbond_acceptor_field": "purple",
            "hbond_donor_field": "grey",
        }

        # DEBUG: collect excluded point sets to compare across fields at the end
        _debug_excluded = {}

        for field_name, field_data in coefficients.items():
            if field_name not in significant_ranges:
                continue

            try:
                field_values = (
                    field_data.reshape(grid_dimensions)
                    if field_data.ndim == 1
                    else field_data
                )

                # Build the same PyVista grid used for the contour objects
                grid = pv.ImageData()
                grid.dimensions = np.array(grid_dimensions)
                grid.origin = grid_origin
                grid.spacing = grid_spacing
                grid.point_data[field_name] = field_values.flatten(order="F")

                low = significant_ranges[field_name]["low"]
                high = significant_ranges[field_name]["high"]

                # Generate the isosurface meshes (same iso-values as the contour objects)
                contour_low = grid.contour(
                    [low[0], low[1]], scalars=field_name, method="marching_cubes"
                )
                contour_high = grid.contour(
                    [high[0], high[1]], scalars=field_name, method="marching_cubes"
                )

                # Collect all isosurface vertices for both low and high contours
                iso_verts = []
                if contour_low.n_points > 0:
                    iso_verts.append(contour_low.points)
                if contour_high.n_points > 0:
                    iso_verts.append(contour_high.points)

                n_iso = sum(len(v) for v in iso_verts)

                # DEBUG
                logger.debug(
                    f"[DEBUG] {field_name}:\n"
                    f"  iso-surface vertices: low={contour_low.n_points}, "
                    f"high={contour_high.n_points}  (total {n_iso})\n"
                    f"  thresholds: low=({low[0]:.4f}, {low[1]:.4f})  "
                    f"high=({high[0]:.4f}, {high[1]:.4f})\n"
                    f"  exploration_radius={exploration_radius} Å"
                )

                if n_iso == 0:
                    # No isosurface generated — entire binding site is unexplored
                    logger.debug(
                        f"no iso-surface vertices for {field_name}; all binding site points are excluded"
                    )
                    excluded_points = binding_site_points
                else:
                    all_iso_verts = np.vstack(iso_verts)  # (M, 3)

                    # For each binding site point find the distance to the nearest
                    # isosurface vertex; mark as explored if within exploration_radius.
                    chunk_size = 1000
                    min_dists_iso = np.empty(len(binding_site_points))
                    for start in range(0, len(binding_site_points), chunk_size):
                        chunk = binding_site_points[start : start + chunk_size]
                        d = np.linalg.norm(
                            chunk[:, np.newaxis, :] - all_iso_verts[np.newaxis, :, :],
                            axis=2,
                        )
                        min_dists_iso[start : start + chunk_size] = d.min(axis=1)

                    explored = min_dists_iso <= exploration_radius
                    excluded_points = binding_site_points[~explored]

                    logger.debug(
                        f"explored (within {exploration_radius}Å of iso): "
                        f"{explored.sum()}/{len(binding_site_points)}  "
                        f"excluded: {(~explored).sum()}"
                    )

                _debug_excluded[field_name] = excluded_points

                field_short = field_name.replace("_field", "")
                object_name = f"{field_short}_excluded"
                color = field_colors.get(field_name, "white")

                if len(excluded_points) > 0:
                    self._add_binding_site_to_pymol(
                        cmd, excluded_points, object_name, color
                    )
                    pct = 100 * len(excluded_points) / len(binding_site_points)
                    logger.info(
                        f"{object_name}: {len(excluded_points)} excluded points "
                        f"({pct:.0f}% of binding site)"
                    )
                else:
                    logger.info(
                        f"{object_name}: no excluded points — "
                        f"field covers entire binding site"
                    )

            except Exception as e:
                logger.warning(
                    f"Could not compute excluded volume for {field_name}: {e}"
                )

        # DEBUG: cross-compare excluded point sets
        if len(_debug_excluded) > 1:
            field_names = list(_debug_excluded.keys())
            logger.debug("[DEBUG] Cross-field excluded-point comparison:")
            for i in range(len(field_names)):
                for j in range(i + 1, len(field_names)):
                    fa, fb = field_names[i], field_names[j]
                    pa, pb = _debug_excluded[fa], _debug_excluded[fb]
                    if pa.shape == pb.shape and np.array_equal(pa, pb):
                        logger.debug(f"  IDENTICAL: {fa} == {fb}  ({len(pa)} pts)")
                    else:
                        logger.debug(
                            f"  different: {fa} ({len(pa)} pts) vs "
                            f"{fb} ({len(pb)} pts)"
                        )

    def _add_molecule_to_plot(self, plotter, mol):
        """Add molecule representation to the plot"""
        mol = Chem.RemoveHs(mol)
        conformer = mol.GetConformer()

        # Add atoms
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            coord = np.array(conformer.GetAtomPosition(idx))
            atomic_num = atom.GetAtomicNum()

            sphere = pv.Sphere(radius=0.2, phi_resolution=20, theta_resolution=20)
            atoms = pv.PolyData(coord).glyph(geom=sphere, scale=False, orient=False)
            color = self._get_atom_color(atomic_num)
            plotter.add_mesh(atoms, color=color)

        # Add bonds
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            start = conformer.GetAtomPosition(begin_idx)
            end = conformer.GetAtomPosition(end_idx)

            color = self._get_bond_color(bond)
            line = pv.Tube(pointa=start, pointb=end, radius=0.1, n_sides=15)
            plotter.add_mesh(line, color=color, smooth_shading=True)

    def _get_atom_color(self, atomic_num):
        """Get color for specific atom type"""
        colors = {
            1: "white",  # Hydrogen
            6: "silver",  # Carbon
            7: "lightblue",  # Nitrogen
            8: "red",  # Oxygen
        }
        return colors.get(atomic_num, "lightgray")

    def _get_bond_color(self, bond):
        """Get color for specific bond type"""
        bond_types = {
            1.0: "silver",  # Single bond
            1.5: "silver",  # Aromatic
            2.0: "silver",  # Double bond
            3.0: "silver",  # Triple bond
        }
        return bond_types.get(bond.GetBondTypeAsDouble(), "silver")
