import logging

import numpy as np
import pyvista as pv
from rdkit import Chem

logger = logging.getLogger(__name__)


class MolecularVisualizer:
    def __init__(self):
        self.plotter = None

    def visualize_aligned_molecules(self, aligned_molecules, outputdir):
        """
        Visualize aligned molecules using PyVista.

        Parameters:
        - aligned_molecules: List of RDKit molecules with 3D coordinates
        - filename: Output filename (default: "aligned_molecules.png")
        """
        self.plotter = pv.Plotter(off_screen=True, border_width=2)
        for mol in aligned_molecules:

            self._add_molecule_to_plot(mol)

        self.plotter.camera_position = "iso"
        filename = f"{outputdir}/Alignments/aligned_molecules.png"
        self.plotter.screenshot(filename, scale=5)
        logger.info(f"Image saved to {filename}")
        self.plotter.close()

    def visualize_field(
        self, mol, grid_spacing, grid_shape, grid_origin, field_values, outputdir
    ):
        """
        Visualize a 3D field and molecule using PyVista volume rendering.

        Parameters:
        - mol: RDKit molecule with 3D coordinates
        - grid_spacing, grid_shape, grid_origin: 1D numpy arrays of grid parameters
        - field_values: 1D or 3D numpy array of field values

        """
        for field_name, field_values in field_values.items():
            if field_values.ndim == 1:
                field_values = field_values.reshape(grid_shape)

            # Normalize hydrophobicity field if applicable
            # if field_name == 'hydrophobic':
            #     field_values = self._custom_normalize_field(field_values)

            # Set up the grid and plotter
            self.plotter = pv.Plotter(off_screen=True, border_width=0)
            pv.global_theme.allow_empty_mesh = True
            grid = pv.ImageData()
            grid.dimensions = np.array(grid_shape)
            grid.origin = grid_origin
            grid.spacing = grid_spacing
            grid.point_data["values"] = field_values.flatten(order="F")

            # Title mapping
            field_mapping = {
                "electrostatic": "Electro. Force",
                "steric": "Steric Similarity",
                "hydrophobic": "Hydrophobicity",
                "hbond_acceptor": "Acceptor",
                "hbond_donor": "Donor",
            }

            # Configure scalar bar arguments
            sargs = dict(
                title=field_mapping.get(field_name, field_name),
                title_font_size=20,
                vertical=True,
                position_x=0.04,
                position_y=0.25,
                label_font_size=16,
                font_family="arial",
                fmt="%.4f",
            )

            # Add volume visualization
            opacity, cmap = self._get_visualization_params(field_name, field_values)
            if np.any(field_values):
                self.plotter.add_volume(
                    grid, cmap=cmap, opacity=opacity, scalar_bar_args=sargs
                )
            else:
                self.plotter.add_volume(grid, cmap=cmap, opacity=[0])
                self.plotter.remove_scalar_bar()

            # Add molecule and finishing touches
            self._add_molecule_to_plot(mol)
            self._add_finishing_touches(field_name)

            # Save and close
            filename = f"{outputdir}/Field_Plots/{field_name}.png"
            self.plotter.screenshot(filename, scale=5)
            logger.info(f"Image saved to {filename}")
            self.plotter.close()

    def _add_finishing_touches(self, title=None):
        """Add final visual elements to the plot."""
        self.plotter.camera_position = "iso"
        # Title mapping
        field_mapping = {
            "electrostatic": "Electrostatic Field",
            "steric": "Steric Field",
            "hydrophobic": "Hydrophobicity Field",
            "hbond_acceptor": "Hydrogen Acceptor Field",
            "hbond_donor": "Hydrogen Donor Field",
        }

        if title:
            self.plotter.add_bounding_box(
                line_width=1,
                color="lightgray",
                opacity=0.1,
                outline=False,
                culling="front",
            )
            self.plotter.add_text(
                field_mapping.get(title, title),
                position="upper_edge",
                font_size=100,
                color="black",
                font="arial",
            )

    def _add_molecule_to_plot(self, mol):
        """Add molecule representation to the plot."""
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
            self.plotter.add_mesh(atoms, color=color)

        # Add bonds
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            start = conformer.GetAtomPosition(begin_idx)
            end = conformer.GetAtomPosition(end_idx)

            color = self._get_bond_color(bond.GetBondTypeAsDouble())
            line = pv.Tube(pointa=start, pointb=end, radius=0.1, n_sides=15)
            self.plotter.add_mesh(line, color=color, smooth_shading=True)

    def _get_visualization_params(self, field_name, field_values=None):
        """Get visualization parameters based on field type and data distribution."""

        if field_values is not None:
            if field_name == "electrostatic":
                # For electrostatic fields:
                # 1. Find the absolute max value to establish scale
                max_abs = max(abs(np.min(field_values)), abs(np.max(field_values)))

                # 2. Create thresholds based on percentage of max value
                thresholds = [
                    -max_abs,
                    -0.6 * max_abs,
                    -0.2 * max_abs,
                    0.2 * max_abs,
                    0.6 * max_abs,
                    max_abs,
                ]

                # 3. Calculate what percentage of points fall in each range
                data_density = np.array(
                    [
                        np.mean(field_values <= thresholds[1]),
                        np.mean(
                            (field_values > thresholds[1])
                            & (field_values <= thresholds[2])
                        ),
                        np.mean(
                            (field_values > thresholds[2])
                            & (field_values <= thresholds[3])
                        ),
                        np.mean(
                            (field_values > thresholds[3])
                            & (field_values <= thresholds[4])
                        ),
                        np.mean(field_values > thresholds[4]),
                    ]
                )

                # 4. Adjust opacity based on data density
                base_opacity = [0.5, 0.3, 0.0, 0.0, 0.3, 0.5]
                density_factor = 1.0 / (
                    1.0 + np.exp(5 * (data_density - 0.3))
                )  # Sigmoid function
                opacity = [
                    o * min(1.0, f)
                    for o, f in zip(
                        base_opacity,
                        [
                            density_factor[0],
                            density_factor[1],
                            density_factor[2],
                            density_factor[2],
                            density_factor[3],
                            density_factor[4],
                        ],
                    )
                ]
                return opacity, "coolwarm"

            elif field_name == "hydrophobic":
                # For electrostatic fields:
                # 1. Find the absolute max value to establish scale
                max_abs = max(abs(np.min(field_values)), abs(np.max(field_values)))

                # 2. Create thresholds based on percentage of max value
                thresholds = [
                    -max_abs,
                    -0.6 * max_abs,
                    -0.2 * max_abs,
                    0.2 * max_abs,
                    0.6 * max_abs,
                    max_abs,
                ]

                # 3. Calculate what percentage of points fall in each range
                data_density = np.array(
                    [
                        np.mean(field_values <= thresholds[1]),
                        np.mean(
                            (field_values > thresholds[1])
                            & (field_values <= thresholds[2])
                        ),
                        np.mean(
                            (field_values > thresholds[2])
                            & (field_values <= thresholds[3])
                        ),
                        np.mean(
                            (field_values > thresholds[3])
                            & (field_values <= thresholds[4])
                        ),
                        np.mean(field_values > thresholds[4]),
                    ]
                )

                # 4. Adjust opacity based on data density
                base_opacity = [0.5, 0.3, 0.0, 0.0, 0.3, 0.5]
                density_factor = 1.0 / (
                    1.0 + np.exp(5 * (data_density - 0.3))
                )  # Sigmoid function
                opacity = [
                    o * min(1.0, f)
                    for o, f in zip(
                        base_opacity,
                        [
                            density_factor[0],
                            density_factor[1],
                            density_factor[2],
                            density_factor[2],
                            density_factor[3],
                            density_factor[4],
                        ],
                    )
                ]
                return opacity, "viridis"

        # Default parameters if no field_values provided
        params = {
            "Electrostatic": (
                {"opacity": [0.5, 0.3, 0.0, 0.0, 0.3, 0.5], "cmap": "coolwarm"}
            ),
            "Hydrophobicity": (
                {"opacity": [0.6, 0.3, 0.0, 0.3, 0.6], "cmap": "viridis"}
            ),
            "Steric Similarity": (
                {"opacity": [0.6, 0.5, 0.4, 0.3, 0.2, 0], "cmap": "hot"}
            ),
            "Acceptor": ({"opacity": [0, 0.1, 0.3, 0.4, 0.5, 0.6], "cmap": "inferno"}),
            "Donor": ({"opacity": [0, 0.1, 0.3, 0.4, 0.5, 0.6], "cmap": "cividis"}),
        }

        field_mapping = {
            "electrostatic": "Electrostatic",
            "steric": "Steric Similarity",
            "hydrophobic": "Hydrophobicity",
            "hbond_acceptor": "Acceptor",
            "hbond_donor": "Donor",
        }

        mapped_field_name = field_mapping.get(field_name, field_name)
        return (
            params.get(
                mapped_field_name,
                {"opacity": [0, 0.1, 0.3, 0.4, 0.5, 0.6], "cmap": "viridis"},
            )["opacity"],
            params.get(
                mapped_field_name,
                {"opacity": [0, 0.1, 0.3, 0.4, 0.5, 0.6], "cmap": "viridis"},
            )["cmap"],
        )

    def _get_atom_color(self, atomic_num):
        """Return color for specific atom type."""
        colors = {
            1: "white",  # Hydrogen
            6: "silver",  # Carbon
            7: "lightblue",  # Nitrogen
            8: "red",  # Oxygen
        }
        return colors.get(atomic_num, "lightgray")

    def _get_bond_color(self, bond_type):
        """Return color for specific bond type."""
        colors = {
            1.0: "silver",  # Single bond
            1.5: "lightblue",  # Aromatic
            2.0: "lightgray",  # Double bond
            3.0: "red",  # Triple bond
        }
        return colors.get(bond_type, "silver")

    def _custom_normalize_field(self, field_values, new_min=0, new_max=1):
        """Normalize field values to specified range."""
        original_min = np.min(field_values)
        original_max = np.max(field_values)
        normalized_values = (field_values - original_min) / (
            original_max - original_min
        ) * (new_max - new_min) + new_min
        return normalized_values
