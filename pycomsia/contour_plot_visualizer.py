import numpy as np
import pyvista as pv
from rdkit import Chem
from rdkit.Chem import AllChem
import os
from pathlib import Path

# Try to import PyMOL, handle gracefully if not available
try:
    import pymol2
    PYMOL_AVAILABLE = True
except ImportError:
    PYMOL_AVAILABLE = False

class ContourPlotVisualizer:
    def __init__(self):
        self.significant_ranges = None
        
    def calculate_significant_ranges(self, reconstructed_coeffs, top_percent=0.5, bottom_percent=0.5):
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
            
            significant_ranges[field_name] = {
                'low': (np.min(low_range), np.max(low_range)),
                'high': (np.min(high_range), np.max(high_range))
            }
        
        self.significant_ranges = significant_ranges
        return significant_ranges
    
    def visualize_contour_plots(self, mol, reconstructed_coeffs, grid_dimensions, 
                                grid_origin, grid_spacing, output, significant_ranges=None):
        """Visualize molecular fields using marching cubes algorithm with field-specific colors and legends"""
        
        color_map = {
            'steric_field': {
                'low': 'green',
                'high': 'yellow'
            },
            'electrostatic_field': {
                'low': 'blue', 
                'high': 'red'
            },
            'hydrophobic_field': {
                'low': 'purple', 
                'high': 'orange'
            },
            'hbond_acceptor_field': {
                'low': 'purple', 
                'high': 'teal'
            },
            'hbond_donor_field': {
                'low': 'grey', 
                'high': 'yellow'
            }
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
            low_range = significant_ranges[field_name]['low']
            high_range = significant_ranges[field_name]['high']
            
            # Get colors for the specific field
            low_color = color_map.get(field_name, {}).get('low', 'blue')
            high_color = color_map.get(field_name, {}).get('high', 'red')
            
            # Create isosurfaces for low values
            contours_low = grid.contour(
                [low_range[0], low_range[1]],
                scalars=field_name,
                method='marching_cubes'
            )
            contours_low = contours_low.interpolate(grid)
            plotter.add_mesh(contours_low, color=low_color, opacity=0.7, smooth_shading=True)
            
            # Create isosurfaces for high values
            contours_high = grid.contour(
                [high_range[0], high_range[1]],
                scalars=field_name,
                method='marching_cubes'
            )
            contours_high = contours_high.interpolate(grid)
           
            plotter.add_mesh(contours_high, color=high_color, opacity=0.7, smooth_shading=True)
            
            # Add molecule visualization
            self._add_molecule_to_plot(plotter, mol)
            
            # Set visualization properties
            plotter.camera_position = 'iso'
            plotter.camera.zoom(1.2)
            plotter.background_color = 'white'
            
            
            filename = f"{output}/Contour_Plots/{field_name}_contourplot.png"
            plotter.screenshot(filename, scale=5)
            print(f"Saved contour plot for {field_name} field to {filename}")
            plotter.close()
    
    def create_pymol_session(self, molecules, output_dir, session_name="pycomsia_ligands",
                           coefficients=None, grid_dimensions=None, grid_origin=None, 
                           grid_spacing=None, significant_ranges=None, protein_file=None):
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
            print("⚠️  PyMOL not available. Install pymol-open-source to create PyMOL sessions.")
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
                    if not mol.HasProp('_Name'):
                        mol.SetProp('_Name', f'Ligand_{i+1:03d}')
                    writer.write(mol)
            writer.close()
            
            print(f"📄 Created temporary SDF with {len(molecules)} molecules: {temp_sdf_file}")
            
            # Initialize PyMOL instance
            with pymol2.PyMOL() as pymol:
                cmd = pymol.cmd
                
                # Load molecules from SDF file
                cmd.load(str(temp_sdf_file), "ligands")
                
                # Load optional protein structure
                if protein_file is not None:
                    protein_path = Path(protein_file)
                    if not protein_path.exists():
                        print(f"⚠️  Protein file not found, skipping: {protein_file}")
                    else:
                        cmd.load(str(protein_path), "protein")
                        cmd.hide("everything", "protein")
                        cmd.show("cartoon", "protein")
                        cmd.color("grey80", "protein")
                        print(f"✅ Protein loaded: {protein_path.name}")
                
                # Remove non-polar hydrogens (keep only polar hydrogens bonded to N, O, S)
                cmd.remove("elem H and not (neighbor elem N+O+S)")
                
                # Set all_states to 1
                cmd.set("all_states", 1)
                
                # Set display preferences
                cmd.hide("everything")
                cmd.show("sticks", "ligands")
                cmd.hide("lines", "ligands")  # Hide wires/wireframe
                cmd.color("atomic", "ligands")
                
                # Add contours for all available fields if data is provided
                if (coefficients is not None and grid_dimensions is not None and significant_ranges is not None):
                    self._add_all_field_contours_to_pymol(
                        cmd, coefficients, 
                        grid_dimensions, grid_origin, grid_spacing,
                        significant_ranges
                    )
                
                # Center and zoom on all molecules
                cmd.center("ligands")
                cmd.zoom("ligands", buffer=5.0)
                
                # Set background to white for better visibility
                cmd.bg_color("white")
                
                # Save PyMOL session
                cmd.save(str(session_file))
                
                print(f"✅ PyMOL session saved: {session_file}")
                print(f"   📊 Contains {len(molecules)} ligands")
                if coefficients is not None:
                    field_count = len([f for f in coefficients.keys() if f in significant_ranges])
                    print(f"   🎨 Field contours included: {field_count} field types")
                print(f"   💡 Open with: pymol {session_file}")
        
        except Exception as e:
            print(f"❌ Error creating PyMOL session: {e}")
        
        finally:
            # Clean up temporary file
            if temp_sdf_file.exists():
                temp_sdf_file.unlink()
                print(f"🗑️  Cleaned up temporary file: {temp_sdf_file}")
    
    def _add_all_field_contours_to_pymol(self, cmd, coefficients, 
                                        grid_dimensions, grid_origin, grid_spacing, 
                                        significant_ranges):
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
            'steric_field': {'low': 'green', 'high': 'yellow'},
            'electrostatic_field': {'low': 'blue', 'high': 'red'},
            'hydrophobic_field': {'low': 'purple', 'high': 'orange'},
            'hbond_acceptor_field': {'low': 'purple', 'high': 'teal'},
            'hbond_donor_field': {'low': 'grey', 'high': 'yellow'}
        }
        
        for field_name, field_data in coefficients.items():
            if field_name in significant_ranges:
                try:
                    print(f"   🎨 Adding {field_name} contours...")
                    
                    # Get colors for this field
                    colors = field_colors.get(field_name, {'low': 'cyan', 'high': 'magenta'})
                    
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
                        low_range = significant_ranges[field_name]['low']
                        contours_low = grid.contour(
                            [low_range[0], low_range[1]],
                            scalars=field_name,
                            method='marching_cubes'
                        )
                        
                        # Generate contours for high values
                        high_range = significant_ranges[field_name]['high'] 
                        contours_high = grid.contour(
                            [high_range[0], high_range[1]],
                            scalars=field_name,
                            method='marching_cubes'
                        )
                        
                        # Add contours as surfaces
                        field_short = field_name.replace('_field', '')
                        if contours_low.n_points > 0:
                            self._add_contour_surfaces_to_pymol(
                                cmd, contours_low, f"{field_short}_negative", colors['low']
                            )
                        
                        if contours_high.n_points > 0:
                            self._add_contour_surfaces_to_pymol(
                                cmd, contours_high, f"{field_short}_positive", colors['high']
                            )
                        
                        print(f"     ✅ {field_short}: {contours_low.n_points} low, {contours_high.n_points} high points")
                        
                    except ImportError:
                        print(f"     ⚠️  PyVista not available for {field_name} contour generation")
                    
                except Exception as e:
                    print(f"     ⚠️  Could not add {field_name} contours: {e}")
    
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
                cmd.pseudoatom(f"{object_name}_temp", 
                             pos=[float(point[0]), float(point[1]), float(point[2])],
                             state=1)
            
            # Show as surface and set properties
            cmd.show("surface", f"{object_name}_temp")
            cmd.color(color, f"{object_name}_temp")
            cmd.set("transparency", 0.5, f"{object_name}_temp")
            
            # Rename the object to final name
            cmd.set_name(f"{object_name}_temp", object_name)
            
            print(f"       🔵 Added {len(points)} {color} surface points as '{object_name}'")
    
    def _add_contour_spheres_to_pymol(self, cmd, contour_mesh, object_name, color, radius=0.1):
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
                cmd.pseudoatom(f"{object_name}_temp", 
                             pos=[float(point[0]), float(point[1]), float(point[2])],
                             state=1)
            
            # Show as spheres and set properties
            cmd.show("spheres", f"{object_name}_temp")
            cmd.set("sphere_scale", radius, f"{object_name}_temp")
            cmd.color(color, f"{object_name}_temp")
            cmd.set("sphere_transparency", 0.3, f"{object_name}_temp")
            
            # Rename the object to final name
            cmd.set_name(f"{object_name}_temp", object_name)
            
            print(f"   🔵 Added {len(points)} {color} spheres as '{object_name}'")
    
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
            1: 'white',     # Hydrogen
            6: 'silver',    # Carbon
            7: 'lightblue', # Nitrogen
            8: 'red',       # Oxygen
        }
        return colors.get(atomic_num, 'lightgray')
    
    def _get_bond_color(self, bond):
        """Get color for specific bond type"""
        bond_types = {
            1.0: 'silver',     # Single bond
            1.5: 'silver',  # Aromatic
            2.0: 'silver',  # Double bond
            3.0: 'silver'         # Triple bond
        }
        return bond_types.get(bond.GetBondTypeAsDouble(), 'silver')