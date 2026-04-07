# PyCoMSIA Example Datasets

This directory contains molecular datasets in SDF format for demonstrating PyCoMSIA (Comparative Molecular Similarity Indices Analysis) functionality. Each dataset consists of training and test sets with molecular structures and biological activity data.

## Available Datasets

### ACE (Angiotensin-Converting Enzyme)
- **ACE_train.sdf** - Training set for ACE inhibitors
- **ACE_test.sdf** - Test set for ACE inhibitors
- **Target**: Cardiovascular drugs, antihypertensive agents

### AChE (Acetylcholinesterase) 
- **AChE_train.sdf** - Training set for AChE inhibitors
- **AChE_test.sdf** - Test set for AChE inhibitors  
- **Target**: Alzheimer's disease, cholinesterase inhibitors

### ATA (Alpha-1A Adrenergic Receptor)
- **ATA_train.sdf** - Training set for alpha-1A adrenergic receptor ligands
- **ATA_test.sdf** - Test set for alpha-1A adrenergic receptor ligands
- **Target**: Cardiovascular and urological conditions

### CCR5 (C-C Chemokine Receptor 5)
- **CCR5_train.sdf** - Training set for CCR5 antagonists
- **CCR5_test.sdf** - Test set for CCR5 antagonists
- **Target**: HIV entry inhibitors, inflammatory diseases

### STEROIDS
- **STEROIDS_train.sdf** - Training set for steroid compounds
- **STEROIDS_test.sdf** - Test set for steroid compounds
- **Target**: Hormonal activity, steroid receptor modulators

### THERM (Thermolysin)
- **THERM_train.sdf** - Training set for thermolysin inhibitors
- **THERM_test.sdf** - Test set for thermolysin inhibitors
- **Target**: Metalloprotease inhibitors

### THR (Thrombin)
- **THR_train.sdf** - Training set for thrombin inhibitors
- **THR_test.sdf** - Test set for thrombin inhibitors
- **Target**: Anticoagulants, blood clotting disorders

## Data Format

All SDF files contain:
- **3D molecular structures** with atomic coordinates
- **Biological activity data** stored as molecular properties
- **SMILES representations** for each compound
- **Standardized molecular conformations** suitable for CoMSIA analysis

## Usage Examples

```python
import pycomsia

# Load data using DataLoader
data_loader = pycomsia.DataLoader() 

# Load training data from SDF
train_smiles, train_molecules, train_activities = data_loader.load_sdf_data(
    'pycomsia/data/ACE_train.sdf', 
    activity_property='Activity',
    is_training=True
)

# Load test data from SDF  
test_smiles, test_molecules, test_activities = data_loader.load_sdf_data(
    'pycomsia/data/ACE_test.sdf',
    activity_property='Activity', 
    is_training=True
)
```

## Dataset Sources

These datasets are commonly used benchmark sets in computational chemistry and drug discovery research for evaluating molecular similarity analysis methods and QSAR/CoMSIA studies.

## Notes

- Molecular coordinates are pre-optimized and aligned for CoMSIA analysis
- Activity values may be in different units (IC₅₀, Ki, etc.) depending on the dataset
- Use appropriate preprocessing and scaling when combining multiple datasets

This directory contains sample additional data you may want to include with your package.
This is a place where non-code related additional information (such as data files, molecular structures,  etc.) can 
go that you want to ship alongside your code.

Please note that it is not recommended to place large files in your git directory. If your project requires files larger
than a few megabytes in size it is recommended to host these files elsewhere. This is especially true for binary files
as the `git` structure is unable to correctly take updates to these files and will store a complete copy of every version
in your `git` history which can quickly add up. As a note most `git` hosting services like GitHub have a 1 GB per repository
cap.

## Including package data

Modify your package's `pyproject.toml` file.
Update the [tool.setuptools.package_data](https://setuptools.pypa.io/en/latest/userguide/datafiles.html#package-data)
and point it at the correct files.
Paths are relative to `package_dir`.

Package data can be accessed at run time with `importlib.resources` or the `importlib_resources` back port.
See https://setuptools.pypa.io/en/latest/userguide/datafiles.html#accessing-data-files-at-runtime
for suggestions.

If modules within your package will access internal data files using
[the recommended approach](https://setuptools.pypa.io/en/latest/userguide/datafiles.html#accessing-data-files-at-runtime),
you may need to include `importlib_resources` in your package dependencies.
In `pyproject.toml`, include the following in your `[project]` table.
```
dependencies = [
    "importlib-resources;python_version<'3.10'",
]
```

## Manifest

* `look_and_say.dat`: first entries of the "Look and Say" integer series, sequence [A005150](https://oeis.org/A005150)
