# PyCoMSIA Test Suite

This comprehensive test suite ensures the reliability and consistency of the PyCoMSIA (Comparative Molecular Similarity Indices Analysis) package. The test suite includes both unit tests and integration tests, with specific regression tests to prevent performance drift over time.

## Test Structure

### Test Files

- **`test_pycomsia.py`** - Main package tests and basic functionality validation
- **`test_data_loader.py`** - Unit tests for the DataLoader module
- **`test_molecular_grid_calculator.py`** - Unit tests for grid generation
- **`test_molecular_field_calculator.py`** - Unit tests for molecular field calculations
- **`test_pls_analysis.py`** - Unit tests for PLS analysis and modeling
- **`test_contour_plot_visualizer.py`** - Unit tests for visualization and contour generation
- **`test_integration.py`** - Integration tests for complete workflows

### Test Categories

#### Unit Tests
Test individual modules and functions in isolation:
- Data loading from SDF files
- Grid parameter calculation
- Molecular field computation 
- PLS model fitting and cross-validation
- Contour range calculation

#### Integration Tests
Test complete workflows and module interactions:
- Full PyCoMSIA analysis pipeline
- Multi-dataset consistency
- Train/test split workflows
- Visualization integration

#### Regression Tests
Prevent drift in critical outputs by testing:
- Grid dimensions for standard parameters
- Field calculation statistics
- PLS model performance metrics
- Contour coordinate consistency
- Activity prediction accuracy

## Running Tests

### Quick Start

```bash
# Run all tests
python run_tests.py

# Run only fast tests (exclude slow integration tests)
python run_tests.py --type fast -v

# Run with coverage report
python run_tests.py --coverage
```

### Test Types

```bash
# Unit tests only
python run_tests.py --type unit

# Integration tests only  
python run_tests.py --type integration

# Regression tests only
python run_tests.py --type regression

# Fast tests (exclude slow tests)
python run_tests.py --type fast
```

### Using pytest directly

```bash
# Run all tests with verbose output
pytest -v pycomsia/tests/

# Run specific test file
pytest pycomsia/tests/test_data_loader.py

# Run tests matching pattern
pytest -k "test_ace" pycomsia/tests/

# Run with coverage
pytest --cov=pycomsia --cov-report=html pycomsia/tests/
```

## Test Data

The test suite uses molecular datasets in `pycomsia/data/`:

- **ACE (Angiotensin-Converting Enzyme)** - Primary test dataset
- **AChE (Acetylcholinesterase)** - Secondary test dataset  
- **CCR5, STEROIDS, THR, etc.** - Additional datasets for consistency testing

These datasets contain pre-aligned molecular structures with biological activity data, ensuring reproducible test results.

## Key Test Cases

### Data Loading Tests
- **Molecule count validation**: Ensures expected number of molecules loaded
- **Activity range checks**: Validates activity values are in reasonable ranges
- **Hydrogen addition**: Verifies molecules have proper hydrogen atoms
- **SMILES consistency**: Tests reproducible SMILES generation

### Grid Calculation Tests  
- **Dimension stability**: Tests that grid dimensions are stable for given parameters
- **Resolution effects**: Validates that changing resolution affects grid appropriately
- **Padding impact**: Ensures padding correctly affects grid size
- **Coordinate ranges**: Checks grid origin coordinates are reasonable

### Field Calculation Tests
- **Field value ranges**: Validates field values are in expected ranges
- **All field types**: Tests steric, electrostatic, hydrophobic, and H-bond fields
- **Deterministic behavior**: Ensures repeated calculations give identical results
- **Field statistics**: Tests mean, std, and distribution of field values

### PLS Analysis Tests
- **Model performance**: Validates R² values are in expected ranges  
- **Coefficient extraction**: Tests PLS coefficient reconstruction
- **Cross-validation**: Verifies leave-one-out CV gives reasonable Q² values
- **Field contributions**: Tests contribution fraction calculations

### Visualization Tests
- **Contour range calculation**: Tests significant range determination
- **Color scheme consistency**: Validates field-specific color mappings
- **Coordinate generation**: Tests contour point coordinate calculation
- **PyMOL integration**: Validates session file creation structure

### Integration Tests
- **Full workflow**: Tests complete analysis pipeline end-to-end
- **Multi-dataset**: Validates consistent behavior across different datasets
- **Train/test splits**: Tests workflows with separate training and test sets
- **Error handling**: Tests behavior with edge cases and invalid inputs

## Expected Performance Baselines

### ACE Dataset (Standard Parameters: resolution=2.0, padding=4.0)
- **Grid dimensions**: (12, 13, 11) 
- **Training R²**: 0.7 - 0.95
- **Test R²**: 0.4 - 0.8  
- **Optimal components**: 2 - 6
- **Molecule count**: 76

These baselines help detect regressions in analysis quality.

## Regression Test Philosophy

The test suite emphasizes **regression prevention** by:

1. **Exact output testing**: Key numerical outputs are tested for stability
2. **Statistical bounds**: Performance metrics must stay within expected ranges  
3. **Deterministic seeding**: Random components use fixed seeds for reproducibility
4. **Cross-dataset validation**: Multiple datasets ensure algorithm robustness
5. **Version comparison**: Tests can detect changes between code versions

## Adding New Tests

When adding new functionality:

1. **Add unit tests** for the new module/function
2. **Add integration tests** if the functionality affects the full workflow
3. **Add regression tests** for any outputs that should remain stable
4. **Update baselines** if intentional changes affect expected outputs
5. **Document test purposes** clearly in docstrings

### Test Naming Convention

```python
def test_module_functionality():           # Basic functionality test
    """Test basic module functionality."""

def test_module_edge_case():              # Edge case test
    """Test module behavior with edge cases."""
    
def test_module_exact_output():           # Regression test
    """Test exact output for regression detection (regression test)."""
```

## Dependencies

Test dependencies are minimal:
- **pytest** - Test framework
- **numpy** - Numerical testing
- **rdkit** - Already required by PyCoMSIA
- **pathlib** - File handling (standard library)
- **tempfile** - Temporary directories (standard library)
- **unittest.mock** - Mocking external dependencies (standard library)

## Continuous Integration

The test suite is designed to work in CI environments:
- **No external network requirements** - All test data is included
- **Predictable execution time** - Fast tests run in <1 minute, full suite in <5 minutes  
- **Clear failure reporting** - Failed tests provide specific error messages
- **Environment independence** - Tests work across Python versions and platforms

## Troubleshooting

### Common Issues

**"ACE_train.sdf not found"**
- Ensure you're running from the PyCoMSIA root directory
- Check that `pycomsia/data/ACE_train.sdf` exists

**"ImportError: No module named 'pycomsia'"**  
- Install PyCoMSIA in development mode: `pip install -e .`
- Or add PyCoMSIA to PYTHONPATH

**"Tests are slow"**
- Use `--type fast` to exclude slow integration tests
- Run specific test files instead of the full suite

**"Regression test failures"**
- Check if changes are intentional and update baselines if needed
- Ensure random seeds haven't changed
- Verify test data hasn't been modified

### Getting Help

For test-related issues:
1. Check test output for specific error messages
2. Run individual test files to isolate problems  
3. Use `-v` flag for verbose output
4. Check that all dependencies are properly installed

## Coverage Goals

The test suite aims for:
- **>90% line coverage** for core modules
- **>80% branch coverage** for complex logic
- **100% coverage** for critical regression tests
- **Representative test data** covering typical use cases

Run `python run_tests.py --coverage` to generate coverage reports.